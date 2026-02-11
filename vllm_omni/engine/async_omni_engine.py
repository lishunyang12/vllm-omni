"""
Async Omni Engine for vLLM-Omni V1 architecture.

Manages multiple stages with StageAsyncCoreClient.
The Orchestrator runs as a coroutine in a background thread (with its own
asyncio event loop).  AsyncOmniEngine in the caller's thread is a thin
proxy that communicates with the Orchestrator via asyncio.Queues.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf
from vllm.config import VllmConfig
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.engine.stage_async_core_client import StageAsyncCoreClient
from vllm_omni.engine.stage_init import (
    acquire_device_locks,
    build_vllm_config,
    extract_stage_metadata,
    prepare_engine_environment,
    release_device_locks,
    setup_stage_devices,
)
from vllm_omni.entrypoints.utils import (
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)

logger = init_logger(__name__)


# ============================================================
# Orchestrator internals (run inside the background thread)
# ============================================================

@dataclass
class _OrchestratorRequestState:
    """Per-request bookkeeping inside the Orchestrator."""

    request_id: str
    prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = 0


class _Orchestrator:
    """Runs inside a background thread's asyncio event loop.

    Owns all StageAsyncCoreClient instances, input/output processors,
    and handles stage-to-stage transfer logic.
    """

    def __init__(
        self,
        model: str,
        stage_configs: Any,
        request_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stage_init_timeout: int = 300,
        log_requests: bool = True,
    ) -> None:
        self.model = model
        self.stage_configs = stage_configs
        self.request_queue = request_queue
        self.output_queue = output_queue
        self.stage_init_timeout = stage_init_timeout
        self.log_requests = log_requests

        self.num_stages = len(stage_configs)

        # Will be populated by _initialize_stages
        self.stage_clients: list[StageAsyncCoreClient] = []
        self.input_processors: list[OmniInputProcessor] = []
        self.output_processors: list[MultimodalOutputProcessor] = []
        self.stage_tokenizers: list[TokenizerLike] = []
        self.stage_vllm_configs: list[VllmConfig] = []
        self.connectors: dict[tuple[str, str], Any] = {}

        # Per-request state
        self.request_states: dict[str, _OrchestratorRequestState] = {}

        # Shutdown coordination
        self._shutdown_event = asyncio.Event()

        self._initialize_stages()

    def _initialize_stages(self) -> None:
        """Initialize all stage clients and their processors sequentially.

        Performs one-time environment setup, initializes connectors,
        then for each stage: extracts metadata, sets up devices, builds
        vllm config, acquires device locks, creates the client, and
        builds tokenizer / input / output processors.
        """
        from vllm_omni.entrypoints.stage_utils import _to_dict

        # One-time global setup
        prepare_engine_environment()

        # Initialize connectors once
        try:
            config_path = resolve_model_config_path(self.model)
            omni_transfer_config, connectors = initialize_orchestrator_connectors(
                config_path,
                worker_backend="multi_process",
                shm_threshold_bytes=65536,
            )
            self.omni_transfer_config = omni_transfer_config
            self.connectors = connectors
            if connectors:
                logger.info(
                    f"[Orchestrator] Initialized {len(connectors)} connectors"
                )
        except Exception as e:
            logger.warning(f"[Orchestrator] Failed to initialize connectors: {e}")
            self.omni_transfer_config = None
            self.connectors = {}

        # Per-stage initialization
        for stage_id, stage_cfg in enumerate(self.stage_configs):
            logger.info(f"[Orchestrator] Initializing stage {stage_id}")

            metadata = extract_stage_metadata(stage_cfg)
            setup_stage_devices(stage_id, metadata.runtime_cfg)
            vllm_config, executor_class = build_vllm_config(stage_cfg, self.model)

            engine_args_dict = _to_dict(stage_cfg.engine_args)
            engine_args_dict["model"] = self.model
            lock_fds = acquire_device_locks(
                stage_id, engine_args_dict, self.stage_init_timeout
            )

            try:
                stage_client = StageAsyncCoreClient(
                    vllm_config=vllm_config,
                    executor_class=executor_class,
                    metadata=metadata,
                )
            finally:
                release_device_locks(lock_fds)

            if vllm_config.model_config.skip_tokenizer_init:
                tokenizer = None
            else:
                tokenizer = cached_tokenizer_from_config(
                    model_config=vllm_config.model_config
                )

            stage_input_processor = OmniInputProcessor(
                vllm_config=vllm_config,
                tokenizer=tokenizer,
            )

            stage_output_processor = MultimodalOutputProcessor(
                tokenizer=tokenizer,
                log_stats=False,
                engine_core_output_type=metadata.engine_output_type,
            )

            self.stage_clients.append(stage_client)
            self.input_processors.append(stage_input_processor)
            self.output_processors.append(stage_output_processor)
            self.stage_tokenizers.append(tokenizer)
            self.stage_vllm_configs.append(vllm_config)

            logger.info(f"[Orchestrator] Stage {stage_id} initialized")

    async def run(self) -> None:
        """Main entry point for the Orchestrator event loop."""
        # Collect metadata to send back to the main thread
        default_sampling_params_list = [
            sc.default_sampling_params for sc in self.stage_clients
        ]
        stage_metadata = []
        for sc in self.stage_clients:
            stage_metadata.append({
                "final_output": sc.final_output,
                "final_output_type": sc.final_output_type,
                "stage_type": sc.stage_type,
            })

        await self.output_queue.put({
            "type": "ready",
            "num_stages": self.num_stages,
            "default_sampling_params_list": default_sampling_params_list,
            "stage_metadata": stage_metadata,
        })

        logger.info("[Orchestrator] Ready signal sent, starting event loop")

        request_task = asyncio.create_task(
            self._request_handler(), name="orchestrator-request-handler"
        )
        output_task = asyncio.create_task(
            self._output_handler(), name="orchestrator-output-handler"
        )

        try:
            # _request_handler exits on shutdown; once it's done, cancel
            # the output handler so we don't leave dangling wait_for tasks.
            await request_task
        finally:
            output_task.cancel()
            try:
                await output_task
            except asyncio.CancelledError:
                pass

            # Cancel any remaining tasks spawned by wait_for / gather so
            # the event loop can close cleanly without "pending task" errors.
            loop = asyncio.get_running_loop()
            pending = [
                t for t in asyncio.all_tasks(loop)
                if t is not asyncio.current_task() and not t.done()
            ]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _request_handler(self) -> None:
        """Read messages from the main thread via request_queue."""
        while True:
            msg = await self.request_queue.get()
            msg_type = msg.get("type")

            if msg_type == "add_request":
                await self._handle_add_request(msg)
            elif msg_type == "abort":
                await self._handle_abort(msg)
            elif msg_type == "shutdown":
                logger.info("[Orchestrator] Received shutdown signal")
                self._shutdown_event.set()
                self._shutdown_stages()
                break
            else:
                logger.warning(f"[Orchestrator] Unknown message type: {msg_type}")

    async def _output_handler(self) -> None:
        """Poll all stages, handle transfers, send final outputs to main."""
        try:
            await self._output_handler_loop()
        except asyncio.CancelledError:
            logger.debug("[Orchestrator] _output_handler cancelled")
            return

    async def _output_handler_loop(self) -> None:
        """Inner loop for _output_handler (separated for clean cancellation)."""
        while not self._shutdown_event.is_set():
            idle = True
            for stage_id in range(self.num_stages):
                if self._shutdown_event.is_set():
                    return
                try:
                    request_outputs = await asyncio.wait_for(
                        self._get_stage_output(stage_id), timeout=0.001
                    )
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if self._shutdown_event.is_set():
                        return
                    logger.exception(
                        "[Orchestrator] _get_stage_output failed for stage-%s",
                        stage_id,
                    )
                    raise

                if not request_outputs:
                    continue
                idle = False

                for output in request_outputs:
                    req_id = output.request_id
                    req_state = self.request_states.get(req_id)
                    if req_state is None:
                        logger.debug(
                            "[Orchestrator] Dropping output for unknown req %s "
                            "at stage-%s",
                            req_id,
                            stage_id,
                        )
                        continue

                    finished = getattr(output, "finished", False)
                    stage_client = self.stage_clients[stage_id]

                    # 1) If this stage produces final output -> send to main
                    if stage_client.final_output:
                        is_fully_done = (
                            finished
                            and stage_id == req_state.final_stage_id
                        )
                        await self.output_queue.put({
                            "type": "output",
                            "request_id": req_id,
                            "stage_id": stage_id,
                            "engine_outputs": [output],
                            "metrics": None,
                            "finished": is_fully_done,
                        })

                    # 2) If stage finished -> forward to next stage
                    if finished and stage_id < req_state.final_stage_id:
                        await self._forward_to_next_stage(
                            req_id, stage_id, output, req_state
                        )

                    # 3) If all stages done -> cleanup
                    if finished and stage_id == req_state.final_stage_id:
                        self.request_states.pop(req_id, None)

            if idle:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)

    async def _forward_to_next_stage(
        self,
        req_id: str,
        stage_id: int,
        output: Any,
        req_state: _OrchestratorRequestState,
    ) -> None:
        """Execute stage transfer: forward output to the next stage."""
        if not isinstance(output, list):
            engine_outputs = [output]
        else:
            engine_outputs = output

        # Set outputs on current stage client (for cross-stage data flow)
        self.stage_clients[stage_id].set_engine_outputs(engine_outputs)

        next_stage_id = stage_id + 1
        logger.info(
            "[Orchestrator] req=%s forwarding stage-%s -> stage-%s",
            req_id,
            stage_id,
            next_stage_id,
        )

        # Process inputs for next stage
        next_inputs = self.stage_clients[next_stage_id].process_engine_inputs(
            stage_list=self.stage_clients,
            prompt=req_state.prompt,
        )
        logger.info(
            "[Orchestrator] req=%s computed %d next_inputs for stage-%s",
            req_id,
            len(next_inputs),
            next_stage_id,
        )

        # Submit to next stage
        for next_input in next_inputs:
            await self._add_request_to_stage(
                next_stage_id,
                req_id,
                next_input,
                req_state.sampling_params_list[next_stage_id],
            )

    async def _get_stage_output(self, stage_id: int) -> list[Any]:
        """Pull and process outputs from a single stage."""
        stage_client = self.stage_clients[stage_id]
        stage_output_processor = self.output_processors[stage_id]

        outputs = await stage_client.get_output_async()

        processed = stage_output_processor.process_outputs(
            outputs.outputs,
            getattr(outputs, "timestamp", None),
            None,
        )

        if getattr(processed, "reqs_to_abort", None):
            await stage_client.abort_requests_async(processed.reqs_to_abort)

        try:
            if hasattr(outputs, "scheduler_stats"):
                stage_output_processor.update_scheduler_stats(
                    outputs.scheduler_stats
                )
        except Exception:
            pass

        return list(getattr(processed, "request_outputs", []) or [])

    async def _add_request_to_stage(
        self,
        stage_id: int,
        request_id: str,
        prompt: Any,
        params: Any,
        arrival_time: float | None = None,
    ) -> None:
        """Process input and submit a request to a specific stage."""
        stage_client = self.stage_clients[stage_id]
        stage_input_processor = self.input_processors[stage_id]
        stage_output_processor = self.output_processors[stage_id]

        # Inject global_request_id
        def _inject_global_id(target_ein: Any) -> None:
            if isinstance(target_ein, dict):
                if "additional_information" not in target_ein:
                    target_ein["additional_information"] = {}
                if target_ein["additional_information"] is None:
                    target_ein["additional_information"] = {}
                if isinstance(target_ein["additional_information"], dict):
                    target_ein["additional_information"][
                        "global_request_id"
                    ] = [str(request_id)]

        if isinstance(prompt, dict):
            _inject_global_id(prompt)
        elif isinstance(prompt, list):
            for item in prompt:
                _inject_global_id(item)

        # Check connector
        if stage_id > 0 and self.connectors:
            connector_key = (str(stage_id - 1), str(stage_id))
            connector = self.connectors.get(connector_key)
            if connector:
                logger.debug(
                    "[Orchestrator] Stage %s checking connector from stage %s",
                    stage_id,
                    stage_id - 1,
                )

        # Process inputs
        prompt_text: str | None = None
        if not isinstance(prompt, EngineCoreRequest):
            if stage_input_processor is None:
                raise ValueError(
                    f"Stage {stage_id} has no input processor"
                )
            processed = stage_input_processor.process_inputs(
                request_id=request_id,
                prompt=prompt,
                params=params,
                arrival_time=arrival_time,
            )
            request = processed
            prompt_text = prompt
        else:
            request = prompt

        stage_input_processor.assign_request_id(request)
        request.request_id = request_id

        stage_output_processor.add_request(
            request=request,
            prompt=prompt_text,
            parent_req=None,
            request_index=0,
            queue=None,
        )

        await stage_client.add_request_async(request)

        if self.log_requests:
            logger.info(
                "[Orchestrator] Added request %s to stage %s",
                request_id,
                stage_id,
            )

    async def _handle_add_request(self, msg: dict[str, Any]) -> None:
        """Handle an add_request message from the main thread."""
        stage_id = msg["stage_id"]
        request_id = msg["request_id"]
        prompt = msg["prompt"]
        params = msg["params"]
        sampling_params_list = msg["sampling_params_list"]
        final_stage_id = msg["final_stage_id"]

        # Track request state
        self.request_states[request_id] = _OrchestratorRequestState(
            request_id=request_id,
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
        )

        await self._add_request_to_stage(stage_id, request_id, prompt, params)

    async def _handle_abort(self, msg: dict[str, Any]) -> None:
        """Handle an abort message from the main thread."""
        request_ids = msg["request_ids"]
        for stage_id in range(self.num_stages):
            await self.stage_clients[stage_id].abort_requests_async(request_ids)
        for req_id in request_ids:
            self.request_states.pop(req_id, None)
        logger.info("[Orchestrator] Aborted request(s) %s", request_ids)

    def _shutdown_stages(self) -> None:
        """Shutdown all stage clients."""
        logger.info("[Orchestrator] Shutting down all stages")
        for stage_id, stage_client in enumerate(self.stage_clients):
            try:
                stage_client.shutdown()
                logger.info(f"[Orchestrator] Stage {stage_id} shut down")
            except Exception as e:
                logger.warning(
                    f"[Orchestrator] Failed to shutdown stage {stage_id}: {e}"
                )


def _run_orchestrator_in_thread(
    model: str,
    stage_configs: Any,
    request_queue: asyncio.Queue,
    output_queue: asyncio.Queue,
    stage_init_timeout: int,
    log_requests: bool,
    loop_ready: threading.Event,
    loop_holder: list,
) -> None:
    """Top-level entry point for the Orchestrator background thread.

    Creates a fresh asyncio event loop and runs the _Orchestrator.
    Exposes the loop via *loop_holder* so the main thread can schedule
    work on it (e.g. queue.put via call_soon_threadsafe).
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop_holder.append(loop)
    loop_ready.set()

    try:
        orchestrator = _Orchestrator(
            model=model,
            stage_configs=stage_configs,
            request_queue=request_queue,
            output_queue=output_queue,
            stage_init_timeout=stage_init_timeout,
            log_requests=log_requests,
        )
        loop.run_until_complete(orchestrator.run())
    except Exception:
        logger.exception("[Orchestrator] Thread crashed")
        # Signal the main thread so it doesn't hang waiting for ready
        try:
            loop.run_until_complete(
                output_queue.put({"type": "error", "error": "Orchestrator thread crashed"})
            )
        except Exception:
            pass
        raise
    finally:
        loop.close()


# ============================================================
# AsyncOmniEngine — thin proxy in the caller's thread
# ============================================================


class AsyncOmniEngine:
    """Thin proxy that launches an Orchestrator in a background thread.

    All stage clients, input/output processors, and stage-to-stage transfer
    logic live inside the Orchestrator coroutine (running in its own thread
    with a dedicated asyncio event loop).  This class communicates with it
    via asyncio.Queues.

    Args:
        model: Model name or path
        stage_configs: List of stage configurations. If None, loads from model.
        stage_configs_path: Path to YAML file with stage configs.
        stage_init_timeout: Timeout for stage initialization (seconds)
        log_requests: Whether to log requests
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        model: str,
        stage_configs: list[Any] | None = None,
        stage_configs_path: str | None = None,
        stage_init_timeout: int = 300,
        log_requests: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.stage_init_timeout = stage_init_timeout
        self.log_requests = log_requests

        logger.info(f"[AsyncOmniEngine] Initializing with model {model}")
        logger.info(f"[AsyncOmniEngine] stage_configs: {stage_configs}")
        logger.info(f"[AsyncOmniEngine] stage_configs_path: {stage_configs_path}")
        logger.info(f"[AsyncOmniEngine] stage_init_timeout: {stage_init_timeout}")
        logger.info(f"[AsyncOmniEngine] log_requests: {log_requests}")
        logger.info(f"[AsyncOmniEngine] kwargs: {kwargs}")

        # --- Resolve stage configs (same logic as before) ---
        tokenizer = kwargs.get("tokenizer", None)
        base_engine_args = (
            {"tokenizer": tokenizer} if tokenizer is not None else None
        )

        if stage_configs_path is None:
            self.config_path = resolve_model_config_path(model)
            resolved_configs = load_stage_configs_from_model(
                model, base_engine_args=base_engine_args
            )
            if not resolved_configs:
                default_stage_cfg = self._create_default_diffusion_stage_cfg(
                    kwargs
                )
                resolved_configs = OmegaConf.create(default_stage_cfg)
        else:
            self.config_path = stage_configs_path
            resolved_configs = load_stage_configs_from_yaml(
                stage_configs_path, base_engine_args=base_engine_args
            )

        self.stage_configs = resolved_configs
        self.num_stages = len(self.stage_configs)

        logger.info(
            f"[AsyncOmniEngine] Launching Orchestrator thread with "
            f"{self.num_stages} stages"
        )

        # Create asyncio queues (will be used from the orchestrator's loop)
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()

        # Launch orchestrator background thread
        loop_ready = threading.Event()
        self._orch_loop_holder: list[asyncio.AbstractEventLoop] = []

        self.orchestrator_thread = threading.Thread(
            target=_run_orchestrator_in_thread,
            args=(
                model,
                self.stage_configs,
                self.request_queue,
                self.output_queue,
                stage_init_timeout,
                log_requests,
                loop_ready,
                self._orch_loop_holder,
            ),
            daemon=True,
            name="orchestrator",
        )
        self.orchestrator_thread.start()

        # Wait for the orchestrator's event loop to be available
        loop_ready.wait()
        self._orch_loop: asyncio.AbstractEventLoop = self._orch_loop_holder[0]

        # Wait for ready signal from orchestrator (blocking, runs in this thread)
        logger.info("[AsyncOmniEngine] Waiting for Orchestrator ready signal")
        ready_msg = self._blocking_get_output(timeout=stage_init_timeout)
        if ready_msg is None:
            raise TimeoutError(
                f"Orchestrator did not become ready within "
                f"{stage_init_timeout}s"
            )

        if ready_msg.get("type") == "error":
            raise RuntimeError(
                f"Orchestrator failed to start: {ready_msg.get('error')}"
            )

        assert ready_msg["type"] == "ready", (
            f"Expected ready message, got {ready_msg['type']}"
        )

        self.num_stages = ready_msg["num_stages"]
        self.default_sampling_params_list = ready_msg[
            "default_sampling_params_list"
        ]
        self.stage_metadata = ready_msg["stage_metadata"]

        logger.info(
            f"[AsyncOmniEngine] Orchestrator ready with {self.num_stages} stages"
        )

    # ---- helpers for cross-thread queue access ----

    def _put_to_request_queue(self, msg: dict[str, Any]) -> None:
        """Thread-safe put onto the orchestrator's request_queue."""
        self._orch_loop.call_soon_threadsafe(
            self.request_queue.put_nowait, msg
        )

    def _blocking_get_output(self, timeout: float) -> dict[str, Any] | None:
        """Blocking get from the output_queue (used during init)."""
        fut = asyncio.run_coroutine_threadsafe(
            self.output_queue.get(), self._orch_loop
        )
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None

    @staticmethod
    def _create_default_diffusion_stage_cfg(kwargs: dict[str, Any]) -> list:
        """Create a default single-stage diffusion config from kwargs."""
        return [
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "engine_args": kwargs,
                "final_output": True,
                "final_output_type": "image",
            }
        ]

    # ==================== Public API ====================

    async def add_request(
        self,
        stage_id: int,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        params: SamplingParams | PoolingParams,
        sampling_params_list: list[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: dict[str, str] | None = None,
        priority: int = 0,
        prompt_text: str | None = None,
    ) -> None:
        """Send an add_request message to the Orchestrator."""
        self._put_to_request_queue({
            "type": "add_request",
            "stage_id": stage_id,
            "request_id": request_id,
            "prompt": prompt,
            "params": params,
            "sampling_params_list": sampling_params_list or [],
            "final_stage_id": final_stage_id,
        })

    def try_get_output(self) -> dict[str, Any] | None:
        """Non-blocking read from the Orchestrator output_queue."""
        try:
            fut = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(self.output_queue.get(), timeout=0),
                self._orch_loop,
            )
            return fut.result(timeout=0.05)
        except Exception:
            return None

    def try_get_output_blocking(self, timeout: float = 0.05) -> dict[str, Any] | None:
        """Blocking read from the Orchestrator output_queue with timeout."""
        try:
            fut = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(self.output_queue.get(), timeout=timeout),
                self._orch_loop,
            )
            return fut.result(timeout=timeout + 0.1)
        except Exception:
            return None

    def get_stage_metadata(self, stage_id: int) -> dict[str, Any]:
        """Get cached metadata for a stage."""
        return self.stage_metadata[stage_id]

    async def abort(self, request_ids: list[str]) -> None:
        """Send abort message to the Orchestrator."""
        self._put_to_request_queue({
            "type": "abort",
            "request_ids": request_ids,
        })

    def shutdown(self) -> None:
        """Send shutdown message and wait for the Orchestrator thread to exit."""
        logger.info("[AsyncOmniEngine] Shutting down Orchestrator")
        try:
            self._put_to_request_queue({"type": "shutdown"})
        except Exception:
            pass
        if hasattr(self, "orchestrator_thread") and self.orchestrator_thread.is_alive():
            self.orchestrator_thread.join(timeout=10)
            if self.orchestrator_thread.is_alive():
                logger.warning(
                    "[AsyncOmniEngine] Orchestrator thread did not exit in time"
                )

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
