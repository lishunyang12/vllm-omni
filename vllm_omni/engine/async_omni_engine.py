"""
Async Omni Engine for vLLM-Omni V1 architecture.

AsyncOmniEngine in the caller's thread is a thin proxy that communicates
with the Orchestrator (running in a background thread) via janus queues.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import janus
from omegaconf import OmegaConf
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.entrypoints.utils import (
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)

logger = init_logger(__name__)


def _inject_global_id(target: Any, request_id: str) -> None:
    """Inject global_request_id into a prompt dict's additional_information."""
    if isinstance(target, dict):
        if "additional_information" not in target:
            target["additional_information"] = {}
        if target["additional_information"] is None:
            target["additional_information"] = {}
        if isinstance(target["additional_information"], dict):
            target["additional_information"]["global_request_id"] = [str(request_id)]


@dataclass
class _PendingLlmStageLaunch:
    stage_id: int
    metadata: Any
    vllm_config: Any
    executor_class: type
    launch_cm: Any
    engine_manager: Any
    coordinator: Any
    addresses: Any
    lock_fds: list[int]
    lock_released: bool = False


class AsyncOmniEngine:
    """Thin proxy that launches an Orchestrator in a background thread.

    All stage clients, input/output processors, and stage-to-stage transfer
    logic live inside the Orchestrator coroutine (running in its own thread
    with a dedicated asyncio event loop). This class communicates with it
    via janus queues (sync side for callers, async side for orchestrator).

    Args:
        model: Model name or path
        stage_configs: List of stage configurations. If None, loads from model.
        stage_configs_path: Path to YAML file with stage configs.
        stage_init_timeout: Timeout for stage initialization (seconds)
        **kwargs: Additional arguments
    """

    def _load_omni_transfer_config(self) -> Any:
        """Load transfer config used for stage connector extraction."""
        from vllm_omni.distributed.omni_connectors import load_omni_transfer_config

        try:
            config_path = self.config_path or resolve_model_config_path(self.model)
            return load_omni_transfer_config(config_path)
        except Exception as e:
            logger.warning("[AsyncOmniEngine] Failed to load transfer config: %s", e)
            return None

    def _get_stage_connector_spec(
        self,
        omni_transfer_config: Any,
        stage_id: int,
        async_chunk: bool,
    ) -> dict[str, Any]:
        """Return the first connector spec for the stage when async chunking is enabled."""
        from vllm_omni.distributed.omni_connectors import get_stage_connector_config

        if not async_chunk:
            return {}

        stage_connectors_cfg = get_stage_connector_config(omni_transfer_config, stage_id)
        for cfg in stage_connectors_cfg.values():
            return dict(cfg.get("spec", {}))
        return {}

    def _initialize_diffusion_stage(self, stage_cfg: Any, metadata: Any) -> Any:
        """Build a diffusion stage client inside the orchestrator thread."""
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
        from vllm_omni.entrypoints.stage_utils import _to_dict

        od_config = OmniDiffusionConfig.from_kwargs(
            model=self.model,
            **_to_dict(stage_cfg.engine_args),
        )
        return StageDiffusionClient(self.model, od_config, metadata)

    def _launch_llm_stage(
        self,
        stage_cfg: Any,
        metadata: Any,
        stage_connector_spec: dict[str, Any],
        stage_init_timeout: int,
    ) -> _PendingLlmStageLaunch:
        """Start stage engine processes early but defer client attachment."""
        from vllm.v1.engine.utils import get_engine_zmq_addresses, launch_core_engines

        from vllm_omni.engine.stage_init import (
            acquire_device_locks,
            build_vllm_config,
            release_device_locks,
        )
        from vllm_omni.entrypoints.stage_utils import _to_dict

        vllm_config, executor_class = build_vllm_config(
            stage_cfg,
            self.model,
            stage_connector_spec=stage_connector_spec,
        )
        engine_args_dict = _to_dict(stage_cfg.engine_args)
        engine_args_dict["model"] = self.model
        lock_fds = acquire_device_locks(metadata.stage_id, engine_args_dict, stage_init_timeout)
        try:
            addresses = get_engine_zmq_addresses(vllm_config)
            launch_cm = launch_core_engines(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=False,
                addresses=addresses,
            )
            engine_manager, coordinator, addresses = launch_cm.__enter__()
        except Exception:
            release_device_locks(lock_fds)
            raise

        logger.info("[AsyncOmniEngine] Stage %s engine launch started", metadata.stage_id)
        return _PendingLlmStageLaunch(
            stage_id=metadata.stage_id,
            metadata=metadata,
            vllm_config=vllm_config,
            executor_class=executor_class,
            launch_cm=launch_cm,
            engine_manager=engine_manager,
            coordinator=coordinator,
            addresses=addresses,
            lock_fds=lock_fds,
        )

    def _release_llm_stage_locks(self, launched: _PendingLlmStageLaunch) -> None:
        """Release device locks for a launched LLM stage once attachment begins."""
        from vllm_omni.engine.stage_init import release_device_locks

        if launched.lock_released:
            return
        release_device_locks(launched.lock_fds)
        launched.lock_released = True

    def _close_pending_llm_stage(self, launched: _PendingLlmStageLaunch) -> None:
        """Close managers owned by a launched stage that never attached."""
        resources = (
            ("engine manager", launched.engine_manager),
            ("coordinator", launched.coordinator),
        )
        for resource_name, resource in resources:
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as cleanup_error:
                logger.warning(
                    "[AsyncOmniEngine] Failed to close launched %s for stage %s: %s",
                    resource_name,
                    launched.stage_id,
                    cleanup_error,
                )

    def _attach_llm_stage(
        self,
        launched: _PendingLlmStageLaunch,
    ) -> tuple[Any, Any, Any, InputProcessor | None]:
        """Wait for startup, then attach the AsyncMPClient on the orchestrator loop."""
        from vllm.tokenizers import cached_tokenizer_from_config

        from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

        try:
            launched.launch_cm.__exit__(None, None, None)
        finally:
            self._release_llm_stage_locks(launched)

        client_addresses = {
            "input_address": launched.addresses.inputs[0],
            "output_address": launched.addresses.outputs[0],
        }
        if launched.addresses.frontend_stats_publish_address is not None:
            client_addresses["stats_update_address"] = launched.addresses.frontend_stats_publish_address

        try:
            stage_client = StageEngineCoreClient(
                vllm_config=launched.vllm_config,
                executor_class=launched.executor_class,
                metadata=launched.metadata,
                client_addresses=client_addresses,
                engine_manager=launched.engine_manager,
                coordinator=launched.coordinator,
            )
        except Exception:
            self._close_pending_llm_stage(launched)
            raise

        try:
            if launched.vllm_config.model_config.skip_tokenizer_init:
                tokenizer = None
            else:
                tokenizer = cached_tokenizer_from_config(
                    model_config=launched.vllm_config.model_config,
                )
            output_processor = MultimodalOutputProcessor(
                tokenizer=tokenizer,
                log_stats=False,
                engine_core_output_type=launched.metadata.engine_output_type,
            )
            input_processor = None
            if launched.stage_id == 0:
                input_processor = InputProcessor(vllm_config=launched.vllm_config)
        except Exception:
            try:
                stage_client.shutdown()
            except Exception as cleanup_error:
                logger.warning(
                    "[AsyncOmniEngine] Failed to cleanup stage %s after attach failure: %s",
                    launched.stage_id,
                    cleanup_error,
                )
            raise

        logger.info("[AsyncOmniEngine] Stage %s initialized", launched.stage_id)
        return stage_client, output_processor, launched.vllm_config, input_processor

    def _finalize_initialized_stages(
        self,
        stage_clients: list[Any | None],
        input_processor: InputProcessor | None,
    ) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
        """Validate successful init and build runtime metadata lists."""
        if any(stage_client is None for stage_client in stage_clients):
            raise RuntimeError("Stage initialization completed with missing stage clients")

        initialized_stage_clients = [stage_client for stage_client in stage_clients if stage_client is not None]
        default_sampling_params_list = [
            stage_client.default_sampling_params for stage_client in initialized_stage_clients
        ]
        stage_metadata = [
            {
                "final_output": stage_client.final_output,
                "final_output_type": stage_client.final_output_type,
                "stage_type": stage_client.stage_type,
            }
            for stage_client in initialized_stage_clients
        ]

        if not isinstance(input_processor, InputProcessor):
            has_llm_stage = any(metadata.get("stage_type") != "diffusion" for metadata in stage_metadata)
            if has_llm_stage:
                raise RuntimeError("Failed to initialize stage-0 InputProcessor for LLM pipeline")

        return initialized_stage_clients, default_sampling_params_list, stage_metadata

    def _cleanup_failed_stage_initialization(
        self,
        stage_clients: list[Any | None],
        llm_launches: list[_PendingLlmStageLaunch],
    ) -> None:
        """Shutdown attached stages and close any launched-but-unattached engines."""
        for cleanup_stage_id, stage_client in reversed(list(enumerate(stage_clients))):
            if stage_client is None:
                continue
            try:
                stage_client.shutdown()
            except Exception as cleanup_error:
                logger.warning(
                    "[AsyncOmniEngine] Failed to shutdown initialized stage %s after init failure: %s",
                    cleanup_stage_id,
                    cleanup_error,
                )

        for launched in reversed(llm_launches):
            if stage_clients[launched.stage_id] is not None:
                continue
            try:
                self._release_llm_stage_locks(launched)
            except Exception:
                pass
            self._close_pending_llm_stage(launched)

    def _initialize_stages(self, stage_init_timeout: int) -> None:
        """Initialize stage clients/processors in orchestrator thread and assign to self."""
        from vllm_omni.engine.stage_init import (
            extract_stage_metadata,
            prepare_engine_environment,
            setup_stage_devices,
        )

        num_stages = len(self.stage_configs)
        stage_clients: list[Any | None] = [None] * num_stages
        output_processors: list[Any | None] = [None] * num_stages
        stage_vllm_configs: list[Any | None] = [None] * num_stages
        input_processor: InputProcessor | None = None
        llm_launches: list[_PendingLlmStageLaunch] = []

        stage0_args = getattr(self.stage_configs[0], "engine_args", None) if num_stages > 0 else None
        async_chunk = bool(getattr(stage0_args, "async_chunk", False))

        prepare_engine_environment()
        omni_transfer_config = self._load_omni_transfer_config()

        try:
            for stage_id, stage_cfg in enumerate(self.stage_configs):
                logger.info("[AsyncOmniEngine] Initializing stage %s", stage_id)
                metadata = extract_stage_metadata(stage_cfg)
                setup_stage_devices(stage_id, metadata.runtime_cfg)

                stage_connector_spec = self._get_stage_connector_spec(
                    omni_transfer_config=omni_transfer_config,
                    stage_id=stage_id,
                    async_chunk=async_chunk,
                )

                if metadata.stage_type == "diffusion":
                    stage_clients[stage_id] = self._initialize_diffusion_stage(stage_cfg, metadata)
                    logger.info("[AsyncOmniEngine] Stage %s initialized (diffusion)", stage_id)
                    continue

                llm_launches.append(
                    self._launch_llm_stage(
                        stage_cfg=stage_cfg,
                        metadata=metadata,
                        stage_connector_spec=stage_connector_spec,
                        stage_init_timeout=stage_init_timeout,
                    )
                )

            for launched in llm_launches:
                stage_client, output_processor, vllm_config, stage0_input_processor = self._attach_llm_stage(launched)
                stage_clients[launched.stage_id] = stage_client
                output_processors[launched.stage_id] = output_processor
                stage_vllm_configs[launched.stage_id] = vllm_config
                if stage0_input_processor is not None:
                    input_processor = stage0_input_processor

            initialized_stage_clients, default_sampling_params_list, stage_metadata = self._finalize_initialized_stages(
                stage_clients,
                input_processor,
            )
        except Exception:
            logger.exception(
                "[AsyncOmniEngine] Stage initialization failed; shutting down %s initialized stage(s)",
                len([stage_client for stage_client in stage_clients if stage_client is not None]),
            )
            self._cleanup_failed_stage_initialization(stage_clients, llm_launches)
            raise

        self.async_chunk = async_chunk
        self.stage_clients = initialized_stage_clients
        self.output_processors = output_processors
        self.stage_vllm_configs = stage_vllm_configs
        self.input_processor = input_processor
        # TODO(Peiqi): Hack here
        supported_tasks: set[str] = set()
        if any(getattr(stage_client, "is_comprehension", False) for stage_client in initialized_stage_clients):
            supported_tasks.add("generate")
        if any(metadata.get("final_output_type") == "audio" for metadata in stage_metadata):
            supported_tasks.add("speech")
        self.supported_tasks = tuple(supported_tasks) if supported_tasks else ("generate",)

        self.default_sampling_params_list = default_sampling_params_list
        self.stage_metadata = stage_metadata
        self.num_stages = len(stage_metadata)

    def _initialize_janus_queues(self) -> None:
        """Initialize janus queues inside orchestrator thread loop context."""
        self.request_queue = janus.Queue()
        self.output_queue = janus.Queue()
        self.rpc_output_queue = janus.Queue()
        logger.debug("[AsyncOmniEngine] janus queues initialized in orchestrator thread loop")

    def _bootstrap_orchestrator(
        self,
        stage_init_timeout: int,
        startup_future: concurrent.futures.Future,
    ) -> None:
        """Create loop, initialize stages, then run Orchestrator."""
        from vllm_omni.engine.orchestrator import Orchestrator

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run_orchestrator() -> None:
            self._initialize_janus_queues()

            self._initialize_stages(stage_init_timeout)
            orchestrator = Orchestrator(
                request_async_queue=self.request_queue.async_q,
                output_async_queue=self.output_queue.async_q,
                rpc_async_queue=self.rpc_output_queue.async_q,
                async_chunk=self.async_chunk,
                stage_clients=self.stage_clients,
                output_processors=self.output_processors,
                stage_vllm_configs=self.stage_vllm_configs,
            )
            if not startup_future.done():
                startup_future.set_result(asyncio.get_running_loop())
            await orchestrator.run()

        try:
            loop.run_until_complete(_run_orchestrator())
        except Exception as e:
            if not startup_future.done():
                startup_future.set_exception(RuntimeError(f"Orchestrator initialization failed: {e}"))
            logger.exception("[AsyncOmniEngine] Orchestrator thread crashed")
            try:
                if self.output_queue is not None:
                    self.output_queue.sync_q.put_nowait({"type": "error", "error": "Orchestrator thread crashed"})
                if self.rpc_output_queue is not None:
                    self.rpc_output_queue.sync_q.put_nowait({"type": "error", "error": "Orchestrator thread crashed"})
            except Exception:
                pass
            raise
        finally:
            try:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                if hasattr(loop, "shutdown_default_executor"):
                    loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                logger.exception("[AsyncOmniEngine] Failed during orchestrator loop cleanup")
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    def __init__(
        self,
        model: str,
        stage_configs: list[Any] | None = None,
        stage_configs_path: str | None = None,
        stage_init_timeout: int = 300,
        **kwargs: Any,
    ) -> None:
        self.model = model

        logger.info(f"[AsyncOmniEngine] Initializing with model {model}")

        # --- Resolve stage configs (same logic as before) ---
        tokenizer = kwargs.get("tokenizer", None)
        base_engine_args = {"tokenizer": tokenizer} if tokenizer is not None else None

        if stage_configs is not None:
            self.config_path = stage_configs_path
            # Keep caller-provided structured configs as-is.
            resolved_configs = stage_configs
            if isinstance(resolved_configs, list) and resolved_configs and isinstance(resolved_configs[0], dict):
                resolved_configs = OmegaConf.create(resolved_configs)
        elif stage_configs_path is None:
            self.config_path = resolve_model_config_path(model)
            resolved_configs = load_stage_configs_from_model(model, base_engine_args=base_engine_args)
            if not resolved_configs:
                default_stage_cfg = self._create_default_diffusion_stage_cfg(kwargs)
                resolved_configs = OmegaConf.create(default_stage_cfg)
        else:
            self.config_path = stage_configs_path
            resolved_configs = load_stage_configs_from_yaml(stage_configs_path, base_engine_args=base_engine_args)

        self.stage_configs = resolved_configs
        self.num_stages = len(self.stage_configs)
        stage0_args = getattr(self.stage_configs[0], "engine_args", None) if self.num_stages > 0 else None
        self.async_chunk = bool(getattr(stage0_args, "async_chunk", False))
        self.stage_clients: list[Any] = []
        self.stage_vllm_configs: list[Any] = []
        self.output_processors: list[MultimodalOutputProcessor | None] = []
        self.input_processor: InputProcessor | None = None
        self.supported_tasks: tuple[str, ...] = ("generate",)
        self.default_sampling_params_list: list[Any] = []
        self.stage_metadata: list[dict[str, Any]] = []
        self.request_queue: janus.Queue[dict[str, Any]] | None = None
        self.output_queue: janus.Queue[dict[str, Any]] | None = None
        self.rpc_output_queue: janus.Queue[dict[str, Any]] | None = None
        self._rpc_lock = threading.Lock()

        logger.info(f"[AsyncOmniEngine] Launching Orchestrator thread with {self.num_stages} stages")

        # Launch orchestrator background thread
        startup_future: concurrent.futures.Future = concurrent.futures.Future()

        self.orchestrator_thread = threading.Thread(
            target=self._bootstrap_orchestrator,
            args=(
                stage_init_timeout,
                startup_future,
            ),
            daemon=True,
            name="orchestrator",
        )
        self.orchestrator_thread.start()

        # Wait for stage/runtime initialization result from orchestrator thread.
        try:
            startup_future.result(timeout=stage_init_timeout)
        except concurrent.futures.TimeoutError as e:
            try:
                self.shutdown()
            except Exception:
                logger.exception("[AsyncOmniEngine] Failed to cleanup after orchestrator startup timeout")
            raise TimeoutError(f"Orchestrator did not become ready within {stage_init_timeout}s") from e
        except Exception:
            try:
                self.shutdown()
            except Exception:
                logger.exception("[AsyncOmniEngine] Failed to cleanup after orchestrator startup failure")
            raise

        # Stage runtime fields are assigned directly on self by the bootstrap thread.
        self.num_stages = len(self.stage_metadata)

        logger.info(f"[AsyncOmniEngine] Orchestrator ready with {self.num_stages} stages")

    # ---- request helpers ----

    def _build_add_request_message(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
    ) -> dict[str, Any]:
        """Build an add_request message after stage-0 preprocessing."""
        effective_sampling_params_list = (
            list(sampling_params_list) if sampling_params_list is not None else list(self.default_sampling_params_list)
        )
        if not effective_sampling_params_list:
            raise ValueError(
                f"Missing sampling params for stage 0. Got {len(effective_sampling_params_list)} stage params."
            )
        params = effective_sampling_params_list[0]

        # Keep the original prompt for downstream stages (they need the raw
        # dict, e.g. for multi_modal_data).
        original_prompt = prompt

        stage_type = self.stage_metadata[0].get("stage_type")
        if stage_type != "diffusion" and not isinstance(prompt, EngineCoreRequest):
            # Inject global_request_id into the raw prompt.
            if isinstance(prompt, dict):
                _inject_global_id(prompt, request_id)
            elif isinstance(prompt, list):
                for item in prompt:
                    _inject_global_id(item, request_id)

            # Full input processing (tokenization, multimodal, etc.)
            request = self.input_processor.process_inputs(
                request_id=request_id,
                prompt=prompt,
                params=params,
                supported_tasks=self.supported_tasks,
                arrival_time=arrival_time,
            )
            # Restore external_req_id to the original user-facing request_id.
            # InputProcessor.process_inputs() renames request_id to an internal
            # UUID (saving the original in external_req_id), but then overwrites
            # external_req_id with the new internal ID. We need external_req_id
            # to match the key used in Orchestrator.request_states so that
            # output routing (output.request_id lookup) can find the req_state.
            request.external_req_id = request_id

            # Register with stage 0's output processor.
            self.output_processors[0].add_request(
                request=request,
                prompt=prompt,
                parent_req=None,
                request_index=0,
                queue=None,
            )
            prompt = request

        return {
            "type": "add_request",
            "request_id": request_id,
            "prompt": prompt,
            "original_prompt": original_prompt,
            "sampling_params_list": effective_sampling_params_list,
            "final_stage_id": final_stage_id,
        }

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

    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
    ) -> None:
        """Process stage-0 input locally, then send to the Orchestrator.

        Input processing and output
        processor registration happen here in the caller's thread, avoiding
        a queue + coroutine-switch round-trip.  The Orchestrator receives a
        ready-to-submit OmniEngineCoreRequest.
        """
        msg = self._build_add_request_message(
            request_id=request_id,
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            arrival_time=arrival_time,
        )
        if self.request_queue is None:
            raise RuntimeError("request_queue is not initialized")
        self.request_queue.sync_q.put_nowait(msg)

    async def add_request_async(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
    ) -> None:
        """Async add_request API."""
        self.add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            arrival_time=arrival_time,
        )

    def try_get_output(self, timeout: float = 0.001) -> dict[str, Any] | None:
        """Read one output message from the Orchestrator output queue."""
        if self.output_queue is None:
            return None
        try:
            return self.output_queue.sync_q.get(timeout=timeout)
        except queue.Empty:
            return None

    async def try_get_output_async(self) -> dict[str, Any] | None:
        """Async read from the Orchestrator output queue."""
        if self.output_queue is None:
            return None
        try:
            return self.output_queue.sync_q.get_nowait()
        except queue.Empty:
            return None

    def get_stage_metadata(self, stage_id: int) -> dict[str, Any]:
        """Get cached metadata for a stage."""
        return self.stage_metadata[stage_id]

    def abort(self, request_ids: list[str]) -> None:
        """Send abort message to the Orchestrator."""
        if self.request_queue is None:
            raise RuntimeError("request_queue is not initialized")
        self.request_queue.sync_q.put_nowait(
            {
                "type": "abort",
                "request_ids": request_ids,
            }
        )

    async def abort_async(self, request_ids: list[str]) -> None:
        """Async abort API."""
        self.abort(request_ids)

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        stage_ids: list[int] | None = None,
    ) -> list[Any]:
        """Send a control RPC to the Orchestrator and wait for aggregated results.

        This uses a dedicated RPC output queue so control-plane messages do not
        race with the normal request output polling loop.
        """
        if self.request_queue is None:
            raise RuntimeError("request_queue is not initialized")
        if self.rpc_output_queue is None:
            raise RuntimeError("rpc_output_queue is not initialized")

        rpc_id = uuid.uuid4().hex
        msg = {
            "type": "collective_rpc",
            "rpc_id": rpc_id,
            "method": method,
            "args": tuple(args),
            "kwargs": kwargs or {},
            "stage_ids": stage_ids,
        }

        with self._rpc_lock:
            self.request_queue.sync_q.put_nowait(msg)
            deadline = None if timeout is None else time.monotonic() + timeout

            while True:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                try:
                    result_msg = self.rpc_output_queue.sync_q.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError(f"collective_rpc timed out after {timeout} seconds") from exc

                if result_msg.get("type") == "error":
                    raise RuntimeError(result_msg.get("error", "Orchestrator returned an error message"))

                if result_msg.get("type") != "collective_rpc_result":
                    logger.warning(
                        "[AsyncOmniEngine] Dropping unexpected rpc queue message type=%s",
                        result_msg.get("type"),
                    )
                    continue

                if result_msg.get("rpc_id") != rpc_id:
                    logger.warning(
                        "[AsyncOmniEngine] Dropping mismatched rpc result rpc_id=%s expected=%s",
                        result_msg.get("rpc_id"),
                        rpc_id,
                    )
                    continue

                return list(result_msg.get("results", []))

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        stage_ids: list[int] | None = None,
    ) -> list[Any]:
        """Async wrapper around collective_rpc()."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.collective_rpc(
                method=method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
                stage_ids=stage_ids,
            ),
        )

    def shutdown(self) -> None:
        """Send shutdown message and wait for the Orchestrator thread to exit."""
        logger.info("[AsyncOmniEngine] Shutting down Orchestrator")
        try:
            if self.request_queue is not None:
                self.request_queue.sync_q.put_nowait({"type": "shutdown"})
        except Exception:
            pass
        if hasattr(self, "orchestrator_thread") and self.orchestrator_thread.is_alive():
            self.orchestrator_thread.join(timeout=10)
            if self.orchestrator_thread.is_alive():
                logger.warning("[AsyncOmniEngine] Orchestrator thread did not exit in time")

        for q in (self.request_queue, self.output_queue, self.rpc_output_queue):
            if q is None:
                continue
            try:
                q.close()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
