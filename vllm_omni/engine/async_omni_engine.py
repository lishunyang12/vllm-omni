"""
Async Omni Engine for vLLM-Omni V1 architecture.

AsyncOmniEngine in the caller's thread is a thin proxy that communicates
with the Orchestrator (running in a background thread) via asyncio.Queues.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any

from omegaconf import OmegaConf
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
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
        **kwargs: Additional arguments
    """

    def _initialize_stages(self, stage_init_timeout: int) -> None:
        """Initialize stage clients/processors in orchestrator thread and assign to self."""
        from vllm.tokenizers import cached_tokenizer_from_config

        from vllm_omni.distributed.omni_connectors import (
            get_stage_connector_config,
            load_omni_transfer_config,
        )
        from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
        from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
        from vllm_omni.engine.stage_init import (
            acquire_device_locks,
            build_vllm_config,
            extract_stage_metadata,
            prepare_engine_environment,
            release_device_locks,
            setup_stage_devices,
        )
        from vllm_omni.entrypoints.stage_utils import _to_dict

        stage_clients: list[Any] = []
        output_processors: list[Any] = []
        stage_vllm_configs: list[Any] = []
        input_processor: InputProcessor | None = None

        num_stages = len(self.stage_configs)
        stage0_args = getattr(self.stage_configs[0], "engine_args", None) if num_stages > 0 else None
        async_chunk = bool(getattr(stage0_args, "async_chunk", False))

        prepare_engine_environment()

        # TODO(AsyncOmniV1): orchestrator-side connector forwarding is disabled;
        # we only load transfer config for stage_connector_spec extraction.
        try:
            config_path = self.config_path or resolve_model_config_path(self.model)
            omni_transfer_config = load_omni_transfer_config(config_path)
        except Exception as e:
            logger.warning("[AsyncOmniEngine] Failed to load transfer config: %s", e)
            omni_transfer_config = None

        for stage_id, stage_cfg in enumerate(self.stage_configs):
            logger.info("[AsyncOmniEngine] Initializing stage %s", stage_id)

            metadata = extract_stage_metadata(stage_cfg)
            setup_stage_devices(stage_id, metadata.runtime_cfg)
            stage_connector_spec: dict[str, Any] = {}
            if async_chunk:
                stage_connectors_cfg = get_stage_connector_config(omni_transfer_config, stage_id)
                for cfg in stage_connectors_cfg.values():
                    stage_connector_spec = dict(cfg.get("spec", {}))
                    break

            if metadata.stage_type == "diffusion":
                from vllm_omni.diffusion.data import OmniDiffusionConfig

                od_config = OmniDiffusionConfig.from_kwargs(model=self.model, **_to_dict(stage_cfg.engine_args))
                stage_client = StageDiffusionClient(self.model, od_config, metadata)
                stage_clients.append(stage_client)
                output_processors.append(None)
                stage_vllm_configs.append(None)
                logger.info("[AsyncOmniEngine] Stage %s initialized (diffusion)", stage_id)
                continue

            vllm_config, executor_class = build_vllm_config(
                stage_cfg,
                self.model,
                stage_connector_spec=stage_connector_spec,
            )

            engine_args_dict = _to_dict(stage_cfg.engine_args)
            engine_args_dict["model"] = self.model
            lock_fds = acquire_device_locks(stage_id, engine_args_dict, stage_init_timeout)
            try:
                stage_client = StageEngineCoreClient(
                    vllm_config=vllm_config,
                    executor_class=executor_class,
                    metadata=metadata,
                )
            finally:
                release_device_locks(lock_fds)

            if vllm_config.model_config.skip_tokenizer_init:
                tokenizer = None
            else:
                tokenizer = cached_tokenizer_from_config(model_config=vllm_config.model_config)

            if stage_id == 0:
                input_processor = InputProcessor(vllm_config=vllm_config)

            stage_output_processor = MultimodalOutputProcessor(
                tokenizer=tokenizer,
                log_stats=False,
                engine_core_output_type=metadata.engine_output_type,
            )

            stage_clients.append(stage_client)
            output_processors.append(stage_output_processor)
            stage_vllm_configs.append(vllm_config)
            logger.info("[AsyncOmniEngine] Stage %s initialized", stage_id)

        default_sampling_params_list = [sc.default_sampling_params for sc in stage_clients]
        stage_metadata = [
            {
                "final_output": sc.final_output,
                "final_output_type": sc.final_output_type,
                "stage_type": sc.stage_type,
            }
            for sc in stage_clients
        ]

        if not isinstance(input_processor, InputProcessor):
            has_llm_stage = any(m.get("stage_type") != "diffusion" for m in stage_metadata)
            if has_llm_stage:
                raise RuntimeError("Failed to initialize stage-0 InputProcessor for LLM pipeline")

        # Assign initialized runtime directly on engine instance.
        self.async_chunk = async_chunk
        self.stage_clients = stage_clients
        self.output_processors = output_processors
        self.stage_vllm_configs = stage_vllm_configs
        self.input_processor = input_processor
        self.default_sampling_params_list = default_sampling_params_list
        self.stage_metadata = stage_metadata
        self.num_stages = len(stage_metadata)

    def _bootstrap_orchestrator(
        self,
        stage_init_timeout: int,
        startup_future: concurrent.futures.Future,
    ) -> None:
        """Create loop, initialize stages, then run Orchestrator."""
        from vllm_omni.engine.orchestrator import Orchestrator

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            self._initialize_stages(stage_init_timeout)
            orchestrator = Orchestrator(
                request_queue=self.request_queue,
                output_queue=self.output_queue,
                async_chunk=self.async_chunk,
                stage_clients=self.stage_clients,
                output_processors=self.output_processors,
                stage_vllm_configs=self.stage_vllm_configs,
            )
            if not startup_future.done():
                startup_future.set_result(loop)
            loop.run_until_complete(orchestrator.run())
        except Exception as e:
            if not startup_future.done():
                startup_future.set_exception(RuntimeError(f"Orchestrator initialization failed: {e}"))
            logger.exception("[AsyncOmniEngine] Orchestrator thread crashed")
            try:
                loop.run_until_complete(self.output_queue.put({"type": "error", "error": "Orchestrator thread crashed"}))
            except Exception:
                pass
            raise
        finally:
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
        self.default_sampling_params_list: list[Any] = []
        self.stage_metadata: list[dict[str, Any]] = []

        logger.info(f"[AsyncOmniEngine] Launching Orchestrator thread with {self.num_stages} stages")

        # Create asyncio queues (will be used from the orchestrator's loop)
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()

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
        self._orch_loop: asyncio.AbstractEventLoop
        try:
            self._orch_loop = startup_future.result(timeout=stage_init_timeout)
        except concurrent.futures.TimeoutError as e:
            raise TimeoutError(f"Orchestrator did not become ready within {stage_init_timeout}s") from e

        # Stage runtime fields are assigned directly on self by the bootstrap thread.
        self.num_stages = len(self.stage_metadata)

        logger.info(f"[AsyncOmniEngine] Orchestrator ready with {self.num_stages} stages")

    # ---- helpers for cross-thread queue access ----

    def _put_to_request_queue(self, msg: dict[str, Any]) -> None:
        """Thread-safe put onto the orchestrator's request_queue."""
        self._orch_loop.call_soon_threadsafe(self.request_queue.put_nowait, msg)

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
        """Process stage 0 input locally, then send to the Orchestrator.

        For stage 0 requests (the common path), input processing and output
        processor registration happen here in the caller's thread, avoiding
        a queue + coroutine-switch round-trip.  The Orchestrator receives a
        ready-to-submit OmniEngineCoreRequest.
        """
        # Keep the original prompt for downstream stages (they need the raw
        # dict, e.g. for multi_modal_data).
        original_prompt = prompt

        stage_type = self.stage_metadata[stage_id].get("stage_type")
        if stage_id == 0 and stage_type != "diffusion" and not isinstance(prompt, EngineCoreRequest):
            # Inject global_request_id into the raw prompt
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
                arrival_time=arrival_time,
            )
            # Restore external_req_id to the original user-facing request_id.
            # InputProcessor.process_inputs() renames request_id to an internal
            # UUID (saving the original in external_req_id), but then overwrites
            # external_req_id with the new internal ID. We need external_req_id
            # to match the key used in Orchestrator.request_states so that
            # output routing (output.request_id lookup) can find the req_state.
            request.external_req_id = request_id

            # Register with stage 0's output processor
            self.output_processors[0].add_request(
                request=request,
                prompt=prompt,
                parent_req=None,
                request_index=0,
                queue=None,
            )

            prompt = request

        self._put_to_request_queue(
            {
                "type": "add_request",
                "stage_id": stage_id,
                "request_id": request_id,
                "prompt": prompt,
                "original_prompt": original_prompt,
                "params": params,
                "sampling_params_list": sampling_params_list or [],
                "final_stage_id": final_stage_id,
            }
        )

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
        self._put_to_request_queue(
            {
                "type": "abort",
                "request_ids": request_ids,
            }
        )

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
                logger.warning("[AsyncOmniEngine] Orchestrator thread did not exit in time")

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
