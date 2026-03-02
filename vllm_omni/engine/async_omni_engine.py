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

from vllm_omni.engine.orchestrator import run_orchestrator_in_thread
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
        base_engine_args = {"tokenizer": tokenizer} if tokenizer is not None else None

        if stage_configs_path is None:
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

        logger.info(f"[AsyncOmniEngine] Launching Orchestrator thread with {self.num_stages} stages")

        # Create asyncio queues (will be used from the orchestrator's loop)
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()

        # Launch orchestrator background thread
        loop_ready = threading.Event()
        self._orch_loop_holder: list[asyncio.AbstractEventLoop] = []

        self.orchestrator_thread = threading.Thread(
            target=run_orchestrator_in_thread,
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
            raise TimeoutError(f"Orchestrator did not become ready within {stage_init_timeout}s")

        if ready_msg.get("type") == "error":
            raise RuntimeError(f"Orchestrator failed to start: {ready_msg.get('error')}")

        assert ready_msg["type"] == "ready", f"Expected ready message, got {ready_msg['type']}"

        self.num_stages = ready_msg["num_stages"]
        self.default_sampling_params_list = ready_msg["default_sampling_params_list"]
        self.stage_metadata = ready_msg["stage_metadata"]
        self.input_processor: InputProcessor = ready_msg["input_processor"]
        self.output_processors: list[MultimodalOutputProcessor] = ready_msg["output_processors"]

        logger.info(f"[AsyncOmniEngine] Orchestrator ready with {self.num_stages} stages")

    # ---- helpers for cross-thread queue access ----

    def _put_to_request_queue(self, msg: dict[str, Any]) -> None:
        """Thread-safe put onto the orchestrator's request_queue."""
        self._orch_loop.call_soon_threadsafe(self.request_queue.put_nowait, msg)

    def _blocking_get_output(self, timeout: float) -> dict[str, Any] | None:
        """Blocking get from the output_queue (used during init)."""
        fut = asyncio.run_coroutine_threadsafe(self.output_queue.get(), self._orch_loop)
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
            # TODO: Here we directly use the req id to assign.
            request.external_req_id = request.request_id

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
