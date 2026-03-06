"""
AsyncOmni V1 - Refactored async orchestrator using AsyncOmniEngine.

This is the new implementation that uses AsyncOmniEngine (which manages
StageEngineCoreClient instances) instead of OmniStage with worker processes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput
from vllm.pooling_params import PoolingParams

from vllm_omni.entrypoints.omni_v1_base import (
    OmniV1Base,
    omni_snapshot_download as _omni_snapshot_download,
)
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.inputs.preprocess import InputPreprocessor
    from vllm.tokenizers import TokenizerLike

    from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams

logger = init_logger(__name__)


def omni_snapshot_download(model_id: str) -> str:
    """Backward-compatible import location for snapshot helper."""
    return _omni_snapshot_download(model_id)


class AsyncOmniV1(EngineClient, OmniV1Base):
    """Asynchronous unified entry point for multi-stage pipelines using AsyncOmniEngine.

    This is the V1 refactored version that uses AsyncOmniEngine instead of
    OmniStage workers. It provides the same interface as AsyncOmni but with
    a cleaner architecture.

    Args:
        model: Model name or path to load.
        stage_configs: Optional list of stage configurations. If None, loads from model.
        stage_configs_path: Optional path to YAML file containing stage configurations.
        stage_init_timeout: Timeout for stage initialization (seconds).
        log_stats: Whether to enable statistics logging.
        async_chunk: Whether to use async chunk mode (parallel stage execution).
        output_modalities: List of output modalities.
        **kwargs: Additional keyword arguments.

    Example:
        >>> async_omni = AsyncOmniV1(model="Qwen/Qwen2.5-Omni-7B")
        >>> async for output in async_omni.generate(
        ...     prompt="Hello",
        ...     request_id="req-1",
        ...     sampling_params_list=[SamplingParams(), SamplingParams()]
        ... ):
        ...     print(output)
    """

    def __init__(
        self,
        model: str,
        stage_configs: list[Any] | None = None,
        stage_configs_path: str | None = None,
        stage_init_timeout: int = 300,
        log_stats: bool = False,
        async_chunk: bool = False,
        output_modalities: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        OmniV1Base.__init__(
            self,
            model=model,
            stage_configs=stage_configs,
            stage_configs_path=stage_configs_path,
            stage_init_timeout=stage_init_timeout,
            log_stats=log_stats,
            async_chunk=async_chunk,
            output_modalities=output_modalities,
            **kwargs,
        )
        self._pause_cond: asyncio.Condition = asyncio.Condition()
        self._paused: bool = False
        self.final_output_task: asyncio.Task | None = None

    @property
    def renderer(self):
        """Renderer is required by EngineClient protocol.

        AsyncOmniV1 is primarily an orchestrator over multiple EngineCore stages,
        and currently does not expose a unified renderer.
        """
        raise NotImplementedError("AsyncOmniV1.renderer is not implemented.")

    # ==================== Generate Method ====================

    async def generate(
        self,
        prompt: OmniPromptType,
        request_id: str,
        sampling_params_list: Sequence[OmniSamplingParams] | None = None,
        *,
        output_modalities: list[str] | None = None,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs for the given prompt asynchronously.

        Coordinates multi-stage pipeline execution. Processes the prompt through
        all stages in the pipeline and yields outputs as they become available.

        Args:
            prompt: Prompt to process. Can be a text string, token IDs,
                or multimodal prompt.
            request_id: Unique identifier for this request
            sampling_params_list: List of SamplingParams, one for each stage.
                Must have the same length as the number of stages.
                If None, uses default sampling params for each stage.
            output_modalities: Optional list of output modalities.

        Yields:
            OmniRequestOutput objects as they are produced by each stage.

        Raises:
            ValueError: If sampling_params_list has incorrect length.
        """
        # Wait until generation is resumed if the engine is paused
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)

        logger.debug(f"[AsyncOmniV1] generate() called for request {request_id}")

        try:
            # Start final output dispatcher on the first call to generate()
            self._final_output_handler()

            sampling_params_list = self.resolve_sampling_params_list(sampling_params_list)

            # Track per-request metrics
            wall_start_ts = time.time()
            req_start_ts: dict[str, float] = {}

            # Determine the final stage for E2E stats
            final_stage_id_for_e2e = self._compute_final_stage_id(output_modalities)

            # Add request to stage 0 (Orchestrator handles all stage transitions)
            logger.info("[AsyncOmniV1] submit request %s to stage-0", request_id)
            req_state, submit_ts = self.submit_request(
                request_id=request_id,
                prompt=prompt,
                sampling_params_list=sampling_params_list,
                final_stage_id_for_e2e=final_stage_id_for_e2e,
                wall_start_ts=wall_start_ts,
            )
            if req_state.metrics is None:
                raise RuntimeError(f"Missing metrics for request {request_id}")
            metrics = req_state.metrics
            req_start_ts[request_id] = submit_ts

            # Process results based on mode
            # Both sequential and async_chunk modes read the same message stream
            # from Orchestrator; stage-transfer behavior differs inside
            # Orchestrator._route_output().
            async for output in self._process_orchestrator_results(
                request_id,
                metrics,
                final_stage_id_for_e2e,
                req_start_ts,
                wall_start_ts,
            ):
                yield output

            logger.debug(f"[AsyncOmniV1] Request {request_id} completed")

            self._log_summary_and_cleanup(request_id)

        except (asyncio.CancelledError, GeneratorExit):
            await self.abort(request_id)
            logger.info(f"[AsyncOmniV1] Request {request_id} aborted.")
            raise

    async def encode(
        self,
        prompt: Any,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: dict[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """EngineClient.encode() stub.

        Omni pipeline currently exposes only generate() API at orchestrator level.
        """
        raise NotImplementedError("AsyncOmniV1.encode is not implemented.")

    # ==================== Processing Methods ====================

    async def _process_orchestrator_results(
        self,
        request_id: str,
        metrics: OrchestratorMetrics,
        final_stage_id_for_e2e: int,
        req_start_ts: dict[str, float],
        wall_start_ts: float,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Read results from the Orchestrator (via the request's asyncio.Queue)
        and yield OmniRequestOutput objects.

        The Orchestrator handles all stage-to-stage transfers. This method
        only processes final outputs that arrive on the per-request queue.
        """
        req_state = self.request_states.get(request_id)
        if req_state is None:
            return

        while True:
            result = await req_state.queue.get()

            stage_id = result.get("stage_id", 0)

            # Check for errors
            if "error" in result:
                logger.error(
                    "[AsyncOmniV1] Orchestrator error for req=%s stage-%s: %s",
                    request_id,
                    stage_id,
                    result["error"],
                )
                raise RuntimeError(result)

            # Process the result (constructs OmniRequestOutput)
            output_to_yield = self._process_single_result(
                result,
                stage_id,
                metrics,
                req_start_ts,
                wall_start_ts,
                final_stage_id_for_e2e,
            )

            if output_to_yield:
                logger.debug(
                    "[AsyncOmniV1] req=%s stage-%s yielding final_output_type=%s",
                    request_id,
                    stage_id,
                    getattr(output_to_yield, "final_output_type", None),
                )
                yield output_to_yield

            # The Orchestrator sets "finished" when the final stage is done
            if result.get("finished"):
                break

    # ==================== Output Handler ====================

    def _final_output_handler(self) -> None:
        """Start the final output handler if not already running.

        This handler reads messages from the Orchestrator output queue and
        routes them to per-request asyncio.Queues.
        """
        if self.final_output_task is not None:
            return

        engine = self.engine

        async def _final_output_loop():
            """Background coroutine that dispatches final outputs to request queues."""
            loop = asyncio.get_event_loop()
            try:
                while True:
                    # Blocking read with timeout (runs in executor to avoid
                    # blocking the event loop)
                    msg = await loop.run_in_executor(None, engine.try_get_output_blocking)

                    should_continue, _, stage_id, req_state = self._handle_output_message(msg)
                    if should_continue:
                        continue

                    req_state.stage_id = stage_id

                    # Route to the per-request queue
                    await req_state.queue.put(msg)

            except Exception as e:
                logger.exception("[AsyncOmniV1] final_output_loop failed.")
                for req_state in list(self.request_states.values()):
                    error_msg = {"request_id": req_state.request_id, "error": str(e)}
                    await req_state.queue.put(error_msg)
                self.final_output_task = None

        self.final_output_task = asyncio.create_task(_final_output_loop())
        logger.debug("[AsyncOmniV1] Final output handler started")

    # ==================== Control Methods ====================

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort request(s) via the Orchestrator."""
        self._abort_requests(request_id)

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause generation."""
        async with self._pause_cond:
            if self._paused:
                return
            self._paused = True

        # TODO: Implement request draining if wait_for_inflight_requests

        if clear_cache:
            # Clear caches for all stages.
            await self.reset_prefix_cache(
                reset_running_requests=not wait_for_inflight_requests,
                reset_connector=True,
            )
            await self.reset_mm_cache()
            await self.reset_encoder_cache()

    async def resume_generation(self) -> None:
        """Resume generation."""
        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()

    async def is_paused(self) -> bool:
        """Check if paused."""
        async with self._pause_cond:
            return self._paused

    async def start_profile(self) -> None:
        """Start profiling all stages.

        TODO: Forward to Orchestrator process via message.
        """
        logger.warning("[AsyncOmniV1] start_profile not yet supported with Orchestrator process")

    async def stop_profile(self) -> None:
        """Stop profiling all stages.

        TODO: Forward to Orchestrator process via message.
        """
        logger.warning("[AsyncOmniV1] stop_profile not yet supported with Orchestrator process")

    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache for all stages.

        TODO: Forward to Orchestrator process via message.
        """
        logger.warning("[AsyncOmniV1] reset_mm_cache not yet supported with Orchestrator process")

    async def reset_encoder_cache(self) -> None:
        """Reset the encoder cache for all stages.

        TODO: Forward to Orchestrator process via message.
        """
        logger.warning("[AsyncOmniV1] reset_encoder_cache not yet supported with Orchestrator process")

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        """Reset the prefix cache for all stages.

        TODO: Forward to Orchestrator process via message.
        """
        logger.warning("[AsyncOmniV1] reset_prefix_cache not yet supported with Orchestrator process")
        return True

    async def sleep(self, level: int = 1) -> None:
        """Sleep all stages.

        TODO: Forward to Orchestrator process via message.
        """
        logger.warning("[AsyncOmniV1] sleep not yet supported with Orchestrator process")

    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up all stages.

        TODO: Forward to Orchestrator process via message.
        """
        logger.warning("[AsyncOmniV1] wake_up not yet supported with Orchestrator process")

    async def is_sleeping(self) -> bool:
        """Return whether all stages are sleeping.

        TODO: Forward to Orchestrator process via message.
        """
        return False

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into all stages.

        TODO: Forward to Orchestrator process via message.
        """
        logger.warning("[AsyncOmniV1] add_lora not yet supported with Orchestrator process")
        return False

    # ==================== Properties ====================

    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return self.final_output_task is not None and not self.final_output_task.done()

    # ==================== EngineClient Interface ====================

    async def get_vllm_config(self) -> VllmConfig:
        """Get vllm config.

        TODO: Forward to Orchestrator process via message.
        """
        return None  # type: ignore[return-value]

    async def get_model_config(self) -> Any:
        """Get model config.

        TODO: Forward to Orchestrator process via message.
        """
        return None

    async def get_input_preprocessor(self) -> InputPreprocessor:
        """Get input preprocessor."""
        return None  # TODO: Implement if needed

    async def get_tokenizer(self) -> TokenizerLike:
        """Get tokenizer.

        TODO: Forward to Orchestrator process via message.
        """
        return None  # type: ignore[return-value]

    async def is_tracing_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return False

    async def do_log_stats(self) -> None:
        """Log statistics.

        TODO: Forward to Orchestrator process via message.
        """
        pass

    async def check_health(self) -> None:
        """Check engine health by verifying the Orchestrator process is alive."""
        super().check_health()

    # ==================== Shutdown ====================

    def shutdown(self) -> None:
        """Shutdown the engine."""
        if self.final_output_task is not None:
            self.final_output_task.cancel()
            self.final_output_task = None
        OmniV1Base.shutdown(self)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass
