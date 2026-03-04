"""
AsyncOmni V1 - Refactored async orchestrator using AsyncOmniEngine.

This is the new implementation that uses AsyncOmniEngine (which manages
StageEngineCoreClient instances) instead of OmniStage with worker processes.
"""

from __future__ import annotations

import asyncio
import os
import time
import types
from collections.abc import AsyncGenerator, Iterable, Sequence
from pprint import pformat
from typing import TYPE_CHECKING, Any

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput
from vllm.pooling_params import PoolingParams
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.utils import get_final_stage_id_for_e2e
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.inputs.preprocess import InputPreprocessor
    from vllm.tokenizers import TokenizerLike

    from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams

logger = init_logger(__name__)


def _dummy_snapshot_download(model_id):
    return model_id


def omni_snapshot_download(model_id) -> str:
    # TODO: this is just a workaround for quickly use modelscope, we should support
    # modelscope in weight loading feature instead of using `snapshot_download`
    if os.environ.get("VLLM_USE_MODELSCOPE", False):
        from modelscope.hub.snapshot_download import snapshot_download

        return snapshot_download(model_id)
    else:
        return _dummy_snapshot_download(model_id)


class AsyncOmniV1(EngineClient):
    """Asynchronous unified entry point for multi-stage pipelines using AsyncOmniEngine.

    This is the V1 refactored version that uses AsyncOmniEngine instead of
    OmniStage workers. It provides the same interface as AsyncOmni but with
    a cleaner architecture.

    Args:
        model: Model name or path to load.
        stage_configs: Optional list of stage configurations. If None, loads from model.
        stage_configs_path: Optional path to YAML file containing stage configurations.
        stage_init_timeout: Timeout for stage initialization (seconds).
        log_requests: Whether to log requests.
        enable_stats: Whether to enable statistics logging.
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
        log_requests: bool = True,
        enable_stats: bool = False,
        async_chunk: bool = False,
        output_modalities: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        model = omni_snapshot_download(model)
        self.model = model
        self.log_requests = log_requests
        self.enable_stats = enable_stats
        self.async_chunk = async_chunk
        self.output_modalities = output_modalities or []

        logger.info(f"[AsyncOmniV1] Initializing with model {model}")
        logger.info(f"[AsyncOmniV1] stage_configs: {stage_configs}")
        logger.info(f"[AsyncOmniV1] stage_configs_path: {stage_configs_path}")
        logger.info(f"[AsyncOmniV1] stage_init_timeout: {stage_init_timeout}")
        logger.info(f"[AsyncOmniV1] log_requests: {log_requests}")
        logger.info(f"[AsyncOmniV1] enable_stats: {enable_stats}")
        logger.info(f"[AsyncOmniV1] async_chunk: {async_chunk}")
        logger.info(f"[AsyncOmniV1] output_modalities: {output_modalities}")
        logger.info(f"[AsyncOmniV1] kwargs: {kwargs}")
        # Initialize AsyncOmniEngine (launches Orchestrator child process)
        import time

        st = time.time()
        self.engine = AsyncOmniEngine(
            model=model,
            stage_configs=stage_configs,
            stage_configs_path=stage_configs_path,
            stage_init_timeout=stage_init_timeout,
            **kwargs,
        )
        et = time.time()
        logger.info(f"[AsyncOmniV1] AsyncOmniEngine initialized in {et - st:.2f} seconds")
        # Keep compatibility with V0 behavior: async_chunk can be enabled
        # from YAML stage config without passing the constructor flag.
        self.async_chunk = bool(self.async_chunk or getattr(self.engine, "async_chunk", False))
        logger.info(f"[AsyncOmniV1] effective async_chunk: {self.async_chunk}")

        # Pause/resume control
        self._pause_cond: asyncio.Condition = asyncio.Condition()
        self._paused: bool = False

        # Request state tracking
        self.request_states: dict[str, ClientRequestState] = {}
        self.output_handler: asyncio.Task | None = None

        # Get default sampling params from the Orchestrator ready message
        self.default_sampling_params_list = self.engine.default_sampling_params_list

        # Build output_modalities from stage metadata (mirrors V0's omni.py:
        # self.output_modalities = [st.final_output_type for st in self.stage_list])
        # This is used as default_modalities in get_final_stage_id_for_e2e.
        if not self.output_modalities:
            self.output_modalities = [
                self.engine.get_stage_metadata(i).get("final_output_type") for i in range(self.engine.num_stages)
            ]

        logger.info(f"[AsyncOmniV1] Initialized with {self.engine.num_stages} stages for model {model}")

    @property
    def num_stages(self) -> int:
        """Get the number of stages."""
        return self.engine.num_stages

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
            # Start output handler on the first call to generate()
            self._run_output_handler()

            # Use default sampling params if not provided
            if sampling_params_list is None:
                sampling_params_list = self.default_sampling_params_list

            if len(sampling_params_list) != self.num_stages:
                raise ValueError(f"Expected {self.num_stages} sampling params, got {len(sampling_params_list)}")

            # Track per-request metrics
            wall_start_ts = time.time()
            req_start_ts: dict[int, float] = {}

            # Determine the final stage for E2E stats
            # Use stage_metadata from the Orchestrator instead of direct stage_client access
            # Wrap dicts in SimpleNamespace so getattr() works in get_final_stage_id_for_e2e
            stage_meta_list = [
                types.SimpleNamespace(**self.engine.get_stage_metadata(i)) for i in range(self.num_stages)
            ]
            final_stage_id_for_e2e = get_final_stage_id_for_e2e(
                output_modalities,
                self.output_modalities,
                stage_meta_list,
            )

            # Create metrics tracker
            metrics = OrchestratorMetrics(
                self.num_stages,
                self.enable_stats,
                wall_start_ts,
                final_stage_id_for_e2e,
            )

            # Create request state
            req_state = ClientRequestState(request_id)
            req_state.metrics = metrics
            self.request_states[request_id] = req_state

            # Add request to stage 0 (Orchestrator handles all stage transitions)
            logger.info("[AsyncOmniV1] submit request %s to stage-0", request_id)
            await self.engine.add_request(
                stage_id=0,
                request_id=request_id,
                prompt=prompt,
                params=sampling_params_list[0],
                sampling_params_list=list(sampling_params_list),
                final_stage_id=final_stage_id_for_e2e,
            )
            metrics.stage_first_ts[0] = time.time()
            req_start_ts[request_id] = time.time()

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

            # Log summary
            try:
                summary = metrics.build_and_log_summary()
                logger.info("[Summary] %s", pformat(summary, sort_dicts=False))
            except Exception as e:
                logger.exception(f"[AsyncOmniV1] Failed to build/log summary: {e}")
            finally:
                self.request_states.pop(request_id, None)

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
        if False:  # pragma: no cover - keep as AsyncGenerator type
            yield None  # type: ignore[misc]

    # ==================== Processing Methods ====================

    async def _process_orchestrator_results(
        self,
        request_id: str,
        metrics: OrchestratorMetrics,
        final_stage_id_for_e2e: int,
        req_start_ts: dict[int, float],
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
            engine_outputs, finished, output_to_yield = self._process_single_result(
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

    def _process_single_result(
        self,
        result: dict[str, Any],
        stage_id: int,
        metrics: OrchestratorMetrics,
        req_start_ts: dict[int, float],
        wall_start_ts: float,
        final_stage_id_for_e2e: int,
    ) -> tuple[Any, bool, OmniRequestOutput | None]:
        """Process a single result from a stage.

        Returns:
            engine_outputs: The decoded outputs.
            finished: Whether the stage processing is finished.
            output_to_yield: An OmniRequestOutput to yield, or None.
        """
        req_id = result.get("request_id")

        # Check for errors
        if "error" in result:
            logger.error(f"[AsyncOmniV1] Stage {stage_id} error on request {req_id}: {result['error']}")
            raise RuntimeError(result)

        # Get engine outputs
        engine_outputs = result.get("engine_outputs")
        if isinstance(engine_outputs, list) and len(engine_outputs) > 0:
            engine_outputs = engine_outputs[0]

        finished = engine_outputs.finished

        # Mark stage timestamps: use the Orchestrator's submit timestamp
        # for stage_first_ts so it reflects when the request was actually
        # submitted, not when the result arrived in the main thread.
        submit_ts = result.get("stage_submit_ts")
        now = time.time()
        if metrics.stage_first_ts[stage_id] is None:
            metrics.stage_first_ts[stage_id] = submit_ts if submit_ts is not None else now
        metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, now)

        # Process metrics
        _m = result.get("metrics")
        if finished and _m is not None:
            metrics.on_stage_metrics(stage_id, req_id, _m)

        # Determine if this is a final output stage (use cached metadata)
        stage_meta = self.engine.get_stage_metadata(stage_id)
        output_to_yield = None

        if stage_meta["final_output"]:
            # Finalize request metrics
            try:
                rid_key = str(req_id)
                if stage_id == final_stage_id_for_e2e and rid_key not in metrics.e2e_done and finished:
                    metrics.on_finalize_request(
                        stage_id,
                        req_id,
                        req_start_ts.get(req_id, wall_start_ts),
                    )
            except Exception as e:
                logger.exception(f"[AsyncOmniV1] Finalize request handling error: {e}")

            # Construct output to yield
            images = []
            if stage_meta["final_output_type"] == "image":
                if isinstance(engine_outputs, OmniRequestOutput) and engine_outputs.images:
                    images = engine_outputs.images
                elif hasattr(engine_outputs, "images") and engine_outputs.images:
                    images = engine_outputs.images

            if stage_meta["final_output_type"] == "image":
                output_to_yield = OmniRequestOutput(
                    stage_id=stage_id,
                    final_output_type=stage_meta["final_output_type"],
                    request_output=engine_outputs,
                    images=images,
                )
            else:
                output_to_yield = OmniRequestOutput(
                    stage_id=stage_id,
                    final_output_type=stage_meta["final_output_type"],
                    request_output=engine_outputs,
                )

        return engine_outputs, finished, output_to_yield

    # ==================== Output Handler ====================

    def _run_output_handler(self) -> None:
        """Start the output handler if not already running.

        The output handler reads results from the Orchestrator's output_queue
        (via asyncio.Queue) and routes them to per-request asyncio.Queues.
        """
        if self.output_handler is not None:
            return

        request_states = self.request_states
        engine = self.engine

        async def output_handler():
            """Background coroutine that reads from the Orchestrator output queue."""
            loop = asyncio.get_event_loop()
            try:
                while True:
                    # Blocking read with timeout (runs in executor to avoid
                    # blocking the event loop)
                    msg = await loop.run_in_executor(None, engine.try_get_output_blocking)
                    if msg is None:
                        continue

                    msg_type = msg.get("type")
                    if msg_type == "stage_metrics":
                        # Non-final stage metrics: aggregate directly
                        # without routing through the per-request queue.
                        req_id = msg.get("request_id")
                        req_state = request_states.get(req_id)
                        if req_state is not None and req_state.metrics is not None:
                            _m = msg.get("metrics")
                            if _m is not None:
                                stage_id = msg.get("stage_id", 0)
                                req_state.metrics.on_stage_metrics(
                                    stage_id,
                                    req_id,
                                    _m,
                                )
                                # Use the Orchestrator's submit timestamp
                                # for stage_first_ts (when the request was
                                # actually submitted to this stage).
                                submit_ts = msg.get("stage_submit_ts")
                                now = time.time()
                                if req_state.metrics.stage_first_ts[stage_id] is None:
                                    req_state.metrics.stage_first_ts[stage_id] = (
                                        submit_ts if submit_ts is not None else now
                                    )
                                req_state.metrics.stage_last_ts[stage_id] = max(
                                    req_state.metrics.stage_last_ts[stage_id] or 0.0,
                                    now,
                                )
                        continue
                    if msg_type != "output":
                        logger.warning(
                            "[AsyncOmniV1] output_handler got unexpected msg type: %s",
                            msg_type,
                        )
                        continue

                    req_id = msg.get("request_id")
                    req_state = request_states.get(req_id)

                    if req_state is None:
                        logger.debug(
                            "[AsyncOmniV1] Dropping output for unknown req %s",
                            req_id,
                        )
                        continue

                    # Update stage_id on the request state
                    req_state.stage_id = msg.get("stage_id", 0)

                    # Route to the per-request queue
                    await req_state.queue.put(msg)

            except Exception as e:
                logger.exception("[AsyncOmniV1] output_handler failed.")
                for req_state in request_states.values():
                    error_msg = {"request_id": req_state.request_id, "error": str(e)}
                    await req_state.queue.put(error_msg)
                self.output_handler = None

        self.output_handler = asyncio.create_task(output_handler())
        logger.debug("[AsyncOmniV1] Output handler started")

    # ==================== Control Methods ====================

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort request(s) via the Orchestrator."""
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)

        # Send abort to Orchestrator (it will abort in all stages)
        await self.engine.abort(request_ids)

        # Remove from request states
        for req_id in request_ids:
            self.request_states.pop(req_id, None)

        if self.log_requests:
            logger.info(f"[AsyncOmniV1] Aborted request(s) {','.join(request_ids)}")

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
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        """Check if the engine is stopped."""
        return self.errored

    @property
    def errored(self) -> bool:
        """Check if the Orchestrator thread has died."""
        return hasattr(self.engine, "orchestrator_thread") and not self.engine.orchestrator_thread.is_alive()

    @property
    def dead_error(self) -> BaseException:
        """Get the dead error."""
        return EngineDeadError()

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
        if self.errored:
            raise EngineDeadError("Orchestrator process is not alive")

    # ==================== Shutdown ====================

    def shutdown(self) -> None:
        """Shutdown the engine."""
        logger.info("[AsyncOmniV1] Shutting down")
        self.engine.shutdown()
        if self.output_handler is not None:
            self.output_handler.cancel()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass
