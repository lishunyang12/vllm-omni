"""
Orchestrator for vLLM-Omni V1 architecture.

Runs inside a background thread with its own asyncio event loop.
Owns all StageAsyncCoreClient instances, input/output processors,
and handles stage-to-stage transfer logic.
"""

from __future__ import annotations

import asyncio
import copy
import time as _time
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutputs

from vllm_omni.distributed.omni_connectors.adapter import compute_talker_prompt_ids_length
from vllm_omni.engine import (
    AdditionalInformationEntry,
    AdditionalInformationPayload,
    OmniEngineCoreRequest,
    PromptEmbedsPayload,
)
from vllm_omni.metrics.stats import StageRequestStats as StageRequestMetrics
from vllm_omni.metrics.stats import StageStats
from vllm_omni.metrics.utils import count_tokens_from_outputs

logger = init_logger(__name__)


def _dtype_to_name(dtype: torch.dtype) -> str:
    """Convert torch dtype to string representation.

    Args:
        dtype: PyTorch dtype to convert

    Returns:
        String representation of the dtype (e.g., "float32", "int64")
    """
    mapping = {
        torch.float32: "float32",
        torch.float: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.bfloat16: "bfloat16",
        torch.float64: "float64",
        torch.double: "float64",
        torch.int64: "int64",
        torch.long: "int64",
        torch.int32: "int32",
        torch.int: "int32",
        torch.int16: "int16",
        torch.short: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    return mapping.get(dtype, str(dtype).replace("torch.", ""))


def build_engine_core_request_from_tokens(
    request_id: str,
    prompt: dict[str, Any],
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    model_config: ModelConfig | None = None,
) -> OmniEngineCoreRequest:
    """Build an OmniEngineCoreRequest directly from an OmniTokensPrompt.

    Lightweight alternative to the full InputProcessor pipeline - skips
    tokenization, multimodal preprocessing, LoRA validation, and platform
    validation.  Intended for stage 1+ where the upstream stage has already
    produced token IDs and optional embeddings.
    """
    if arrival_time is None:
        arrival_time = _time.time()

    prompt_token_ids = prompt["prompt_token_ids"]

    # Clone params and set max_tokens if needed
    sampling_params = None
    pooling_params = None
    if isinstance(params, SamplingParams):
        sampling_params = params.clone()
        if sampling_params.max_tokens is None and model_config is not None:
            sampling_params.max_tokens = model_config.max_model_len - len(prompt_token_ids)
    else:
        pooling_params = params.clone()

    # TODO: payload are parsed with the request by now. But will move to connector later.
    # Serialize prompt_embeds if present
    prompt_embeds_payload: PromptEmbedsPayload | None = None
    pe: torch.Tensor | None = prompt.get("prompt_embeds")
    if pe is not None:
        pe_cpu = pe.detach().to("cpu").contiguous()
        prompt_embeds_payload = PromptEmbedsPayload(
            data=pe_cpu.numpy().tobytes(),
            shape=list(pe_cpu.shape),
            dtype=_dtype_to_name(pe_cpu.dtype),
        )

    # Serialize additional_information if present
    additional_info_payload: AdditionalInformationPayload | None = None
    raw_info: dict[str, Any] | None = prompt.get("additional_information")
    if raw_info is not None:
        entries: dict[str, AdditionalInformationEntry] = {}
        for key, value in raw_info.items():
            if isinstance(value, torch.Tensor):
                v_cpu = value.detach().to("cpu").contiguous()
                entries[key] = AdditionalInformationEntry(
                    tensor_data=v_cpu.numpy().tobytes(),
                    tensor_shape=list(v_cpu.shape),
                    tensor_dtype=_dtype_to_name(v_cpu.dtype),
                )
            elif isinstance(value, list):
                entries[key] = AdditionalInformationEntry(list_data=value)
            else:
                logger.warning(
                    "[build_engine_core_request_from_tokens] req=%s skipping unsupported type key=%s type=%s",
                    request_id,
                    key,
                    type(value).__name__,
                )
        additional_info_payload = AdditionalInformationPayload(entries=entries)

    eos_token_id = None
    if model_config is not None and hasattr(model_config, "hf_config"):
        eos_token_id = getattr(model_config.hf_config, "eos_token_id", None)

    return OmniEngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        eos_token_id=eos_token_id,
        arrival_time=arrival_time,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        prompt_embeds=prompt_embeds_payload,
        additional_information=additional_info_payload,
    )


# ============================================================
# Orchestrator internals (run inside the background thread)
# ============================================================


@dataclass
class OrchestratorRequestState:
    """Per-request bookkeeping inside the Orchestrator."""

    request_id: str
    prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = -1

    # Metrics: timestamp when request was submitted to each stage
    stage_submit_ts: dict[int, float] = field(default_factory=dict)


class Orchestrator:
    """Runs inside a background thread's asyncio event loop.

    Owns all StageAsyncCoreClient instances, input/output processors,
    and handles stage-to-stage transfer logic.
    """

    def __init__(
        self,
        request_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stage_clients: list[Any],
        output_processors: list[Any],
        stage_vllm_configs: list[Any],
        *,
        async_chunk: bool = False,
    ) -> None:
        self.request_queue = request_queue
        self.output_queue = output_queue

        self.num_stages = len(stage_clients)
        self.async_chunk = bool(async_chunk)

        self.stage_clients: list[Any] = stage_clients
        self.output_processors: list[Any] = output_processors
        self.stage_vllm_configs: list[Any] = stage_vllm_configs

        # Per-request state
        self.request_states: dict[str, OrchestratorRequestState] = {}

        # Per-stage metrics accumulators (mirrors V0's _batch_seq / _agg_*)
        self._batch_seq: list[int] = [0] * self.num_stages
        self._agg_total_tokens: list[int] = [0] * self.num_stages
        self._agg_total_gen_time_ms: list[float] = [0.0] * self.num_stages

        # Shutdown coordination
        self._shutdown_event = asyncio.Event()

    async def run(self) -> None:
        """Main entry point for the Orchestrator event loop."""
        logger.info("[Orchestrator] Starting event loop")

        request_task = asyncio.create_task(self._request_handler(), name="orchestrator-request-handler")
        output_task = asyncio.create_task(self._output_handler(), name="orchestrator-output-handler")

        try:
            # Run both tasks concurrently; if either fails the other is cancelled.
            await asyncio.gather(request_task, output_task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[Orchestrator] Fatal error in orchestrator tasks")
            raise
        finally:
            for t in (request_task, output_task):
                if not t.done():
                    t.cancel()
            try:
                await asyncio.gather(request_task, output_task, return_exceptions=True)
            except Exception:
                pass

            # Cancel any remaining tasks spawned by wait_for / gather so
            # the event loop can close cleanly without "pending task" errors.
            loop = asyncio.get_running_loop()
            pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task() and not t.done()]
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
        """Inner loop for _output_handler (separated for clean cancellation).

        Control flow: poll raw → process through output processor → route.
        """
        while not self._shutdown_event.is_set():
            idle = True
            for stage_id in range(self.num_stages):
                if self._shutdown_event.is_set():
                    return

                # 1) Diffusion stage: poll non-blocking queue
                # TODO (Peiqi): the output of diffusion stage is OmniRequestOutput,
                # which is different from EngineCoreOutputs (LLM stages). We may want to unify
                # the output format in the future to simplify the processing logic in Orchestrator.
                stage_client = self.stage_clients[stage_id]
                if stage_client.stage_type == "diffusion":
                    output = stage_client.get_diffusion_output_async()
                    if output is not None:
                        idle = False
                        req_state = self.request_states.get(output.request_id)
                        if req_state is not None:
                            stage_metrics = self._build_stage_metrics(stage_id, output.request_id, [output], req_state)
                            await self._route_output(stage_id, output, req_state, stage_metrics)
                    continue

                # 1) Poll raw outputs from the stage
                try:
                    raw_outputs = await asyncio.wait_for(self._poll_stage_raw(stage_id), timeout=0.001)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if self._shutdown_event.is_set():
                        return
                    logger.exception(
                        "[Orchestrator] _poll_stage_raw failed for stage-%s",
                        stage_id,
                    )
                    raise

                if raw_outputs is None:
                    continue
                idle = False

                # 2) Process raw outputs through the output processor
                request_outputs = await self._process_stage_outputs(stage_id, raw_outputs)

                # 3) Route each processed output
                for output in request_outputs:
                    req_state = self.request_states.get(output.request_id)
                    if req_state is None:
                        logger.warning(
                            "[Orchestrator] Dropping output for unknown req %s at stage-%s (known reqs: %s)",
                            output.request_id,
                            stage_id,
                            list(self.request_states.keys()),
                        )
                        continue
                    stage_metrics = None
                    if output.finished:
                        stage_metrics = self._build_stage_metrics(
                            stage_id,
                            output.request_id,
                            [output],
                            req_state,
                        )
                    await self._route_output(stage_id, output, req_state, stage_metrics)

            if idle:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)

    async def _route_output(
        self,
        stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        stage_metrics: Any,
    ) -> None:
        """Route a processed output: send to main thread and/or forward to next stage."""
        req_id = output.request_id
        finished = output.finished
        submit_ts = req_state.stage_submit_ts.get(stage_id)
        stage_client = self.stage_clients[stage_id]

        if stage_client.final_output:
            await self.output_queue.put(
                {
                    "type": "output",
                    "request_id": req_id,
                    "stage_id": stage_id,
                    "engine_outputs": [output],
                    "metrics": stage_metrics,
                    "finished": finished and stage_id == req_state.final_stage_id,
                    "stage_submit_ts": submit_ts,
                }
            )
        elif stage_metrics is not None:
            await self.output_queue.put(
                {
                    "type": "stage_metrics",
                    "request_id": req_id,
                    "stage_id": stage_id,
                    "metrics": stage_metrics,
                    "stage_submit_ts": submit_ts,
                }
            )

        if finished and stage_id < req_state.final_stage_id and not self.async_chunk:
            await self._forward_to_next_stage(req_id, stage_id, output, req_state)

        if finished and stage_id == req_state.final_stage_id:
            self.request_states.pop(req_id, None)

    def _build_stage_metrics(
        self,
        stage_id: int,
        req_id: str,
        request_outputs: list[RequestOutput],
        req_state: OrchestratorRequestState,
    ) -> StageRequestMetrics:
        """Build StageRequestMetrics for a finished request at a stage.

        Reuses the same StageRequestMetrics dataclass as V0 so that
        OrchestratorMetrics / on_stage_metrics / on_finalize_request
        work identically.
        """
        now = _time.time()
        submit_ts = req_state.stage_submit_ts.get(stage_id, now)
        stage_gen_time_ms = (now - submit_ts) * 1000.0

        num_tokens_out = count_tokens_from_outputs(request_outputs)
        num_tokens_in = 0
        if stage_id == 0:
            for ro in request_outputs:
                ptids = getattr(ro, "prompt_token_ids", None)
                if ptids is not None:
                    num_tokens_in += len(ptids)

        # Monotonic batch counter per stage (mirrors V0's _batch_seq)
        self._batch_seq[stage_id] += 1
        batch_id = self._batch_seq[stage_id]

        # Accumulate for running-average stage_stats
        self._agg_total_tokens[stage_id] += num_tokens_out
        self._agg_total_gen_time_ms[stage_id] += stage_gen_time_ms

        return StageRequestMetrics(
            num_tokens_in=num_tokens_in,
            num_tokens_out=num_tokens_out,
            stage_gen_time_ms=stage_gen_time_ms,
            batch_id=batch_id,
            batch_size=1,
            rx_decode_time_ms=0.0,
            rx_transfer_bytes=0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(
                total_token=self._agg_total_tokens[stage_id],
                total_gen_time_ms=self._agg_total_gen_time_ms[stage_id],
            ),
        )

    async def _forward_to_next_stage(
        self,
        req_id: str,
        stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Forward output from current stage to the next stage.

        Handles the full pipeline: set outputs on current stage, compute
        next-stage inputs, build lightweight requests, and submit them.
        """
        next_stage_id = stage_id + 1
        next_client = self.stage_clients[next_stage_id]
        params = req_state.sampling_params_list[next_stage_id]

        # TODO: current we don't have model in this situation, verify next.
        # Diffusion next stage: use custom_process_input_func or raw prompt
        if next_client.stage_type == "diffusion":
            self.stage_clients[stage_id].set_engine_outputs([output])
            if next_client.custom_process_input_func is not None:
                diffusion_prompt = next_client.custom_process_input_func(
                    self.stage_clients,
                    next_client.engine_input_source,
                    req_state.prompt,
                    False,
                )
                if isinstance(diffusion_prompt, list):
                    diffusion_prompt = diffusion_prompt[0]
            else:
                diffusion_prompt = req_state.prompt
            await next_client.add_request_async(req_id, diffusion_prompt, params)
            req_state.stage_submit_ts[next_stage_id] = _time.time()
            return

        if not isinstance(output, list):
            engine_outputs = [output]
        else:
            engine_outputs = output

        self.stage_clients[stage_id].set_engine_outputs(engine_outputs)

        # Process inputs for next stage
        try:
            next_inputs = next_client.process_engine_inputs(
                stage_list=self.stage_clients,
                prompt=req_state.prompt,
            )
        except Exception:
            logger.exception(
                "[Orchestrator] req=%s process_engine_inputs FAILED for stage-%s",
                req_id,
                next_stage_id,
            )
            raise

        # Build and submit requests for each input
        for next_input in next_inputs:
            request = build_engine_core_request_from_tokens(
                request_id=req_id,
                prompt=next_input,
                params=params,
                model_config=self.stage_vllm_configs[next_stage_id].model_config,
            )

            # TODO: Here we directly use the req id to assign.
            request.external_req_id = request.request_id

            self.output_processors[next_stage_id].add_request(
                request=request,
                prompt=None,
                parent_req=None,
                request_index=0,
                queue=None,
            )

            await next_client.add_request_async(request)

        # Record submit timestamp for the next stage
        req_state.stage_submit_ts[next_stage_id] = _time.time()

    async def _poll_stage_raw(self, stage_id: int) -> EngineCoreOutputs | None:
        """Pull raw EngineCoreOutputs from a stage client without processing.

        Returns the raw outputs object, or None when there is nothing
        to consume.
        """
        outputs = await self.stage_clients[stage_id].get_output_async()
        if not outputs.outputs:
            return None
        return outputs

    async def _process_stage_outputs(self, stage_id: int, raw_outputs: EngineCoreOutputs) -> list[RequestOutput]:
        """Run the output processor on raw outputs, returning RequestOutputs.

        Also handles abort forwarding and scheduler stats updates.
        """
        processor = self.output_processors[stage_id]

        processed = processor.process_outputs(
            raw_outputs.outputs,
            raw_outputs.timestamp,
            None,
        )

        if processed.reqs_to_abort:
            await self.stage_clients[stage_id].abort_requests_async(processed.reqs_to_abort)

        if raw_outputs.scheduler_stats is not None:
            processor.update_scheduler_stats(raw_outputs.scheduler_stats)

        return processed.request_outputs

    async def _handle_add_request(self, msg: dict[str, Any]) -> None:
        """Handle an add_request message from the main thread."""
        stage_id = msg["stage_id"]
        request_id = msg["request_id"]
        prompt = msg["prompt"]
        original_prompt = msg.get("original_prompt", prompt)
        params = msg["params"]
        sampling_params_list = msg["sampling_params_list"]
        final_stage_id = msg["final_stage_id"]

        logger.info(
            "[Orchestrator] _handle_add_request: stage=%s req=%s "
            "prompt_type=%s original_prompt_type=%s final_stage=%s "
            "num_sampling_params=%d",
            stage_id,
            request_id,
            type(prompt).__name__,
            type(original_prompt).__name__,
            final_stage_id,
            len(sampling_params_list),
        )

        # Track request state - use original_prompt so downstream stages
        # (e.g. thinker2talker) can access the raw dict with multi_modal_data.
        req_state = OrchestratorRequestState(
            request_id=request_id,
            prompt=original_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
        )
        req_state.stage_submit_ts[stage_id] = _time.time()
        self.request_states[request_id] = req_state

        # Stage-0 prompt is already a fully-formed OmniEngineCoreRequest
        # (pre-processed by AsyncOmniEngine.add_request, output processor
        # already registered there) - submit directly.
        request = prompt
        stage_client = self.stage_clients[stage_id]
        if stage_client.stage_type == "diffusion":
            await stage_client.add_request_async(request_id, prompt, params)
        else:
            await stage_client.add_request_async(request)

        if self.async_chunk and stage_id == 0 and final_stage_id > 0:
            await self._prewarm_async_chunk_stages(request_id, request, req_state)

    async def _prewarm_async_chunk_stages(
        self,
        request_id: str,
        stage0_request: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Pre-submit downstream stages for async-chunk mode.

        In async-chunk mode, stages exchange data through connectors/chunk adapters,
        so downstream stages should be armed once at request start instead of waiting
        for stage-finished forwarding.
        """
        if req_state.final_stage_id <= 0:
            return

        prompt_token_ids = getattr(stage0_request, "prompt_token_ids", None)
        if prompt_token_ids is None:
            logger.warning(
                "[Orchestrator] async_chunk prewarm skipped for req=%s: stage0 prompt_token_ids missing",
                request_id,
            )
            return

        # Match V0 behavior: pre-arm stage-1+ with placeholder prompt IDs.
        try:
            next_prompt_len = max(1, compute_talker_prompt_ids_length(prompt_token_ids))
        except Exception:
            next_prompt_len = max(1, len(prompt_token_ids))
        original_prompt = req_state.prompt
        if isinstance(original_prompt, dict):
            base_input = copy.deepcopy(original_prompt)
        else:
            base_input = {}
        base_input["prompt_token_ids"] = [0] * next_prompt_len
        base_input["multi_modal_data"] = None
        base_input["mm_processor_kwargs"] = None

        for next_stage_id in range(1, req_state.final_stage_id + 1):
            next_client = self.stage_clients[next_stage_id]
            params = req_state.sampling_params_list[next_stage_id]

            if next_client.stage_type == "diffusion":
                await next_client.add_request_async(request_id, req_state.prompt, params)
                req_state.stage_submit_ts[next_stage_id] = _time.time()
                continue

            request = build_engine_core_request_from_tokens(
                request_id=request_id,
                prompt=base_input,
                params=params,
                model_config=self.stage_vllm_configs[next_stage_id].model_config,
            )
            request.external_req_id = request.request_id

            self.output_processors[next_stage_id].add_request(
                request=request,
                prompt=None,
                parent_req=None,
                request_index=0,
                queue=None,
            )
            await next_client.add_request_async(request)
            req_state.stage_submit_ts[next_stage_id] = _time.time()

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
                logger.warning(f"[Orchestrator] Failed to shutdown stage {stage_id}: {e}")
