"""
Orchestrator for vLLM-Omni multi-stage runtime.

Runs inside a background thread with its own asyncio event loop.
Owns logical request progression across stage pools and handles
stage-to-stage transfer logic.

Distributed membership (replica attach/detach, hub monitoring) is
handled by :class:`MembershipController`, which is injected optionally.
"""

from __future__ import annotations

import asyncio
import os
import time as _time
from dataclasses import dataclass, field
from typing import Any

import janus
import torch
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.metrics.stats import IterationStats

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.membership_controller import MembershipController
from vllm_omni.engine.messages import (
    AbortRequestMessage,
    AddCompanionRequestMessage,
    AppendDuplexInputMessage,
    CloseDuplexSessionMessage,
    CollectiveRPCRequestMessage,
    CollectiveRPCResultMessage,
    DuplexControlResultMessage,
    EngineQueueMessage,
    ErrorMessage,
    OpenDuplexSessionMessage,
    OutputMessage,
    RegisterRemoteReplicaMessage,
    ShutdownRequestMessage,
    SignalDuplexTurnMessage,
    StageMetricsMessage,
    StageSubmissionMessage,
    UnregisterRemoteReplicaMessage,
)
from vllm_omni.engine.orchestrator_monitor import create_orch_monitor, replica_key
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.engine.omni import (
    DuplexAdapterPattern,
    DuplexInputMode,
    DuplexRuntimeCapabilities,
    DuplexSessionRuntimeManager,
    DuplexSessionRuntimeState,
    DuplexSignalSource,
    SessionMode,
    build_duplex_data_plane_prompt,
    duplex_resource_request_id,
)
from vllm_omni.metrics.prometheus import OmniRequestCounter
from vllm_omni.metrics.stat_logger import OmniPrometheusStatLogger
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def _build_terminal_empty_output(
    request_id: str,
    *,
    final_output_type: str | None,
    audio_sample_rate: int = 24000,
) -> RequestOutput:
    """Build a terminal empty output when no downstream stage input exists."""
    completion = CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    if final_output_type == "audio":
        completion.multimodal_output = {
            "audio": torch.zeros((0,), dtype=torch.float32),
            "sr": audio_sample_rate,
        }
    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )


def _coerce_int_scalar(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        for item in value:
            coerced = _coerce_int_scalar(item)
            if coerced > 0:
                return coerced
        return 0
    if hasattr(value, "item"):
        value = value.item()
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return 0
    return coerced if coerced > 0 else 0


def _infer_stage_audio_sample_rate(stage_pool: StagePool, default: int = 24000) -> int:
    """Infer the final audio stage sample rate from stage metadata when possible."""
    sample_rate_attrs = ("audio_sample_rate", "sample_rate", "sampling_rate", "output_sample_rate", "sr")
    stage_client = getattr(stage_pool, "stage_client", None)
    stage_config = getattr(stage_pool, "_stage_vllm_config", None)
    for source in (stage_client, stage_config):
        for attr in sample_rate_attrs:
            sample_rate = _coerce_int_scalar(getattr(source, attr, None))
            if sample_rate > 0:
                return sample_rate
    return default


def _minicpmo45_profile_logs_enabled() -> bool:
    return os.environ.get("MINICPMO45_PROFILE_LOGS") == "1"


def build_engine_core_request_from_tokens(
    request_id: str,
    prompt: dict[str, Any],
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    model_config: ModelConfig | None = None,
    resumable: bool = False,
    mm_features: list | None = None,
) -> OmniEngineCoreRequest:
    """Build an OmniEngineCoreRequest directly from an OmniTokensPrompt."""
    if arrival_time is None:
        arrival_time = _time.time()

    prompt_token_ids = prompt["prompt_token_ids"]

    sampling_params = None
    pooling_params = None
    if isinstance(params, SamplingParams):
        sampling_params = params.clone()
        if sampling_params.max_tokens is None and model_config is not None:
            sampling_params.max_tokens = model_config.max_model_len - len(prompt_token_ids)
    else:
        pooling_params = params.clone()

    prompt_embeds: torch.Tensor | None = prompt.get("prompt_embeds")
    raw_additional_information = prompt.get("additional_information")
    model_intermediate_buffer = prompt.get("model_intermediate_buffer")
    wire_payload: dict[str, Any] | None = None
    if isinstance(raw_additional_information, dict):
        wire_payload = dict(raw_additional_information)
    additional_info_payload = serialize_additional_information(
        wire_payload,
        log_prefix=f"build_engine_core_request_from_tokens req={request_id}",
    )

    return OmniEngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        arrival_time=arrival_time,
        lora_request=getattr(params, "lora_request", None),
        cache_salt=prompt.get("cache_salt"),
        data_parallel_rank=None,
        prompt_embeds=prompt_embeds,
        resumable=resumable,
        additional_information=additional_info_payload,
        model_intermediate_buffer=model_intermediate_buffer if isinstance(model_intermediate_buffer, dict) else None,
    )


def _duplex_force_listen_count(extra_body: object) -> int:
    """HF-compatible MiniCPM-o force-listen default."""
    raw_force_listen_count = None
    if isinstance(extra_body, dict):
        raw_force_listen_count = extra_body.get("force_listen_count")
    try:
        return 0 if raw_force_listen_count is None else max(0, int(raw_force_listen_count))
    except (TypeError, ValueError):
        return 0


@dataclass
class OrchestratorRequestState:
    """Per-request bookkeeping inside the Orchestrator."""

    request_id: str
    prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = -1
    final_output_stage_ids: set[int] = field(default_factory=set)
    finished_final_output_stage_ids: set[int] = field(default_factory=set)

    # Wall-clock timestamp when the client-facing engine request was accepted.
    request_timestamp: float = 0.0

    # Metrics: timestamp when request was submitted to each stage.
    stage_submit_ts: dict[int, float] = field(default_factory=dict)
    mm_processor_kwargs: dict | None = None
    mm_features: list | None = None
    pd_prefill_multimodal_output: dict[str, Any] | None = None

    streaming: StreamingInputState = field(default_factory=lambda: StreamingInputState())

    # Per-request pipeline timing accumulator (milliseconds)
    pipeline_timings: dict[str, float] = field(default_factory=dict)
    duplex_stage_fences: dict[int, DuplexFence] = field(default_factory=dict)


@dataclass
class StreamingInputState:
    # Flag of streaming input request
    enabled: bool = False
    # Flag of segment of streaming input finished
    segment_finished: bool = False
    # Streaming update prompt length
    new_prompt_len_snapshot: int | None = None
    # Model/bridge-specific runtime states (e.g., thinker->talker)
    bridge_states: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Runs inside a background thread's asyncio event loop."""

    # Class-level defaults so tests that bypass __init__ via object.__new__
    # don't AttributeError when transfer / counter emit paths access them.
    _running_counter: OmniRequestCounter | None = None
    _transfer_emitter: Any = None
    _stat_logger: OmniPrometheusStatLogger | None = None

    def __init__(
        self,
        request_async_queue: janus.AsyncQueue[EngineQueueMessage],
        output_async_queue: janus.AsyncQueue[dict[str, Any]],
        rpc_async_queue: janus.AsyncQueue[dict[str, Any]],
        stage_pools: list[StagePool],
        *,
        async_chunk: bool = False,
        pd_config: dict[str, Any] | None = None,
        membership_controller: MembershipController | None = None,
        running_counter: OmniRequestCounter | None = None,
        transfer_emitter: Any = None,
        log_stats: bool = False,
        enable_orch_monitor: bool = False,
    ) -> None:
        self.request_async_queue = request_async_queue
        self.output_async_queue = output_async_queue
        self.rpc_async_queue = rpc_async_queue

        self.async_chunk = bool(async_chunk)
        self.num_stages = len(stage_pools)
        self.stage_pools: list[StagePool] = stage_pools
        self._orch_monitor = create_orch_monitor(
            enabled=enable_orch_monitor,
            replica_sampler=self._sample_replica_metrics,
        )
        for stage_id, pool in enumerate(self.stage_pools):
            for replica_id in pool.live_replica_ids():
                self._orch_monitor.register_replica(stage_id, replica_id)

        # PD disaggregation state
        self._pd_pair: tuple[int, int] | None = None
        self._pd_bootstrap_addr: str | None = None
        self._pd_prefill_engine_id: str | None = None
        self._pd_kv_params: dict[str, Any] = {}
        if pd_config is not None:
            self._pd_pair = pd_config.get("pd_pair")
            self._pd_bootstrap_addr = pd_config.get("bootstrap_addr")
            self._pd_prefill_engine_id = pd_config.get("prefill_engine_id")
        self.request_states: dict[str, OrchestratorRequestState] = {}
        self._init_metrics_state(stage_pools, running_counter, transfer_emitter, log_stats=log_stats)

        self.duplex_sessions = DuplexSessionRuntimeManager()
        self._cfg_tracker = CfgCompanionTracker()
        self._stage_input_processors: dict[int, Any] = {}

        self._shutdown_event = asyncio.Event()
        self._stages_shutdown = False
        self._fatal_error: str | None = None
        self._fatal_error_stage_id: int | None = None

        # Distributed membership (optional, injected by DistStageRuntime)
        self._membership = membership_controller

    def _init_metrics_state(
        self,
        stage_pools: list[StagePool],
        running_counter: OmniRequestCounter | None,
        transfer_emitter: Any,
        log_stats: bool = False,
    ) -> None:
        """Wire up all metric-related orchestrator state.

        Sets ``self._running_counter`` and ``self._transfer_emitter``
        (both optional, used by request-add / forward paths), builds the
        ``(stage_id, replica_id) ↔ engine_idx`` lookup used at record() time,
        and best-effort constructs the ``OmniPrometheusStatLogger`` wrap
        that exposes ~37 upstream ``vllm:*`` families with per-(stage,
        replica) labels. Failure to build the wrap is logged and metrics
        are simply disabled — orchestrator construction continues so unit
        tests with a minimal ``vllm_config`` still pass.

        ``log_stats=False`` short-circuits the wrap entirely so the
        ~65 upstream ``vllm:*`` families are not registered in the
        Prometheus default registry at all. The per-step record() path
        already no-ops on ``scheduler_stats is None`` (which is what
        the upstream scheduler returns when its own log_stats is False),
        so this gate is mainly to keep the ``/metrics`` surface clean
        when the user did not request stats.
        """
        self._running_counter = running_counter
        self._transfer_emitter = transfer_emitter

        # Flat engine_idx ↔ (stage, replica) maps. The reverse map is
        # consulted at record() time to translate the orchestrator's
        # (stage_id, replica_id) loop variables into an engine_idx the
        # underlying PrometheusStatLogger can address.
        stage_replica_map: dict[int, tuple[str, str]] = {}
        self._stage_replica_to_engine_idx: dict[tuple[int, int], int] = {}
        flat_idx = 0
        for stage_id, pool in enumerate(stage_pools):
            for replica_id in range(pool.num_replicas):
                stage_replica_map[flat_idx] = (str(stage_id), str(replica_id))
                self._stage_replica_to_engine_idx[(stage_id, replica_id)] = flat_idx
                flat_idx += 1

        if not log_stats:
            self._stat_logger = None
            return

        vllm_config_for_stats = next(
            (p.stage_vllm_config for p in stage_pools if p.stage_vllm_config is not None),
            None,
        )
        if vllm_config_for_stats is None:
            self._stat_logger = None
            return
        try:
            self._stat_logger = OmniPrometheusStatLogger(
                vllm_config=vllm_config_for_stats,
                stage_replica_map=stage_replica_map,
            )
        except Exception:
            # Minimal vllm_config in unit-test contexts can lack fields the
            # upstream PrometheusStatLogger expects. Skip wrap rather than
            # break orchestrator construction.
            logger.exception("[Orchestrator] OmniPrometheusStatLogger init failed; metrics wrap disabled")
            self._stat_logger = None

    async def run(self) -> None:
        """Main entry point for the Orchestrator event loop."""
        logger.info("[Orchestrator] Starting event loop")

        request_task = asyncio.create_task(self._request_handler(), name="orchestrator-request-handler")
        output_task = asyncio.create_task(
            self._orchestration_output_handler(),
            name="orchestrator-stage-output-handler",
        )

        # Start membership watcher if distributed mode is active.
        membership_watcher: asyncio.Task[None] | None = None
        if self._membership is not None:
            self._membership.install_unregister_handlers(
                output_queue=self.output_async_queue,
                cleanup_callback=lambda ids: self._cleanup_request_ids(ids, abort=True),
            )
            membership_watcher = self._membership.start()

        tasks = [request_task, output_task]
        if membership_watcher is not None:
            tasks.append(membership_watcher)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            raise
        except EngineDeadError as e:
            # EngineDeadError from _orchestration_loop means the diffusion
            # engine died.  All pending requests were already notified and
            # _shutdown_event was already set by the loop's handler.
            # During teardown this is expected; the finally block handles
            # proper cleanup.  Do not re-raise.
            logger.info("[Orchestrator] Engine dead during shutdown: %s", e)
        except Exception:
            logger.exception("[Orchestrator] Fatal error in orchestrator tasks")
            raise
        finally:
            self._shutdown_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass

            if self._fatal_error is not None:
                await self._drain_pending_requests_on_fatal()

            if self._membership is not None:
                await self._membership.drain_tasks(timeout=10.0)
                self._membership.shutdown()

            self._orch_monitor.flush()
            self._shutdown_stages()

            loop = asyncio.get_running_loop()
            pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task() and not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    # ---- Request handling ----

    async def _request_handler(self) -> None:
        """Read messages from the main thread via request_async_queue."""
        while True:
            msg = await self.request_async_queue.get()
            msg_type = msg.type

            if msg_type == "add_request":
                await self._handle_add_request(msg)
            elif msg_type == "streaming_update":
                await self._handle_streaming_update(msg)
            elif msg_type == "add_companion_request":
                await self._handle_add_companion(msg)
            elif msg_type == "open_duplex_session":
                await self._handle_open_duplex_session(msg)
            elif msg_type == "append_duplex_input":
                await self._handle_append_duplex_input(msg)
            elif msg_type == "signal_duplex_turn":
                await self._handle_signal_duplex_turn(msg)
            elif msg_type == "close_duplex_session":
                await self._handle_close_duplex_session(msg)
            elif msg_type == "abort":
                await self._handle_abort(msg)
            elif msg_type == "collective_rpc":
                await self._handle_collective_rpc(msg)
            elif isinstance(msg, RegisterRemoteReplicaMessage):
                if self._membership is not None:
                    await self._membership.handle_register(msg.stage_id, msg.replica_id)
                    self._orch_monitor.register_replica(msg.stage_id, msg.replica_id)
            elif isinstance(msg, UnregisterRemoteReplicaMessage):
                if self._membership is not None:
                    await self._membership.handle_unregister(msg.stage_id, msg.input_addr)
            elif isinstance(msg, ShutdownRequestMessage):
                logger.info("[Orchestrator] Received shutdown signal")
                self._shutdown_event.set()
                # Pre-mark stage clients as shutting down to prevent
                # proc_monitor daemon threads from flagging normal
                # process exit as EngineDeadError during teardown.
                for pool in self.stage_pools:
                    for client in pool.clients:
                        if hasattr(client, "_shutting_down"):
                            client._shutting_down = True
                # Stage teardown runs once in run()'s finally after the
                # orchestration loop observes _shutdown_event and exits.
                break
            else:
                logger.warning("[Orchestrator] Unknown message type: %s", msg_type)

    @staticmethod
    def _coerce_duplex_capabilities(raw: dict[str, object]) -> DuplexRuntimeCapabilities:
        def enum_set(enum_cls: Any, values: object) -> set[Any]:
            if not isinstance(values, list):
                return set()
            result = set()
            for value in values:
                try:
                    result.add(enum_cls(value))
                except ValueError:
                    logger.warning("[Orchestrator] Ignoring unknown duplex capability value: %s", value)
            return result

        adapter_patterns = enum_set(DuplexAdapterPattern, raw.get("adapter_patterns"))
        input_modes = enum_set(DuplexInputMode, raw.get("input_modes")) or {DuplexInputMode.TURN_COMMIT_ONLY}
        signal_sources = enum_set(DuplexSignalSource, raw.get("signal_sources")) or {
            DuplexSignalSource.CLIENT_EVENT,
            DuplexSignalSource.SERVER_POLICY,
        }
        chunk_period = raw.get("chunk_period_ms")
        target_latency = raw.get("target_barge_in_latency_ms")
        return DuplexRuntimeCapabilities(
            adapter_patterns=adapter_patterns,
            input_modes=input_modes,
            signal_sources=signal_sources,
            supports_kv_lease=bool(raw.get("supports_kv_lease", False)),
            supports_core_kv_lease=bool(raw.get("supports_core_kv_lease", False)),
            supports_model_internal_state=bool(raw.get("supports_model_internal_state", False)),
            supports_stage_resumption=bool(raw.get("supports_stage_resumption", False)),
            supports_scheduler_native_append=bool(raw.get("supports_scheduler_native_append", False)),
            supports_core_resumable_request=bool(raw.get("supports_core_resumable_request", False)),
            supports_stage_connector_handoff=bool(raw.get("supports_stage_connector_handoff", False)),
            supports_independent_io_streams=bool(raw.get("supports_independent_io_streams", False)),
            supports_realtime_endpoint=bool(raw.get("supports_realtime_endpoint", False)),
            supports_multi_session=bool(raw.get("supports_multi_session", False)),
            supports_multi_session_same_replica=bool(raw.get("supports_multi_session_same_replica", False)),
            supports_barge_in=bool(raw.get("supports_barge_in", True)),
            supports_playback_ack=bool(raw.get("supports_playback_ack", True)),
            supports_audio_truncate=bool(raw.get("supports_audio_truncate", False)),
            implementation_level=str(raw.get("implementation_level") or "serving_session_adapter"),
            stage_handoff_transport=(
                str(raw.get("stage_handoff_transport")) if raw.get("stage_handoff_transport") is not None else None
            ),
            chunk_period_ms=int(chunk_period) if isinstance(chunk_period, int | float) else None,
            target_barge_in_latency_ms=int(target_latency) if isinstance(target_latency, int | float) else None,
        )

    async def _handle_open_duplex_session(self, msg: OpenDuplexSessionMessage) -> None:
        try:
            session_mode = SessionMode(msg.session_mode)
            capabilities = self._coerce_duplex_capabilities(msg.capabilities)
            self.duplex_sessions.open_session(
                msg.fence,
                session_mode=session_mode,
                capabilities=capabilities,
                session_config=msg.session_config,
            )
            session = self.duplex_sessions.require(msg.session_id)
            request_state = self._ensure_duplex_stage_request_state(session, stage_id=0)
            await self._put_duplex_control_result(
                msg.control_id,
                fence=msg.fence,
                operation="open",
                session_id=msg.session_id,
                stage_results=[
                    {
                        "stage_id": -1,
                        "replica_id": -1,
                        "result": {
                            "supported": True,
                            "implementation_level": capabilities.implementation_level,
                            "data_plane_session": True,
                            "session_mode": session_mode.value,
                            "scheduler_request_context": request_state is not None,
                            "request_id": request_state.request_id if request_state is not None else None,
                            "stage_request_topology": self._duplex_stage_request_topology(session),
                        },
                    }
                ],
            )
        except Exception as exc:
            logger.exception("[Orchestrator] open_duplex_session failed: %s", exc)
            await self._put_duplex_control_result(
                msg.control_id,
                fence=msg.fence,
                operation="open",
                session_id=msg.session_id,
                stage_results=[],
                error=str(exc),
            )

    async def _handle_append_duplex_input(self, msg: AppendDuplexInputMessage) -> None:
        stage_results: list[dict[str, object]] = []
        try:
            session = self.duplex_sessions.require(msg.session_id)
            if msg.expected_epoch is not None and msg.expected_epoch != msg.fence.epoch:
                raise ValueError("expected_epoch must match fence.epoch")
            mode = DuplexInputMode(msg.mode)
            update = session.append_input(
                msg.payload,
                mode=mode,
                fence=msg.fence,
                final=msg.final,
            )
            stage_results = await self._append_duplex_input_via_data_plane(
                msg,
                session=session,
                seq=update.seq,
                turn_id=update.turn_id,
                turn_seq=update.turn_seq,
                mode=mode,
            )
            await self._put_duplex_control_result(
                msg.control_id,
                fence=msg.fence,
                operation="append",
                session_id=msg.session_id,
                stage_results=stage_results,
            )
        except Exception as exc:
            logger.exception("[Orchestrator] append_duplex_input failed: %s", exc)
            await self._put_duplex_control_result(
                msg.control_id,
                fence=msg.fence,
                operation="append",
                session_id=msg.session_id,
                stage_results=[],
                error=str(exc),
            )

    async def _append_duplex_input_via_data_plane(
        self,
        msg: AppendDuplexInputMessage,
        *,
        session: DuplexSessionRuntimeState,
        seq: int,
        turn_id: int,
        turn_seq: int,
        mode: DuplexInputMode,
    ) -> list[dict[str, object]]:
        if not self.stage_pools:
            return [
                {
                    "stage_id": -1,
                    "replica_id": -1,
                    "result": {
                        "supported": False,
                        "error": "duplex_data_plane_has_no_stage",
                    },
                }
            ]

        stage_id = 0
        pool = self.stage_pools[stage_id]
        request_id = self._duplex_stage_request_id(msg.fence, stage_id=stage_id)
        existing_binding = session.stage_bindings.get(stage_id)
        already_submitted = existing_binding is not None and existing_binding.request_id == request_id
        req_state = self._ensure_duplex_stage_request_state(session, stage_id=stage_id)
        if req_state is None:
            raise RuntimeError("duplex_data_plane_has_no_stage")
        self._sync_duplex_bridge_state(req_state, session=session, fence=msg.fence)

        prompt = self._build_duplex_data_plane_prompt(
            request_id=request_id,
            session=session,
            seq=seq,
            turn_seq=turn_seq,
            mode=mode,
            payload=msg.payload,
            final=msg.final,
        )
        request = build_engine_core_request_from_tokens(
            request_id=request_id,
            prompt=prompt,
            params=req_state.sampling_params_list[stage_id],
            model_config=pool.stage_vllm_config.model_config,
            resumable=True,
        )
        request.external_req_id = request.request_id
        if already_submitted:
            replica_id = await pool.submit_update(request_id, req_state, request)
        else:
            replica_id = await pool.submit_initial(request_id, req_state, request, prompt_text=None)
        self._record_duplex_stage_submission(
            stage_id,
            request_id,
            replica_id,
            req_state,
            already_submitted=already_submitted,
        )
        req_state.stage_submit_ts[stage_id] = _time.time()
        return [
            {
                "stage_id": stage_id,
                "replica_id": replica_id,
                "result": {
                    "supported": True,
                    "implementation_level": session.capabilities.implementation_level,
                    "data_plane_append": True,
                    "request_id": request_id,
                    "response_stage_id": req_state.final_stage_id,
                    "stage_request_topology": self._duplex_stage_request_topology(session),
                    "seq": seq,
                    "turn_id": turn_id,
                    "response_seq": msg.fence.response_seq,
                    "turn_seq": turn_seq,
                    "mode": mode.value,
                    "resumable": True,
                },
            }
        ]

    @staticmethod
    def _duplex_stage_max_tokens(session_config: dict[str, Any], stage_id: int) -> int | None:
        raw = session_config.get("duplex_stage_max_tokens")
        if raw is None:
            extra_body = session_config.get("extra_body")
            if isinstance(extra_body, dict):
                raw = extra_body.get("duplex_stage_max_tokens")
        value: object | None = None
        if isinstance(raw, dict):
            value = raw.get(stage_id)
            if value is None:
                value = raw.get(str(stage_id))
        elif isinstance(raw, list | tuple) and stage_id < len(raw):
            value = raw[stage_id]
        if isinstance(value, int | float) and int(value) > 0:
            return int(value)
        return None

    @staticmethod
    def _duplex_stage_sampling_overrides(session_config: dict[str, Any], stage_id: int) -> dict[str, Any]:
        raw = session_config.get("duplex_stage_sampling_params")
        if raw is None:
            extra_body = session_config.get("extra_body")
            if isinstance(extra_body, dict):
                raw = extra_body.get("duplex_stage_sampling_params")
        value: object | None = None
        if isinstance(raw, dict):
            value = raw.get(stage_id)
            if value is None:
                value = raw.get(str(stage_id))
        elif isinstance(raw, list | tuple) and stage_id < len(raw):
            value = raw[stage_id]
        return dict(value) if isinstance(value, dict) else {}

    def _duplex_sampling_params_list(self, session: DuplexSessionRuntimeState) -> list[Any]:
        params: list[Any] = []
        for stage_id, pool in enumerate(self.stage_pools):
            stage_params = pool.stage_client.default_sampling_params
            max_tokens = self._duplex_stage_max_tokens(session.session_config, stage_id)
            overrides = self._duplex_stage_sampling_overrides(session.session_config, stage_id)
            if isinstance(stage_params, SamplingParams) and (max_tokens is not None or overrides):
                stage_params = stage_params.clone()
                if max_tokens is not None:
                    stage_params.max_tokens = max_tokens
                for name, value in overrides.items():
                    if hasattr(stage_params, name):
                        setattr(stage_params, name, value)
                        if name == "stop_token_ids":
                            all_stop_token_ids = getattr(stage_params, "_all_stop_token_ids", None)
                            if isinstance(all_stop_token_ids, set):
                                all_stop_token_ids.update(int(token_id) for token_id in value)
            params.append(stage_params)
        return params

    @staticmethod
    def _duplex_stage_request_id(fence: DuplexFence, *, stage_id: int) -> str:
        return duplex_resource_request_id(fence, f"stage{stage_id}")

    def _ensure_duplex_stage_request_state(
        self,
        session: DuplexSessionRuntimeState,
        *,
        stage_id: int,
    ) -> OrchestratorRequestState | None:
        if not self.stage_pools or stage_id >= len(self.stage_pools):
            return None
        request_id = self._duplex_stage_request_id(session.fence, stage_id=stage_id)
        req_state = self.request_states.get(request_id)
        if req_state is not None:
            self._sync_duplex_bridge_state(req_state, session=session, fence=session.fence)
            return req_state
        req_state = OrchestratorRequestState(
            request_id=request_id,
            prompt=None,
            sampling_params_list=self._duplex_sampling_params_list(session),
            final_stage_id=len(self.stage_pools) - 1,
        )
        req_state.streaming.enabled = True
        self._sync_duplex_bridge_state(req_state, session=session, fence=session.fence)
        self.request_states[request_id] = req_state
        return req_state

    @staticmethod
    def _sync_duplex_bridge_state(
        req_state: OrchestratorRequestState,
        *,
        session: DuplexSessionRuntimeState,
        fence: DuplexFence,
    ) -> None:
        duplex_state = req_state.streaming.bridge_states.setdefault("duplex", {})
        if not isinstance(duplex_state, dict):
            duplex_state = {}
            req_state.streaming.bridge_states["duplex"] = duplex_state
        previous_epoch = duplex_state.get("epoch")
        if not isinstance(duplex_state.get("model_turn_id"), int) or previous_epoch != fence.epoch:
            duplex_state["model_turn_id"] = fence.turn_id
        duplex_state.update(
            {
                "session_id": session.session_id,
                "fence": fence,
                "epoch": fence.epoch,
                "turn_id": fence.turn_id,
                "response_seq": fence.response_seq,
                "session_config": dict(session.session_config),
            }
        )

    def _duplex_stage_request_topology(self, session: DuplexSessionRuntimeState) -> dict[str, Any]:
        """Return the session's scheduler data-plane request topology.

        One duplex session owns one logical resumable scheduler request for the
        current epoch. Each physical stage keeps its own runner/scheduler state
        for that logical request id after it is admitted. Downstream stages are
        admitted on the first handoff and then updated until the final segment.
        This is deliberately not a persistent core KV lease.
        """
        logical_request_id = self._duplex_stage_request_id(session.fence, stage_id=0)
        created = logical_request_id in self.request_states
        stage_requests: list[dict[str, Any]] = []
        all_stage_actors_submitted = bool(self.stage_pools)
        for stage_id in range(len(self.stage_pools)):
            binding = session.stage_bindings.get(stage_id)
            is_stage0 = stage_id == 0
            if binding is None:
                all_stage_actors_submitted = False
            stage_requests.append(
                {
                    "stage_id": stage_id,
                    "logical_request_id": logical_request_id,
                    "planned_request_id": logical_request_id,
                    "bound_request_id": binding.request_id if binding is not None else None,
                    "created": created,
                    "submitted": binding is not None,
                    "replica_id": binding.replica_id if binding is not None else None,
                    "resumable": True,
                    "admission": "open" if is_stage0 else "first_handoff",
                    "update_policy": "append_streaming_input" if is_stage0 else "segment_payload_update",
                    "stage_actor_state": "submitted" if binding is not None else "planned",
                    "long_lived_stage_actor": is_stage0 or binding is not None,
                }
            )
        return {
            "kind": "scheduler_data_plane_resumable_stage_pipeline",
            "persistent_core_kv_lease": False,
            "one_long_lived_request_per_stage": all_stage_actors_submitted,
            "stage0_long_lived_request": True,
            "stage_actors_become_long_lived_after_first_handoff": True,
            "downstream_stage_update_policy": "segment_payload_update",
            "logical_request_id": logical_request_id,
            "stage_request_id_scope": "epoch_scoped_logical_request_id_per_stage_runner_state",
            "stage_requests": stage_requests,
        }

    @staticmethod
    def _build_duplex_data_plane_prompt(
        *,
        request_id: str,
        session: DuplexSessionRuntimeState,
        seq: int,
        turn_seq: int,
        mode: DuplexInputMode,
        payload: object,
        final: bool,
    ) -> dict[str, Any]:
        return build_duplex_data_plane_prompt(
            request_id=request_id,
            fence=session.fence,
            session_config=session.session_config,
            seq=seq,
            turn_seq=turn_seq,
            mode=mode,
            payload=payload,
            final=final,
        )

    async def _handle_signal_duplex_turn(self, msg: SignalDuplexTurnMessage) -> None:
        stage_results: list[dict[str, object]] = []
        try:
            session = self.duplex_sessions.require(msg.session_id)
            session.accept_fence(msg.fence)
            if msg.event in {"barge_in", "input.cancel", "response.cancel"}:
                bound_stage_requests = [item for item in session.stage_bindings.items() if item[1].fence == msg.fence]
                stage_results.extend(
                    await self._signal_bound_duplex_stage_replicas(
                        msg,
                        bound_stage_requests=bound_stage_requests,
                    )
                )
                stale_request_ids = session.release_fence(msg.fence)
                if stale_request_ids:
                    await self._cleanup_request_ids(stale_request_ids, abort=True)
            stage_results.append(
                {
                    "stage_id": -1,
                    "replica_id": -1,
                    "result": {
                        "supported": True,
                        "data_plane_signal": True,
                        "event": msg.event,
                        "fence": msg.fence,
                    },
                }
            )
            await self._put_duplex_control_result(
                msg.control_id,
                fence=msg.fence,
                operation="signal",
                session_id=msg.session_id,
                stage_results=stage_results,
            )
        except Exception as exc:
            logger.exception("[Orchestrator] signal_duplex_turn failed: %s", exc)
            await self._put_duplex_control_result(
                msg.control_id,
                fence=msg.fence,
                operation="signal",
                session_id=msg.session_id,
                stage_results=[],
                error=str(exc),
            )

    async def _signal_bound_duplex_stage_replicas(
        self,
        msg: SignalDuplexTurnMessage,
        *,
        bound_stage_requests: list[tuple[int, Any]],
    ) -> list[dict[str, object]]:
        stage_results: list[dict[str, object]] = []
        payload = dict(msg.payload or {})
        payload.setdefault("turn_id", msg.fence.turn_id)
        payload.setdefault("response_seq", msg.fence.response_seq)
        for stage_id, binding in bound_stage_requests:
            if stage_id < 0 or stage_id >= len(self.stage_pools):
                continue
            replica_id = getattr(binding, "replica_id", None)
            if replica_id is None:
                continue
            try:
                result = await self.stage_pools[stage_id].collective_rpc(
                    int(replica_id),
                    "signal_duplex_turn_async",
                    timeout=msg.timeout,
                    kwargs={
                        "session_id": msg.session_id,
                        "epoch": msg.fence.epoch,
                        "event": msg.event,
                        "payload": dict(payload),
                    },
                )
            except Exception as exc:
                result = {
                    "supported": False,
                    "error": str(exc),
                }
            if isinstance(result, dict) and result.get("supported") is False and not result.get("error"):
                result = {
                    "supported": True,
                    "ignored_unsupported_stage_signal": True,
                    "event": msg.event,
                    "reason": result.get("reason"),
                }
            stage_results.append(
                {
                    "stage_id": stage_id,
                    "replica_id": int(replica_id),
                    "result": result,
                }
            )
        return stage_results

    async def _handle_close_duplex_session(self, msg: CloseDuplexSessionMessage) -> None:
        stage_results: list[dict[str, object]] = []
        try:
            session = self.duplex_sessions.get(msg.session_id)
            if session is None:
                await self._put_duplex_control_result(
                    msg.control_id,
                    fence=msg.fence,
                    operation="close",
                    session_id=msg.session_id,
                    stage_results=[],
                )
                return
            stale_request_ids = session.stage_request_ids()
            self.duplex_sessions.close_session(msg.fence)
            if stale_request_ids:
                await self._cleanup_request_ids(stale_request_ids, abort=True)
            stage_results.append(
                {
                    "stage_id": -1,
                    "replica_id": -1,
                    "result": {
                        "supported": True,
                        "data_plane_close": True,
                        "reason": msg.reason,
                    },
                }
            )
            await self._put_duplex_control_result(
                msg.control_id,
                fence=msg.fence,
                operation="close",
                session_id=msg.session_id,
                stage_results=stage_results,
            )
        except Exception as exc:
            logger.exception("[Orchestrator] close_duplex_session failed: %s", exc)
            await self._put_duplex_control_result(
                msg.control_id,
                fence=msg.fence,
                operation="close",
                session_id=msg.session_id,
                stage_results=[],
                error=str(exc),
            )

    @classmethod
    def _iter_duplex_result_dicts(cls, result: object):
        if isinstance(result, dict):
            yield result
            return
        if isinstance(result, list | tuple):
            for item in result:
                yield from cls._iter_duplex_result_dicts(item)

    @classmethod
    def _duplex_result_has_passive_stage(cls, result: object) -> bool:
        if isinstance(result, dict):
            if result.get("passive_stage") is True:
                return True
            return any(cls._duplex_result_has_passive_stage(value) for value in result.values())
        if isinstance(result, list | tuple):
            return any(cls._duplex_result_has_passive_stage(item) for item in result)
        return False

    @classmethod
    def _duplex_result_counts(cls, stage_results: list[dict[str, object]]) -> tuple[int, int, int]:
        unsupported_count = 0
        error_count = 0
        passive_count = 0
        for item in stage_results:
            for result in cls._iter_duplex_result_dicts(item.get("result")):
                if result.get("supported") is False:
                    unsupported_count += 1
                if result.get("error"):
                    error_count += 1
                if cls._duplex_result_has_passive_stage(result.get("native_result")):
                    passive_count += 1
        return unsupported_count, error_count, passive_count

    async def _put_duplex_control_result(
        self,
        control_id: str,
        *,
        fence: DuplexFence,
        operation: str,
        session_id: str,
        stage_results: list[dict[str, object]],
        error: str | None = None,
    ) -> None:
        if error is not None:
            stage_results = [
                {
                    "stage_id": -1,
                    "replica_id": -1,
                    "result": {"supported": False, "error": error},
                }
            ]
        unsupported_count, error_count, passive_count = self._duplex_result_counts(stage_results)
        await self.rpc_async_queue.put(
            DuplexControlResultMessage(
                control_id=control_id,
                fence=fence,
                operation=operation,
                session_id=session_id,
                ok=error_count == 0 and unsupported_count == 0,
                stage_results=stage_results,
                unsupported_count=unsupported_count,
                error_count=error_count,
                passive_count=passive_count,
            )
        )

    async def _handle_add_request(self, msg: StageSubmissionMessage) -> None:
        """Handle an add_request message from the main thread."""
        stage_id = 0
        request_id = msg.request_id
        prompt = msg.prompt
        original_prompt = msg.original_prompt
        sampling_params_list = msg.sampling_params_list
        if not sampling_params_list:
            raise ValueError(f"Missing sampling params for stage 0. Got {len(sampling_params_list)} stage params.")
        final_stage_id = msg.final_stage_id
        final_output_stage_ids = set(msg.final_output_stage_ids or [final_stage_id])

        logger.debug(
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

        req_state = OrchestratorRequestState(
            request_id=request_id,
            prompt=original_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            final_output_stage_ids=final_output_stage_ids,
            request_timestamp=float(msg.request_timestamp or _time.time()),
            mm_features=getattr(prompt, "mm_features", None),
        )
        self.request_states[request_id] = req_state
        if self._running_counter is not None:
            self._running_counter.increment()
        req_state.streaming.enabled = bool(getattr(prompt, "resumable", False))
        req_state.stage_submit_ts[stage_id] = _time.time()
        enqueue_ts = msg.enqueue_ts
        if enqueue_ts > 0:
            req_state.pipeline_timings["queue_wait_ms"] = (_time.perf_counter() - enqueue_ts) * 1000.0
        preprocess_ms = msg.preprocess_ms
        if preprocess_ms > 0:
            req_state.pipeline_timings["preprocess_ms"] = preprocess_ms
        await self.stage_pools[stage_id].submit_initial(
            request_id,
            req_state,
            prompt,
            prompt_text=msg.output_prompt_text,
        )

        if self.async_chunk and stage_id == 0 and final_stage_id > 0:
            await self._prewarm_async_chunk_stages(request_id, prompt, req_state)

    async def _handle_streaming_update(self, msg: StageSubmissionMessage) -> None:
        """Handle a streaming_update message for an existing request."""
        stage_id = 0
        request_id = msg.request_id
        request = msg.prompt
        final_stage_id = msg.final_stage_id
        req_state = self.request_states.get(request_id)
        if req_state is None:
            logger.warning(
                "[Orchestrator] streaming_update for unknown req=%s, falling back to add_request",
                request_id,
            )
            fallback_msg = StageSubmissionMessage(
                type="add_request",
                request_id=msg.request_id,
                prompt=msg.prompt,
                original_prompt=msg.original_prompt,
                output_prompt_text=msg.output_prompt_text,
                sampling_params_list=msg.sampling_params_list,
                final_stage_id=msg.final_stage_id,
                final_output_stage_ids=msg.final_output_stage_ids,
                preprocess_ms=msg.preprocess_ms,
                request_timestamp=msg.request_timestamp,
                enqueue_ts=msg.enqueue_ts,
            )
            await self._handle_add_request(fallback_msg)
            return

        if msg.sampling_params_list:
            req_state.sampling_params_list = msg.sampling_params_list

        req_state.streaming.enabled = True
        req_state.stage_submit_ts[stage_id] = _time.time()
        await self.stage_pools[stage_id].submit_update(
            request_id,
            req_state,
            request,
            prompt_text=msg.output_prompt_text,
        )

        if self.async_chunk and stage_id == 0 and final_stage_id > 0:
            await self._prewarm_async_chunk_stages(request_id, request, req_state)

    async def _handle_add_companion(self, msg: AddCompanionRequestMessage) -> None:
        """Handle an add_companion_request message: submit companion to stage 0."""
        companion_id = msg.companion_id
        parent_id = msg.parent_id
        role = msg.role
        companion_prompt = msg.prompt
        sampling_params_list = msg.sampling_params_list

        parent_state = self.request_states.get(parent_id)
        if parent_state is None:
            logger.info(
                "[Orchestrator] Dropping CFG companion %s (role=%s): parent %s is no longer active",
                companion_id,
                role,
                parent_id,
            )
            return

        self._cfg_tracker.register_companion(parent_id, role, companion_id)

        companion_state = OrchestratorRequestState(
            request_id=companion_id,
            prompt=companion_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=0,
            final_output_stage_ids={0},
            request_timestamp=parent_state.request_timestamp,
        )
        self.request_states[companion_id] = companion_state
        companion_state.stage_submit_ts[0] = _time.time()
        companion_replica_id = await self.stage_pools[0].submit_initial(
            companion_id,
            companion_state,
            companion_prompt,
            prompt_text=msg.companion_prompt_text,
            affinity_request_id=parent_id,
        )

        logger.info(
            "[Orchestrator] CFG companion submitted: %s (role=%s, parent=%s, stage-0 replica-%s)",
            companion_id,
            role,
            parent_id,
            companion_replica_id,
        )

    async def _handle_abort(self, msg: AbortRequestMessage) -> None:
        """Handle an abort message from the main thread."""
        request_ids = msg.request_ids
        await self._cleanup_request_ids(
            self._cfg_tracker.abort_parents(request_ids),
            abort=True,
        )
        logger.info("[Orchestrator] Aborted request(s) %s", request_ids)

    async def _abort_request_ids(self, request_ids: list[str]) -> None:
        """Forward abort requests to all stage pools."""
        if not request_ids:
            return
        for pool in self.stage_pools:
            await pool.abort_requests(request_ids)

    def _release_request_bindings(self, request_ids: list[str]) -> None:
        """Release all stage-local route bindings for the given request ids."""
        for pool in self.stage_pools:
            pool.release_bindings(request_ids)

    async def _handle_collective_rpc(self, msg: CollectiveRPCRequestMessage) -> None:
        """Handle a control-plane RPC request from the main thread."""
        rpc_id = msg.rpc_id
        method = msg.method
        timeout = msg.timeout
        args = tuple(msg.args)
        kwargs = dict(msg.kwargs or {})
        requested_stage_ids = msg.stage_ids

        target_pools: list[StagePool] = []
        if requested_stage_ids is None:
            target_pools.extend(self.stage_pools)
        else:
            for lid in requested_stage_ids:
                if not (0 <= lid < self.num_stages):
                    logger.warning("[Orchestrator] collective_rpc: ignoring invalid stage_id %s", lid)
                    continue
                target_pools.append(self.stage_pools[lid])

        results: list[Any] = []
        stage_ids: list[int] = []
        for pool in target_pools:
            for replica_id in pool.live_replica_ids():
                stage_result = await pool.collective_rpc(
                    replica_id=replica_id,
                    method=method,
                    timeout=timeout,
                    args=args,
                    kwargs=kwargs,
                )
                stage_ids.append(pool.stage_id)
                results.append(stage_result)

        await self.rpc_async_queue.put(
            CollectiveRPCResultMessage(
                rpc_id=rpc_id,
                method=method,
                stage_ids=stage_ids,
                results=results,
            )
        )

    # ---- Orchestration loop ----

    def _sample_replica_metrics(self) -> dict[str, tuple[int, int]]:
        samples: dict[str, tuple[int, int]] = {}
        for stage_id, pool in enumerate(self.stage_pools):
            for replica_id in pool.live_replica_ids():
                key = replica_key(stage_id, replica_id)
                samples[key] = pool.replica_monitor_sample(replica_id)
        return samples

    async def _orchestration_output_handler(self) -> None:
        """Poll all stages, handle transfers, send final outputs to main."""
        try:
            await self._orchestration_loop()
        except asyncio.CancelledError:
            logger.debug("[Orchestrator] _orchestration_output_handler cancelled")
            return

    async def _orchestration_loop(self) -> None:
        """Poll stage pools and route logical outputs."""
        while not self._shutdown_event.is_set():
            idle = True
            for stage_id in range(self.num_stages):
                pool = self.stage_pools[stage_id]
                for replica_id in pool.live_replica_ids():
                    if self._shutdown_event.is_set():
                        return

                    if pool.stage_type == "diffusion":
                        output = pool.poll_diffusion_output(replica_id)
                        if output is None:
                            continue

                        pool.record_output_timestamps([output])
                        await self._handle_processed_outputs(stage_id, replica_id, [output])
                        idle = False
                    else:
                        try:
                            raw_outputs = await pool.poll_llm_raw_output(replica_id, timeout_s=0.001)
                            if raw_outputs is None:
                                continue

                            await self._handle_kv_ready_raw_outputs(stage_id, raw_outputs)
                            for eco in raw_outputs.outputs:
                                req_state = self.request_states.get(getattr(eco, "request_id", None))
                                if req_state is None or not req_state.streaming.enabled:
                                    continue
                                req_state.streaming.segment_finished = bool(getattr(eco, "is_segment_finished", False))
                                if self._is_duplex_session_request(req_state) and _minicpmo45_profile_logs_enabled():
                                    logger.info(
                                        "[Orchestrator] duplex raw output: stage=%s "
                                        "req=%s finished=%s segment_finished=%s "
                                        "new_tokens=%s",
                                        stage_id,
                                        getattr(eco, "request_id", None),
                                        getattr(eco, "finished", None),
                                        getattr(eco, "is_segment_finished", None),
                                        getattr(eco, "new_token_ids", None),
                                    )
                                req_state.streaming.new_prompt_len_snapshot = getattr(
                                    eco,
                                    "new_prompt_len_snapshot",
                                    None,
                                )
                                if req_state.streaming.enabled:
                                    await self._apply_raw_terminal_stage_finish(stage_id, eco, req_state)
                            # OmniSchedulerMixin.make_stats() already throttles
                            # per-scheduler at 1 Hz, so raw_outputs.scheduler_stats
                            # being non-None means this replica passed its own gate.
                            # A second global throttle here would drop stats for
                            # other (stage, replica) pairs in the same 1s window.
                            record_stats = self._stat_logger is not None and raw_outputs.scheduler_stats is not None
                            iteration_stats = IterationStats() if record_stats else None
                            raw_output = await pool.process_llm_raw_outputs(
                                replica_id,
                                raw_outputs,
                                iteration_stats=iteration_stats,
                            )
                            if record_stats:
                                self._stat_logger.record(
                                    raw_outputs.scheduler_stats,
                                    iteration_stats,
                                    engine_idx=self._stage_replica_to_engine_idx[(stage_id, replica_id)],
                                )
                        except asyncio.CancelledError:
                            raise
                        except EngineDeadError as e:
                            logger.error(
                                "[Orchestrator] Stage-%s replica-%s is dead: %s",
                                stage_id,
                                replica_id,
                                e,
                            )
                            affected_session_ids = pool.duplex_session_ids_for_replica(replica_id)
                            affected_request_ids = pool.mark_replica_unavailable(replica_id)
                            if affected_session_ids:
                                for session_id in affected_session_ids:
                                    session = self.duplex_sessions.get(session_id)
                                    if session is None:
                                        continue
                                    stale_request_ids = session.stage_request_ids()
                                    self.duplex_sessions.close_session(session.fence)
                                    affected_request_ids.extend(stale_request_ids)
                                    logger.warning(
                                        "[Orchestrator] closed duplex session %s after stage-%s replica-%s died; "
                                        "stale_request_ids=%s",
                                        session_id,
                                        stage_id,
                                        replica_id,
                                        stale_request_ids,
                                    )
                            affected_request_ids = list(dict.fromkeys(affected_request_ids))
                            if pool.available_replica_ids():
                                for req_id in affected_request_ids:
                                    await self.output_async_queue.put(
                                        ErrorMessage(
                                            error=str(e),
                                            fatal=False,
                                            request_id=req_id,
                                            stage_id=stage_id,
                                        )
                                    )
                                await self._cleanup_request_ids(
                                    affected_request_ids,
                                    close_duplex_sessions=True,
                                )
                                continue

                            self._fatal_error = str(e)
                            self._fatal_error_stage_id = stage_id
                            for req_id in affected_request_ids:
                                await self.output_async_queue.put(
                                    ErrorMessage(
                                        error=str(e),
                                        fatal=True,
                                        request_id=req_id,
                                        stage_id=stage_id,
                                    )
                                )
                            await self._cleanup_request_ids(
                                affected_request_ids,
                                close_duplex_sessions=True,
                            )
                            self._shutdown_event.set()
                            raise
                        except Exception:
                            if self._shutdown_event.is_set():
                                return
                            logger.exception(
                                "[Orchestrator] Stage-%s replica-%s processing failed",
                                stage_id,
                                replica_id,
                            )
                            raise

                        await self._handle_processed_outputs(stage_id, replica_id, raw_output)
                        idle = False

            self._orch_monitor.note_loop(idle=idle)
            if idle:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)

    async def _handle_processed_outputs(self, stage_id: int, replica_id: int, outputs: list[Any]) -> None:
        """Route processed stage outputs produced by one stage poll."""
        pool = self.stage_pools[stage_id]
        for output in outputs:
            req_state = self.request_states.get(output.request_id)
            if req_state is None:
                logger.warning(
                    "[Orchestrator] Dropping output for unknown req %s at stage-%s (known reqs: %s)",
                    output.request_id,
                    stage_id,
                    list(self.request_states.keys()),
                )
                continue

            if getattr(output, "error", None) is not None:
                await self._handle_stage_error(stage_id, output)
                continue

            stage_metrics = None
            if output.finished:
                stage_metrics = pool.build_stage_metrics(
                    [output],
                    submit_ts=req_state.stage_submit_ts.get(stage_id, _time.time()),
                    request_timestamp=req_state.request_timestamp,
                    replica_id=replica_id,
                    sampling_params=req_state.sampling_params_list[stage_id],
                )
                stage_metrics.pipeline_timings = dict(req_state.pipeline_timings)

            await self._route_output(stage_id, replica_id, output, req_state, stage_metrics)

    async def _handle_stage_error(self, stage_id: int, output: Any) -> None:
        """Emit a frontend-visible error and clean up request state."""
        if self._cfg_tracker.is_companion(output.request_id):
            parent_id = self._cfg_tracker.get_parent_id(output.request_id) or output.request_id
        else:
            parent_id = output.request_id
        await self.output_async_queue.put(
            ErrorMessage(
                request_id=parent_id,
                stage_id=stage_id,
                error=output.error,
                status_code=getattr(output, "error_status_code", None),
                error_type=getattr(output, "error_type", None),
            )
        )
        await self._cleanup_request_ids(
            [parent_id, *self._cfg_tracker.cleanup_parent(parent_id)],
            abort=True,
            close_duplex_sessions=True,
        )

    # ---- Shared helpers ----

    async def _cleanup_request_ids(
        self,
        request_ids: list[str],
        *,
        abort: bool = False,
        close_duplex_sessions: bool = False,
    ) -> None:
        """Release pool bindings and logical request state for the given ids."""
        if not request_ids:
            return

        cleanup_ids = list(dict.fromkeys(request_ids))
        if close_duplex_sessions:
            closed_sessions = self.duplex_sessions.close_sessions_for_request_ids(cleanup_ids)
            for session_id, stale_request_ids in closed_sessions.items():
                logger.info(
                    "[Orchestrator] closed duplex session %s while cleaning failed request ids %s",
                    session_id,
                    stale_request_ids,
                )
                cleanup_ids.extend(stale_request_ids)
            cleanup_ids = list(dict.fromkeys(cleanup_ids))

        if abort:
            await self._abort_request_ids(cleanup_ids)
        self._release_request_bindings(cleanup_ids)
        for request_id in cleanup_ids:
            self._pd_kv_params.pop(request_id, None)
            if self.request_states.pop(request_id, None) is not None and self._running_counter is not None:
                self._running_counter.decrement()

    async def _apply_raw_terminal_stage_finish(
        self,
        stage_id: int,
        eco: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Record session-level finish markers dropped by the streaming output processor.

        Streaming segment stops set ``is_segment_finished=True`` and are handled
        via processed outputs. Session termination (e.g. ``finish_requests`` after
        ``resumable=False``) emits a terminal ``finish_reason`` with
        ``is_segment_finished=False``, but vLLM's output processor may remove the
        request state before that EngineCoreOutput is processed.

        Only update ``finished_final_output_stage_ids`` here. Request cleanup stays
        in ``_route_output`` so downstream async-chunk stages can still deliver
        outputs after stage-0 session end.
        """
        if getattr(eco, "finish_reason", None) is None:
            return
        if getattr(eco, "is_segment_finished", False):
            return

        final_output_stage_ids = req_state.final_output_stage_ids or {req_state.final_stage_id}
        if stage_id not in final_output_stage_ids:
            return
        req_state.finished_final_output_stage_ids.add(stage_id)

    def _maybe_clone_diffusion_params_for_cfg(self, request_id: str, params: Any) -> Any:
        """Attach CFG companion ids to diffusion sampling params when needed."""
        companion_request_ids = self._cfg_tracker.get_companion_request_ids(request_id)
        if not companion_request_ids:
            return params

        import copy

        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        if not isinstance(params, OmniDiffusionSamplingParams):
            return params

        params = copy.deepcopy(params)
        params.cfg_kv_request_ids = companion_request_ids
        return params

    def _duplex_session_for_req_state(self, req_state: OrchestratorRequestState) -> DuplexSessionRuntimeState | None:
        duplex_state = req_state.streaming.bridge_states.get("duplex")
        if not isinstance(duplex_state, dict):
            return None
        session_id = duplex_state.get("session_id")
        if not isinstance(session_id, str):
            return None
        return self.duplex_sessions.get(session_id)

    def _record_duplex_stage_submission(
        self,
        stage_id: int,
        request_id: str,
        replica_id: int,
        req_state: OrchestratorRequestState,
        *,
        already_submitted: bool,
    ) -> None:
        session = self._duplex_session_for_req_state(req_state)
        if session is None:
            return
        fence = self._duplex_fence_for_req_state(req_state)
        if fence is None:
            raise RuntimeError("duplex stage submission is missing a DuplexFence")
        req_state.duplex_stage_fences[stage_id] = fence
        session.bind_stage_request(stage_id, request_id, replica_id=replica_id, fence=fence)
        if _minicpmo45_profile_logs_enabled():
            logger.info(
                "[Orchestrator] duplex stage submit: session=%s epoch=%s stage=%s replica=%s req=%s update=%s",
                session.session_id,
                session.epoch,
                stage_id,
                replica_id,
                request_id,
                already_submitted,
            )

    async def _route_output(
        self,
        stage_id: int,
        replica_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        stage_metrics: Any,
    ) -> None:
        """Route a processed output: send to frontend and/or forward."""
        req_id = output.request_id
        finished = output.finished
        submit_ts = req_state.stage_submit_ts.get(stage_id)

        # CFG companion: stash output so parent can bundle [parent, *companions]
        # into source_outputs for the bridge (e.g. thinker2imagegen).
        if finished and self._cfg_tracker.is_companion(req_id):
            self._cfg_tracker.set_companion_output(req_id, output)
            await self._handle_cfg_companion_ready(req_id)
            await self._cleanup_request_ids([req_id])
            return

        request_finished = False
        if finished and self.stage_pools[stage_id].final_output:
            req_state.finished_final_output_stage_ids.add(stage_id)
            final_output_stage_ids = req_state.final_output_stage_ids or {req_state.final_stage_id}
            request_finished = final_output_stage_ids.issubset(req_state.finished_final_output_stage_ids)
        # Duplex stage-0 segment boundaries are not client-visible outputs:
        # listen decisions are emitted via _emit_duplex_model_listen_output
        # below and spoken content flows through the talker stage. Forwarding
        # the raw stage-0 output as well injects one cumulative-text,
        # no-audio message per unit that every downstream consumer must
        # filter out again (the official implementation returns exactly one
        # result per audio chunk).
        is_duplex_stage0_segment = (
            stage_id == 0 and self._is_duplex_session_request(req_state) and req_state.streaming.segment_finished
        )
        if self.stage_pools[stage_id].final_output and not is_duplex_stage0_segment:
            await self.output_async_queue.put(
                OutputMessage(
                    request_id=req_id,
                    fence=self._duplex_fence_for_req_state(req_state, stage_id=stage_id),
                    stage_id=stage_id,
                    replica_id=replica_id,
                    engine_outputs=output,
                    metrics=stage_metrics,
                    finished=(
                        request_finished
                        or (self._is_duplex_session_request(req_state) and req_state.streaming.segment_finished)
                    ),
                    stage_submit_ts=submit_ts,
                )
            )
        elif stage_metrics is not None:
            await self.output_async_queue.put(
                StageMetricsMessage(
                    request_id=req_id,
                    stage_id=stage_id,
                    replica_id=replica_id,
                    metrics=stage_metrics,
                    stage_submit_ts=submit_ts,
                )
            )

        if self._pd_pair is not None and finished and stage_id == self._pd_pair[0]:
            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is not None:
                self._pd_kv_params[req_id] = kv_params if isinstance(kv_params, dict) else dict(kv_params)
            req_state.pd_prefill_multimodal_output = getattr(output, "multimodal_output", None)

        if self._is_duplex_session_request(req_state) and _minicpmo45_profile_logs_enabled():
            logger.info(
                "[Orchestrator] duplex route check: stage=%s replica=%s "
                "req=%s finished=%s streaming=%s segment_finished=%s "
                "final_stage=%s async_chunk=%s next_submitted=%s",
                stage_id,
                replica_id,
                req_id,
                finished,
                req_state.streaming.enabled,
                req_state.streaming.segment_finished,
                req_state.final_stage_id,
                self.async_chunk,
                self._next_stage_already_submitted(stage_id, req_state),
            )

        if self._is_duplex_model_listen_segment(stage_id, output, req_state):
            await self._emit_duplex_model_listen_output(
                stage_id,
                req_id,
                output,
                req_state,
                stage_metrics,
                submit_ts,
            )
            return

        if (
            (finished or (req_state.streaming.enabled and req_state.streaming.segment_finished))
            and stage_id < req_state.final_stage_id
            and not self.async_chunk
            and (not self._next_stage_already_submitted(stage_id, req_state) or req_state.streaming.enabled)
        ):
            if (
                finished
                and self._cfg_tracker.has_companions(req_id)
                and not self._cfg_tracker.all_companions_done(req_id)
            ):
                self._cfg_tracker.defer_parent(req_id, output, stage_id)
            else:
                stage_params = req_state.sampling_params_list[stage_id]
                final_only_finished = (
                    req_state.streaming.enabled
                    and finished
                    and getattr(stage_params, "output_kind", None) == RequestOutputKind.FINAL_ONLY
                )
                await self._forward_to_next_stage(
                    req_id,
                    stage_id,
                    output,
                    req_state,
                    src_replica_id=replica_id,
                    is_streaming_session=req_state.streaming.enabled,
                    is_final_update=final_only_finished,
                )
                if (
                    req_state.streaming.enabled
                    and finished
                    and not final_only_finished
                    and not self._is_duplex_session_request(req_state)
                ):
                    # For streaming sessions, send the terminal (resumable=False) update only on a finish
                    await self._forward_to_next_stage(
                        req_id,
                        stage_id,
                        output,
                        req_state,
                        src_replica_id=replica_id,
                        is_streaming_session=True,
                        is_final_update=True,
                    )

        if request_finished and not self._is_duplex_session_request(req_state):
            await self._cleanup_request_ids([req_id, *self._cfg_tracker.cleanup_parent(req_id)])

    def _next_stage_already_submitted(self, stage_id: int, req_state: OrchestratorRequestState) -> bool:
        return (stage_id + 1) in req_state.stage_submit_ts

    def _get_stage_input_processor(self, stage_id: int) -> Any:
        processor = self._stage_input_processors.get(stage_id)
        if processor is None:
            from vllm_omni.engine.stage_init_utils import build_stage0_input_processor

            processor = build_stage0_input_processor(self.stage_pools[stage_id].stage_vllm_config)
            self._stage_input_processors[stage_id] = processor
        return processor

    def _upgrade_processed_stage_request(self, request: Any, raw_prompt: Any) -> Any:
        prompt_embeds = getattr(request, "prompt_embeds", None)
        additional_information = None

        if isinstance(raw_prompt, dict):
            if prompt_embeds is None:
                raw_prompt_embeds = raw_prompt.get("prompt_embeds")
                if isinstance(raw_prompt_embeds, torch.Tensor):
                    prompt_embeds = raw_prompt_embeds
            additional_information = serialize_additional_information(
                raw_prompt.get("additional_information"),
                log_prefix="Orchestrator stage input",
            )

        if prompt_embeds is None and additional_information is None:
            return request

        return OmniEngineCoreRequest.from_request(
            request,
            prompt_embeds=prompt_embeds,
            additional_information=additional_information,
        )

    def _next_stage_input_is_tokens(self, next_input: Any) -> bool:
        return isinstance(next_input, dict) and "prompt_token_ids" in next_input

    def _build_next_stage_request(
        self,
        req_id: str,
        next_stage_id: int,
        next_input: Any,
        params: SamplingParams | PoolingParams,
        *,
        mm_features: list | None = None,
        resumable: bool = False,
    ) -> Any:
        next_pool = self.stage_pools[next_stage_id]
        if self._next_stage_input_is_tokens(next_input):
            request = build_engine_core_request_from_tokens(
                request_id=req_id,
                prompt=next_input,
                params=params,
                model_config=next_pool.stage_vllm_config.model_config,
                mm_features=mm_features,
                resumable=resumable,
            )
            request.external_req_id = request.request_id
            return request

        processor = self._get_stage_input_processor(next_stage_id)
        request = processor.process_inputs(
            request_id=req_id,
            prompt=next_input,
            params=params,
            supported_tasks=("generate",),
            arrival_time=_time.time(),
            resumable=resumable,
        )
        request = self._upgrade_processed_stage_request(request, next_input)
        request.external_req_id = req_id
        return request

    @staticmethod
    def _is_duplex_session_request(req_state: OrchestratorRequestState) -> bool:
        duplex_state = req_state.streaming.bridge_states.get("duplex")
        return isinstance(duplex_state, dict) and isinstance(duplex_state.get("session_id"), str)

    @classmethod
    def _duplex_fence_for_req_state(
        cls,
        req_state: OrchestratorRequestState,
        *,
        stage_id: int | None = None,
    ) -> DuplexFence | None:
        if not cls._is_duplex_session_request(req_state):
            return None
        if stage_id is not None:
            stage_fence = req_state.duplex_stage_fences.get(stage_id)
            if stage_fence is not None:
                return stage_fence
        fence = req_state.streaming.bridge_states["duplex"].get("fence")
        if not isinstance(fence, DuplexFence):
            raise RuntimeError("duplex request state is missing a DuplexFence")
        return fence

    @classmethod
    def _is_duplex_model_listen_segment(
        cls,
        stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
    ) -> bool:
        if stage_id >= req_state.final_stage_id:
            return False
        if not cls._is_duplex_session_request(req_state):
            return False
        if not (req_state.streaming.enabled and req_state.streaming.segment_finished):
            return False

        completion = cls._first_completion(output)
        mm_output = cls._completion_multimodal_output(output, completion)
        listen_id = cls._duplex_special_token_ids(mm_output).get("listen_token_id")
        if listen_id is None:
            return False

        stop_reason = getattr(completion, "stop_reason", None) if completion is not None else None
        if cls._coerce_int(stop_reason) == listen_id:
            return True

        token_ids = cls._duplex_listen_segment_token_ids(output, completion, req_state)
        return bool(token_ids and token_ids[-1] == listen_id)

    async def _emit_duplex_model_listen_output(
        self,
        stage_id: int,
        req_id: str,
        output: Any,
        req_state: OrchestratorRequestState,
        stage_metrics: Any,
        submit_ts: float | None,
    ) -> None:
        self._annotate_duplex_model_listen_output(output)
        completion = self._first_completion(output)
        mm_output = self._completion_multimodal_output(output, completion)
        direct_mm_output = dict(mm_output) if isinstance(mm_output, dict) else {}
        direct_mm_output.setdefault("duplex_direct_response", True)
        direct_mm_output.setdefault("duplex_native_decision", "listen")
        direct_mm_output.setdefault("model_listen", True)
        direct_mm_output.setdefault("listen_source", "model_listen")
        if _minicpmo45_profile_logs_enabled():
            logger.info(
                "[Orchestrator] duplex model listen: stage=%s req=%s segment_finished=%s; bypassing downstream stages",
                stage_id,
                req_id,
                req_state.streaming.segment_finished,
            )
        await self.output_async_queue.put(
            OutputMessage(
                request_id=req_id,
                fence=self._duplex_fence_for_req_state(req_state, stage_id=stage_id),
                stage_id=stage_id,
                engine_outputs=OmniRequestOutput(
                    request_id=req_id,
                    finished=True,
                    stage_id=stage_id,
                    final_output_type="text",
                    request_output=output,
                    _multimodal_output=direct_mm_output,
                ),
                metrics=stage_metrics,
                finished=True,
                stage_submit_ts=submit_ts,
            )
        )

    @classmethod
    def _annotate_duplex_model_listen_output(cls, output: Any) -> None:
        completion = cls._first_completion(output)
        mm_output = cls._completion_multimodal_output(output, completion)
        if not isinstance(mm_output, dict):
            return
        mm_output["duplex_direct_response"] = True
        mm_output["duplex_native_decision"] = "listen"
        mm_output["model_listen"] = True
        mm_output["listen_source"] = "model_listen"

    @staticmethod
    def _first_completion(output: Any) -> Any:
        outputs = getattr(output, "outputs", None)
        return outputs[0] if isinstance(outputs, list) and outputs else None

    @staticmethod
    def _completion_multimodal_output(output: Any, completion: Any) -> dict[str, Any]:
        mm_output = getattr(output, "multimodal_output", None)
        if isinstance(mm_output, dict):
            return mm_output
        mm_output = getattr(completion, "multimodal_output", None) if completion is not None else None
        return mm_output if isinstance(mm_output, dict) else {}

    @classmethod
    def _duplex_special_token_ids(cls, mm_output: dict[str, Any]) -> dict[str, int]:
        out: dict[str, int] = {}
        sources: list[Any] = []
        raw_special = mm_output.get("special_token_ids")
        if isinstance(raw_special, dict):
            sources.append(raw_special)
        raw_meta = mm_output.get("meta")
        if isinstance(raw_meta, dict):
            sources.append(raw_meta)
        sources.append(
            {
                key.removeprefix("meta."): value
                for key, value in mm_output.items()
                if isinstance(key, str) and key.startswith("meta.")
            }
        )
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                if not isinstance(key, str):
                    continue
                token_id = cls._coerce_int(value)
                if token_id is not None and token_id >= 0:
                    out[key] = token_id
        return out

    @classmethod
    def _duplex_listen_segment_token_ids(
        cls,
        output: Any,
        completion: Any,
        req_state: OrchestratorRequestState,
    ) -> list[int]:
        token_ids = cls._completion_token_ids(completion)
        if token_ids:
            return token_ids
        return cls._raw_streaming_segment_token_ids(output, req_state)

    @classmethod
    def _completion_token_ids(cls, completion: Any) -> list[int]:
        if completion is None:
            return []
        for attr in ("token_ids", "cumulative_token_ids"):
            token_ids = cls._coerce_int_list(getattr(completion, attr, None))
            if token_ids:
                return token_ids
        return []

    @classmethod
    def _raw_streaming_segment_token_ids(
        cls,
        output: Any,
        req_state: OrchestratorRequestState,
    ) -> list[int]:
        if not (req_state.streaming.enabled and req_state.streaming.segment_finished):
            return []
        token_ids = cls._coerce_int_list(getattr(output, "new_token_ids", None))
        if token_ids:
            return token_ids
        return []

    @classmethod
    def _coerce_int_list(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            try:
                value = value.detach().cpu().reshape(-1).tolist()
            except Exception:
                return []
        if not isinstance(value, (list, tuple)):
            return []
        out: list[int] = []
        for item in value:
            token_id = cls._coerce_int(item)
            if token_id is not None:
                out.append(token_id)
        return out

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if hasattr(value, "detach"):
            try:
                value = value.detach().cpu().reshape(-1)
                if value.numel() == 0:
                    return None
                value = value[0].item()
            except Exception:
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    async def _handle_cfg_companion_ready(self, req_id: str) -> None:
        """Mark a CFG companion as done; if all companions are done, flush deferred parent."""
        parent_id = self._cfg_tracker.on_companion_completed(req_id)
        if parent_id is None:
            return

        deferred = self._cfg_tracker.pop_pending_parent(parent_id)
        if deferred is None:
            return

        parent_state = self.request_states.get(parent_id)
        if parent_state is None:
            return

        stage_id = deferred["stage_id"]
        if (stage_id + 1) in parent_state.stage_submit_ts:
            return

        await self._forward_to_next_stage(
            parent_id,
            stage_id,
            deferred["engine_outputs"],
            parent_state,
        )

    async def _handle_kv_ready_raw_outputs(
        self,
        stage_id: int,
        raw_outputs: EngineCoreOutputs,
    ) -> None:
        """Forward split requests once stage-0 KV is ready."""
        if self.async_chunk:
            return

        for raw_output in raw_outputs.outputs:
            kv_params = getattr(raw_output, "kv_transfer_params", None)
            if not (isinstance(kv_params, dict) and kv_params.get("kv_ready")):
                continue

            req_id = raw_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                continue
            if self._cfg_tracker.is_companion(req_id):
                await self._handle_cfg_companion_ready(req_id)
                continue
            if stage_id >= req_state.final_stage_id:
                continue
            if (stage_id + 1) in req_state.stage_submit_ts:
                continue

            if self._cfg_tracker.has_companions(req_id) and not self._cfg_tracker.all_companions_done(req_id):
                self._cfg_tracker.defer_parent(req_id, raw_output, stage_id)
            else:
                await self._forward_to_next_stage(req_id, stage_id, raw_output, req_state)

    def _build_pd_decode_params(self, req_id: str, sp: Any) -> Any:
        """Build decode-side sampling params with KV transfer params for PD routing.

        Clones the sampling params and injects kv_transfer_params that tell the
        decode engine where to pull the KV cache from (prefill engine's bootstrap addr).
        """
        sp = sp.clone()
        if sp.extra_args is None:
            sp.extra_args = {}

        # Get KV params captured from the prefill output (must include remote_request_id).
        kv_prefill_params = self._pd_kv_params.pop(req_id, None)
        if not kv_prefill_params or "remote_request_id" not in kv_prefill_params:
            raise RuntimeError(
                f"[Orchestrator][PD] Missing prefill kv_transfer_params.remote_request_id for req={req_id}"
            )

        decode_kv_params: dict[str, Any] = {
            "transfer_id": f"xfer-{req_id}",
        }

        if self._pd_bootstrap_addr:
            decode_kv_params["remote_bootstrap_addr"] = self._pd_bootstrap_addr

        if self._pd_prefill_engine_id:
            decode_kv_params["remote_engine_id"] = self._pd_prefill_engine_id

        # Overlay params from prefill side (includes remote_request_id set by monkey patch).
        decode_kv_params.update(kv_prefill_params)

        # Ensure these flags are set correctly after any overlay.
        decode_kv_params["do_remote_prefill"] = True
        decode_kv_params["do_remote_decode"] = False
        if not decode_kv_params.get("transfer_id"):
            decode_kv_params["transfer_id"] = f"xfer-{req_id}"

        sp.extra_args["kv_transfer_params"] = decode_kv_params

        logger.debug(
            "[Orchestrator][PD] decode kv_transfer_params for req=%s: %s",
            req_id,
            decode_kv_params,
        )
        return sp

    def _emit_tx_edge(
        self,
        *,
        from_stage: int,
        from_replica: int,
        to_stage: int,
        to_pool: StagePool,
        request_id: str,
        tx_ms: float,
    ) -> None:
        """Emit per-edge transfer_tx_s + transfer_size_bytes histograms.

        ``tx_ms`` is the orchestrator-side wall-clock spent in ``next_pool.
        submit_*`` (serialize + queue submit to the receiving worker). Best-
        effort size_bytes left at 0 — orchestrator doesn't have a cheap handle
        on the serialized payload size; a follow-up can plumb that from the
        connector adapter.
        """
        if self._transfer_emitter is None:
            return
        to_replica = to_pool.get_bound_replica_id(request_id)
        if to_replica is None:
            return
        try:
            self._transfer_emitter.observe_size(from_stage, from_replica, to_stage, to_replica, 0)
            self._transfer_emitter.observe_tx_time(from_stage, from_replica, to_stage, to_replica, tx_ms / 1000.0)
        except Exception:
            logger.debug(
                "[Orchestrator] transfer_tx emit failed for edge %d->%d req=%s",
                from_stage,
                to_stage,
                request_id,
                exc_info=True,
            )

    async def _forward_to_next_stage(
        self,
        req_id: str,
        src_stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        *,
        src_replica_id: int | None = None,
        is_streaming_session: bool = False,
        is_final_update: bool = False,
    ) -> None:
        """Forward output from the current logical stage to the next one."""
        next_logical = src_stage_id + 1
        next_pool = self.stage_pools[next_logical]
        next_client = next_pool.stage_client
        params = req_state.sampling_params_list[next_logical]
        source_outputs = [output]
        next_stage_resumable = is_streaming_session and not is_final_update
        already_submitted = self._next_stage_already_submitted(src_stage_id, req_state)
        requires_multimodal_data = getattr(next_client, "requires_multimodal_data", False)
        _t_submit_start = _time.perf_counter()

        if next_pool.stage_type == "diffusion":
            companion_outputs = self._cfg_tracker.pop_companion_outputs(req_id)
            expected = len(self._cfg_tracker.get_companion_request_ids(req_id))
            if expected > len(companion_outputs):
                logger.warning(
                    "[Orchestrator] req=%s: only %d/%d CFG companion outputs arrived; "
                    "downstream CFG conditioning may degrade",
                    req_id,
                    len(companion_outputs),
                    expected,
                )
            diffusion_source_outputs = [output, *companion_outputs]
            if next_client.custom_process_input_func is not None:
                _t_ar2d = _time.perf_counter()
                _fn = next_client.custom_process_input_func
                _extra_kwargs: dict[str, Any] = {}
                # TODO: replace signature probe with explicit kwarg contract.
                try:
                    import inspect as _inspect

                    if "sampling_params" in _inspect.signature(_fn).parameters:
                        _extra_kwargs["sampling_params"] = params
                except (TypeError, ValueError):
                    pass
                diffusion_prompt = _fn(
                    diffusion_source_outputs,
                    req_state.prompt,
                    requires_multimodal_data,
                    **_extra_kwargs,
                )
                _dt_ar2d = (_time.perf_counter() - _t_ar2d) * 1000
                req_state.pipeline_timings["ar2diffusion_ms"] = _dt_ar2d
                logger.info(
                    "[Orchestrator] ar2diffusion req=%s wall_time=%.3fms stage=%d->%d",
                    req_id,
                    _dt_ar2d,
                    src_stage_id,
                    next_logical,
                )
                if diffusion_prompt is None:
                    error_output = OmniRequestOutput.from_error(
                        req_id,
                        f"Stage-{src_stage_id} produced no valid inputs for diffusion stage-{next_logical}",
                    )
                    logger.warning(
                        "[Orchestrator] req=%s stage=%d produced empty diffusion inputs for stage=%d; "
                        "routing terminal error output",
                        req_id,
                        src_stage_id,
                        next_logical,
                    )
                    await self.output_async_queue.put(
                        OutputMessage(
                            request_id=req_id,
                            fence=self._duplex_fence_for_req_state(req_state, stage_id=next_logical),
                            stage_id=next_logical,
                            engine_outputs=error_output,
                            metrics=None,
                            finished=True,
                        )
                    )
                    await self._cleanup_request_ids(
                        [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
                    )
                    return
                if isinstance(diffusion_prompt, list):
                    if not diffusion_prompt:
                        error_output = OmniRequestOutput.from_error(
                            req_id,
                            f"Stage-{src_stage_id} produced no valid inputs for diffusion stage-{next_logical}",
                        )
                        logger.warning(
                            "[Orchestrator] req=%s stage=%d produced empty diffusion inputs for stage=%d; "
                            "routing terminal error output",
                            req_id,
                            src_stage_id,
                            next_logical,
                        )
                        await self.output_async_queue.put(
                            OutputMessage(
                                request_id=req_id,
                                fence=self._duplex_fence_for_req_state(req_state, stage_id=next_logical),
                                stage_id=next_logical,
                                engine_outputs=error_output,
                                metrics=None,
                                finished=True,
                            )
                        )
                        await self._cleanup_request_ids(
                            [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
                        )
                        return
                    if len(diffusion_prompt) == 1:
                        diffusion_prompt = diffusion_prompt[0]
            else:
                diffusion_prompt = req_state.prompt

            if already_submitted:
                replica_id = await next_pool.submit_update(req_id, req_state, diffusion_prompt)
            else:
                replica_id = await next_pool.submit_initial(
                    req_id,
                    req_state,
                    diffusion_prompt,
                    submit_kwargs={
                        "kv_sender_info": self._build_kv_sender_info(
                            list(getattr(next_client, "engine_input_source", None) or [src_stage_id]),
                            request_id=req_id,
                        )
                    },
                    params_override=self._maybe_clone_diffusion_params_for_cfg(req_id, params),
                )
            self._record_duplex_stage_submission(
                next_logical,
                req_id,
                replica_id,
                req_state,
                already_submitted=already_submitted,
            )
            req_state.stage_submit_ts[next_logical] = _time.time()
            _tx_ms = (_time.perf_counter() - _t_submit_start) * 1000.0
            self._emit_tx_edge(
                from_stage=src_stage_id,
                from_replica=src_replica_id if src_replica_id is not None else 0,
                to_stage=next_logical,
                to_pool=next_pool,
                request_id=req_id,
                tx_ms=_tx_ms,
            )
            return

        # PD disaggregation: prefill → decode routing uses original prompt + KV transfer params
        if self._pd_pair is not None and (src_stage_id, next_logical) == self._pd_pair:
            params = self._build_pd_decode_params(req_id, params)

            # Use the original user prompt for the decode stage (not processed embeddings)
            original_prompt = req_state.prompt
            raw_decode_inputs = [original_prompt] if not isinstance(original_prompt, list) else original_prompt

            decode_inputs: list[dict[str, Any]] = []
            for decode_input in raw_decode_inputs:
                if isinstance(decode_input, dict):
                    decode_inputs.append(decode_input)
                    continue
                prompt_token_ids = getattr(decode_input, "prompt_token_ids", None)
                if prompt_token_ids is None:
                    raise TypeError(
                        "[Orchestrator][PD] decode input must be dict or have prompt_token_ids, "
                        f"got {type(decode_input).__name__} for req={req_id}"
                    )
                decode_inputs.append({"prompt_token_ids": list(prompt_token_ids)})

            for decode_input in decode_inputs:
                request = build_engine_core_request_from_tokens(
                    request_id=req_id,
                    prompt=decode_input,
                    params=params,
                    model_config=next_pool.stage_vllm_config.model_config,
                    mm_features=req_state.mm_features,
                    resumable=next_stage_resumable,
                )
                request.external_req_id = request.request_id
                if already_submitted:
                    replica_id = await next_pool.submit_update(req_id, req_state, request)
                else:
                    replica_id = await next_pool.submit_initial(req_id, req_state, request, prompt_text=None)
                self._record_duplex_stage_submission(
                    next_logical,
                    req_id,
                    replica_id,
                    req_state,
                    already_submitted=already_submitted,
                )

            req_state.stage_submit_ts[next_logical] = _time.time()
            _tx_ms = (_time.perf_counter() - _t_submit_start) * 1000.0
            self._emit_tx_edge(
                from_stage=src_stage_id,
                from_replica=src_replica_id if src_replica_id is not None else 0,
                to_stage=next_logical,
                to_pool=next_pool,
                request_id=req_id,
                tx_ms=_tx_ms,
            )
            return

        if req_state.pd_prefill_multimodal_output is not None:
            req_state.streaming.bridge_states.setdefault(
                "pd_prefill_multimodal_output_by_req",
                {},
            )[req_id] = req_state.pd_prefill_multimodal_output

        decoder_key = "minicpmo45_text_decoder"
        decoder_sentinel = object()
        previous_decoder = decoder_sentinel
        installed_decoder = False
        if self._is_duplex_session_request(req_state):
            source_processor = self.stage_pools[src_stage_id].output_processor
            tokenizer = getattr(source_processor, "tokenizer", None)
            decode = getattr(tokenizer, "decode", None)
            if callable(decode):
                previous_decoder = req_state.streaming.bridge_states.get(decoder_key, decoder_sentinel)

                def _decode_minicpmo45_token_delta(token_ids, _decode=decode):
                    ids = [int(token_id) for token_id in token_ids]
                    try:
                        return _decode(ids, skip_special_tokens=True)
                    except TypeError:
                        return _decode(ids)

                req_state.streaming.bridge_states[decoder_key] = _decode_minicpmo45_token_delta
                installed_decoder = True

        try:
            next_inputs = next_client.process_engine_inputs(
                source_outputs,
                req_state.prompt,
                streaming_context=req_state.streaming,
            )
        except Exception:
            logger.exception(
                "[Orchestrator] req=%s process_engine_inputs FAILED for stage-%s",
                req_id,
                next_logical,
            )
            raise
        finally:
            if installed_decoder:
                if previous_decoder is decoder_sentinel:
                    req_state.streaming.bridge_states.pop(decoder_key, None)
                else:
                    req_state.streaming.bridge_states[decoder_key] = previous_decoder

        if not next_inputs:
            if not getattr(output, "finished", False):
                logger.debug(
                    "[Orchestrator] req=%s stage-%s produced no inputs for stage-%s; waiting for more outputs",
                    req_id,
                    src_stage_id,
                    next_logical,
                )
                return

            final_stage_id = req_state.final_stage_id
            final_pool = self.stage_pools[final_stage_id]
            final_output_type = getattr(final_pool.stage_client, "final_output_type", None)
            terminal_output = _build_terminal_empty_output(
                req_id,
                final_output_type=final_output_type,
                audio_sample_rate=_infer_stage_audio_sample_rate(final_pool),
            )
            submit_ts = _time.time()
            req_state.stage_submit_ts[final_stage_id] = submit_ts
            logger.info(
                "[Orchestrator] req=%s stage-%s produced no terminal inputs for stage-%s; "
                "returning empty %s output from final stage-%s",
                req_id,
                src_stage_id,
                next_logical,
                final_output_type or "text",
                final_stage_id,
            )
            await self.output_async_queue.put(
                OutputMessage(
                    request_id=req_id,
                    fence=self._duplex_fence_for_req_state(req_state, stage_id=final_stage_id),
                    stage_id=final_stage_id,
                    replica_id=0,
                    engine_outputs=terminal_output,
                    metrics=None,
                    finished=True,
                    stage_submit_ts=submit_ts,
                )
            )
            await self._cleanup_request_ids([req_id, *self._cfg_tracker.cleanup_parent(req_id)])
            return

        # Build and submit requests for each input
        for next_input in next_inputs:
            # Only AR thinker stages consume encoder mm_features; downstream
            # (talker/code2wav/…) must not see them (avoids encoder-cache misses).
            model_stage = getattr(getattr(next_pool.stage_vllm_config, "model_config", None), "model_stage", None)
            mm_features = req_state.mm_features if model_stage == "thinker" else None
            request = self._build_next_stage_request(
                req_id,
                next_logical,
                next_input,
                params=params,
                mm_features=mm_features,
                resumable=next_stage_resumable,
            )

            if already_submitted:
                replica_id = await next_pool.submit_update(req_id, req_state, request)
            else:
                replica_id = await next_pool.submit_initial(req_id, req_state, request, prompt_text=None)
            self._record_duplex_stage_submission(
                next_logical,
                req_id,
                replica_id,
                req_state,
                already_submitted=already_submitted,
            )

        req_state.stage_submit_ts[next_logical] = _time.time()
        _tx_ms = (_time.perf_counter() - _t_submit_start) * 1000.0
        self._emit_tx_edge(
            from_stage=src_stage_id,
            from_replica=src_replica_id if src_replica_id is not None else 0,
            to_stage=next_logical,
            to_pool=next_pool,
            request_id=req_id,
            tx_ms=_tx_ms,
        )

    async def _prewarm_async_chunk_stages(
        self,
        request_id: str,
        stage0_request: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Pre-submit downstream stages for async-chunk mode."""
        if req_state.final_stage_id <= 0:
            return

        prompt_token_ids = getattr(stage0_request, "prompt_token_ids", None)
        if prompt_token_ids is None:
            logger.warning(
                "[Orchestrator] async_chunk prewarm skipped for req=%s: stage0 prompt_token_ids missing",
                request_id,
            )
            return

        for next_stage_id in range(1, req_state.final_stage_id + 1):
            next_pool = self.stage_pools[next_stage_id]
            params = req_state.sampling_params_list[next_stage_id]

            req_state.stage_submit_ts[next_stage_id] = _time.time()
            _t_submit_start = _time.perf_counter()

            if next_pool.stage_type == "diffusion":
                await next_pool.submit_initial(
                    request_id,
                    req_state,
                    req_state.prompt,
                    submit_kwargs={
                        "kv_sender_info": self._build_kv_sender_info(
                            list(getattr(next_pool.stage_client, "engine_input_source", None) or [next_stage_id - 1]),
                            request_id=request_id,
                        )
                    },
                )
            else:
                import copy

                from vllm_omni.distributed.omni_connectors.adapter import compute_talker_prompt_ids_length

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
                downstream_resumable = bool(getattr(stage0_request, "resumable", req_state.streaming.enabled))
                request = build_engine_core_request_from_tokens(
                    request_id=request_id,
                    prompt=base_input,
                    params=params,
                    model_config=next_pool.stage_vllm_config.model_config,
                    resumable=downstream_resumable,
                )
                request.external_req_id = request.request_id
                await next_pool.submit_initial(
                    request_id,
                    req_state,
                    request,
                    prompt_text=None,
                )

            # async_chunk pre-submit fires per stage edge (N-1 -> N). Source
            # replica is stage 0's bound replica (single-replica thinker in
            # all current configs); fall back to 0 if unknown.
            _tx_ms = (_time.perf_counter() - _t_submit_start) * 1000.0
            src_replica = self.stage_pools[next_stage_id - 1].get_bound_replica_id(request_id)
            self._emit_tx_edge(
                from_stage=next_stage_id - 1,
                from_replica=src_replica if src_replica is not None else 0,
                to_stage=next_stage_id,
                to_pool=next_pool,
                request_id=request_id,
                tx_ms=_tx_ms,
            )

    def _build_kv_sender_info(
        self,
        sender_stage_ids: list[int],
        *,
        request_id: str | None = None,
    ) -> dict[int, dict[str, Any]] | None:
        """Build per-request sender info for diffusion KV-transfer receivers."""
        sender_infos: dict[int, dict[str, Any]] = {}
        for sender_stage_id in dict.fromkeys(sender_stage_ids):
            if sender_stage_id < 0 or sender_stage_id >= len(self.stage_pools):
                continue

            sender_pool = self.stage_pools[sender_stage_id]
            sender_stage = sender_pool.get_bound_client(request_id) if request_id is not None else None
            if sender_stage is None:
                sender_stage = sender_pool.stage_client
            get_sender_info = getattr(sender_stage, "get_kv_sender_info", None)
            if not callable(get_sender_info):
                continue

            sender_info = get_sender_info()
            if not sender_info:
                logger.warning(
                    "[Orchestrator] Stage-%s has no KV sender info available",
                    sender_stage_id,
                )
                continue

            sender_infos[sender_stage_id] = sender_info

        return sender_infos or None

    # ---- Shutdown / lifecycle ----

    async def _drain_pending_requests_on_fatal(self) -> None:
        """Drain the request queue and broadcast fatal errors for any
        pending add_request messages that were never processed.

        Called from the ``run()`` finally block when a fatal error
        (e.g. ``EngineDeadError``) caused the orchestrator to shut down
        before the request handler could process all queued messages.
        Also broadcasts for any already-tracked requests still in
        ``request_states`` that were not yet notified.
        """
        assert self._fatal_error is not None

        notified: set[str] = set()

        # 1) Drain pending messages from the request queue.
        while True:
            try:
                msg = self.request_async_queue.get_nowait()
            except Exception:
                break
            if msg.type == "add_request":
                req_id = msg.request_id
                await self.output_async_queue.put(
                    ErrorMessage(
                        error=self._fatal_error,
                        fatal=True,
                        request_id=req_id,
                        stage_id=self._fatal_error_stage_id,
                    )
                )
                notified.add(req_id)

        # 2) Broadcast for any tracked requests not already notified
        #    (e.g. request was registered but the EngineDeadError handler
        #    missed it because it wasn't submitted to the dead stage yet).
        for req_id in list(self.request_states):
            if req_id not in notified:
                await self.output_async_queue.put(
                    ErrorMessage(
                        error=self._fatal_error,
                        fatal=True,
                        request_id=req_id,
                        stage_id=self._fatal_error_stage_id,
                    )
                )
            self.request_states.pop(req_id, None)

    def _shutdown_stages(self) -> None:
        """Shutdown all stage pools."""
        if self._stages_shutdown:
            return

        self._stages_shutdown = True
        total = sum(pool.live_num_replicas for pool in self.stage_pools)
        logger.info("[Orchestrator] Shutting down all %d client(s)", total)
        for pool in self.stage_pools:
            for replica_id in pool.live_replica_ids():
                pool.shutdown_replica(replica_id)
