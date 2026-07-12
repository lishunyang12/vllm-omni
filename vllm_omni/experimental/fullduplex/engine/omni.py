# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import time
from base64 import b64decode, b64encode
from binascii import Error as BinasciiError
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from ..core.events import (
    AppendToEngine,
    EngineAppendAccepted,
    EngineFailed,
    RebuildStage0Context,
    ReserveResponse,
    ResetStage1,
)
from ..core.identity import DuplexFence
from ..core.ports import EngineEvent
from ..core.state import DuplexFenceError, DuplexFenceMismatchError


class SessionMode(str, Enum):
    TURN = "turn"
    DUPLEX = "duplex"


class DuplexAdapterPattern(str, Enum):
    CHUNK_GROUP_APPEND = "chunk_group_append"
    PER_STEP_TENSOR_INJECT = "per_step_tensor_inject"
    EXPERIMENTAL_WORKER_CONTROL_RPC = "experimental_worker_control_rpc"
    PER_STEP_TENSOR_HANDOFF = "per_step_tensor_handoff"
    SCHEDULER_DATA_PLANE = "scheduler_data_plane"
    RUNNER_LOCAL_PAYLOAD_REF = "runner_local_payload_ref"
    PARALLEL_FRAME_JOINT = "parallel_frame_joint"


class DuplexInputMode(str, Enum):
    APPEND_TOKENS = "append_tokens"
    APPEND_AUDIO_CHUNK = "append_audio_chunk"
    APPEND_STAGE_HANDOFF = "append_stage_handoff"
    APPEND_TTS_HANDOFF = "append_tts_handoff"
    REPLACE_LATEST_CHUNK = "replace_latest_chunk"
    REENCODE_CONTEXT = "reencode_context"
    ROLLBACK_TO_CHECKPOINT = "rollback_to_checkpoint"
    TURN_COMMIT_ONLY = "turn_commit_only"


class DuplexSignalSource(str, Enum):
    MODEL_NATIVE = "model_native"
    EXTERNAL_VAD = "external_vad"
    CLIENT_EVENT = "client_event"
    SERVER_POLICY = "server_policy"
    DIALOGUE_STATE_MODEL = "dialogue_state_model"


@dataclass
class DuplexRuntimeCapabilities:
    adapter_patterns: set[DuplexAdapterPattern] = field(default_factory=set)
    input_modes: set[DuplexInputMode] = field(default_factory=lambda: {DuplexInputMode.TURN_COMMIT_ONLY})
    signal_sources: set[DuplexSignalSource] = field(
        default_factory=lambda: {
            DuplexSignalSource.CLIENT_EVENT,
            DuplexSignalSource.SERVER_POLICY,
        }
    )
    supports_kv_lease: bool = False
    supports_core_kv_lease: bool = False
    supports_model_internal_state: bool = False
    supports_stage_resumption: bool = False
    supports_scheduler_native_append: bool = False
    supports_core_resumable_request: bool = False
    supports_stage_connector_handoff: bool = False
    supports_independent_io_streams: bool = False
    supports_realtime_endpoint: bool = False
    supports_multi_session: bool = False
    supports_multi_session_same_replica: bool = False
    supports_barge_in: bool = True
    supports_playback_ack: bool = True
    supports_audio_truncate: bool = False
    implementation_level: str = "serving_session_adapter"
    stage_handoff_transport: str | None = None
    chunk_period_ms: int | None = None
    target_barge_in_latency_ms: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_patterns": sorted(pattern.value for pattern in self.adapter_patterns),
            "input_modes": sorted(mode.value for mode in self.input_modes),
            "signal_sources": sorted(source.value for source in self.signal_sources),
            "supports_kv_lease": self.supports_kv_lease,
            "supports_core_kv_lease": self.supports_core_kv_lease,
            "supports_model_internal_state": self.supports_model_internal_state,
            "supports_stage_resumption": self.supports_stage_resumption,
            "supports_scheduler_native_append": self.supports_scheduler_native_append,
            "supports_core_resumable_request": self.supports_core_resumable_request,
            "supports_stage_connector_handoff": self.supports_stage_connector_handoff,
            "supports_independent_io_streams": self.supports_independent_io_streams,
            "supports_realtime_endpoint": self.supports_realtime_endpoint,
            "supports_multi_session": self.supports_multi_session,
            "supports_multi_session_same_replica": self.supports_multi_session_same_replica,
            "supports_barge_in": self.supports_barge_in,
            "supports_playback_ack": self.supports_playback_ack,
            "supports_audio_truncate": self.supports_audio_truncate,
            "implementation_level": self.implementation_level,
            "stage_handoff_transport": self.stage_handoff_transport,
            "chunk_period_ms": self.chunk_period_ms,
            "target_barge_in_latency_ms": self.target_barge_in_latency_ms,
        }


@dataclass
class DuplexPlaybackCommitCursor:
    generated_ms: int = 0
    sent_ms: int = 0
    played_ms: int = 0
    committed_ms: int = 0

    def mark_generated(self, generated_ms: int) -> None:
        self.generated_ms = max(self.generated_ms, max(0, int(generated_ms)))

    def mark_sent(self, sent_ms: int) -> None:
        self.sent_ms = max(self.sent_ms, max(0, int(sent_ms)))

    def acknowledge(self, played_ms: int, committed_ms: int | None = None) -> None:
        self.played_ms = max(self.played_ms, max(0, int(played_ms)))
        self.committed_ms = max(
            self.committed_ms,
            self.played_ms if committed_ms is None else max(0, int(committed_ms)),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "generated_ms": self.generated_ms,
            "sent_ms": self.sent_ms,
            "played_ms": self.played_ms,
            "committed_ms": self.committed_ms,
        }


@dataclass
class DuplexStageBinding:
    stage_id: int
    request_id: str
    fence: DuplexFence
    replica_id: int | None = None
    lease_active: bool = False


@dataclass
class DuplexInputAppend:
    seq: int
    turn_seq: int
    fence: DuplexFence
    mode: DuplexInputMode
    payload_meta: dict[str, Any] = field(default_factory=dict)
    final: bool = False
    source: DuplexSignalSource = DuplexSignalSource.CLIENT_EVENT

    @property
    def turn_id(self) -> int:
        return self.fence.turn_id


@dataclass
class DuplexSessionRuntimeState:
    """Engine resource handles associated with a core-owned identity fence."""

    fence: DuplexFence
    session_mode: SessionMode = SessionMode.DUPLEX
    capabilities: DuplexRuntimeCapabilities = field(default_factory=DuplexRuntimeCapabilities)
    session_config: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    stage_bindings: dict[int, DuplexStageBinding] = field(default_factory=dict)
    input_seq: int = 0
    input_turn_seq: int = 0
    pending_inputs: list[DuplexInputAppend] = field(default_factory=list)
    playback: DuplexPlaybackCommitCursor = field(default_factory=DuplexPlaybackCommitCursor)
    closed: bool = False
    _append_turn_key: tuple[int, int, int] | None = None

    @property
    def session_id(self) -> str:
        return self.fence.session_id

    @property
    def epoch(self) -> int:
        return self.fence.epoch

    @property
    def turn_id(self) -> int:
        return self.fence.turn_id

    def touch(self) -> None:
        self.updated_at = time.monotonic()

    def accept_fence(self, fence: DuplexFence) -> None:
        if fence.session_id != self.session_id:
            raise DuplexFenceMismatchError(self.fence, fence)
        current = self.fence
        if fence.epoch < current.epoch or (
            fence.epoch == current.epoch
            and (fence.turn_id < current.turn_id or fence.response_seq < current.response_seq)
        ):
            raise DuplexFenceMismatchError(current, fence)
        if fence.epoch != self.fence.epoch:
            self.input_seq = 0
            self.input_turn_seq = 0
            self.pending_inputs.clear()
            self._append_turn_key = None
        self.fence = fence
        self.touch()

    def bind_stage_request(
        self,
        stage_id: int,
        request_id: str,
        replica_id: int | None = None,
        *,
        fence: DuplexFence,
    ) -> None:
        self.accept_fence(fence)
        self.stage_bindings[stage_id] = DuplexStageBinding(
            stage_id=stage_id,
            request_id=request_id,
            fence=fence,
            replica_id=replica_id,
            lease_active=self.capabilities.supports_core_kv_lease,
        )
        self.touch()

    def stage_request_ids(self, fence: DuplexFence | None = None) -> list[str]:
        return [
            binding.request_id for binding in self.stage_bindings.values() if fence is None or binding.fence == fence
        ]

    def append_input(
        self,
        payload: Any,
        *,
        mode: DuplexInputMode,
        fence: DuplexFence,
        final: bool = False,
        source: DuplexSignalSource = DuplexSignalSource.CLIENT_EVENT,
    ) -> DuplexInputAppend:
        if mode not in self.capabilities.input_modes:
            raise ValueError(f"Duplex input mode {mode.value!r} is not supported by session {self.session_id}")
        self.accept_fence(fence)
        self.input_seq += 1
        turn_key = (fence.epoch, fence.turn_id, fence.response_seq)
        if turn_key != self._append_turn_key:
            self._append_turn_key = turn_key
            self.input_turn_seq = 0
        self.input_turn_seq += 1
        update = DuplexInputAppend(
            seq=self.input_seq,
            turn_seq=self.input_turn_seq,
            fence=fence,
            mode=mode,
            payload_meta=self._payload_metadata(payload),
            final=final,
            source=source,
        )
        self.pending_inputs.append(update)
        self.touch()
        return update

    @staticmethod
    def _payload_metadata(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            meta: dict[str, Any] = {"type": "dict", "keys": sorted(str(key) for key in payload)}
            audio = payload.get("audio") or payload.get("data")
            if isinstance(audio, str):
                try:
                    meta["audio_bytes"] = len(b64decode(audio, validate=True))
                except (BinasciiError, ValueError):
                    meta["audio_chars"] = len(audio)
            if isinstance(payload.get("format"), str):
                meta["format"] = payload["format"]
            if isinstance(payload.get("sample_rate_hz"), int | float):
                meta["sample_rate_hz"] = int(payload["sample_rate_hz"])
            return meta
        if isinstance(payload, str):
            return {"type": "str", "chars": len(payload)}
        if isinstance(payload, bytes | bytearray | memoryview):
            return {"type": type(payload).__name__, "bytes": len(payload)}
        if isinstance(payload, list | tuple):
            return {"type": type(payload).__name__, "items": len(payload)}
        return {"type": type(payload).__name__}

    def acknowledge_playback(self, played_ms: int, committed_ms: int | None = None) -> None:
        self.playback.acknowledge(played_ms, committed_ms)
        self.touch()

    def release_fence(self, fence: DuplexFence) -> list[str]:
        stale = self.stage_request_ids(fence)
        self.stage_bindings = {
            stage_id: binding for stage_id, binding in self.stage_bindings.items() if binding.fence != fence
        }
        self.pending_inputs = [item for item in self.pending_inputs if item.fence != fence]
        self.touch()
        return stale

    def close(self, fence: DuplexFence | None = None) -> list[str]:
        if fence is not None:
            self.accept_fence(fence)
        self.closed = True
        stale = self.stage_request_ids()
        self.stage_bindings.clear()
        self.pending_inputs.clear()
        self.touch()
        return stale

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_mode": self.session_mode.value,
            "fence": self.fence,
            "epoch": self.epoch,
            "input_seq": self.input_seq,
            "turn_id": self.turn_id,
            "input_turn_seq": self.input_turn_seq,
            "closed": self.closed,
            "stage_bindings": {
                stage_id: {
                    "request_id": binding.request_id,
                    "fence": binding.fence,
                    "replica_id": binding.replica_id,
                    "lease_active": binding.lease_active,
                }
                for stage_id, binding in self.stage_bindings.items()
            },
            "playback": self.playback.as_dict(),
            "capabilities": self.capabilities.as_dict(),
        }


class DuplexSessionRuntimeManager:
    def __init__(self) -> None:
        self._sessions: dict[str, DuplexSessionRuntimeState] = {}

    def open_session(
        self,
        fence: DuplexFence,
        *,
        session_mode: SessionMode = SessionMode.DUPLEX,
        capabilities: DuplexRuntimeCapabilities | None = None,
        session_config: dict[str, Any] | None = None,
    ) -> DuplexSessionRuntimeState:
        if not isinstance(fence, DuplexFence):
            raise TypeError("open_session requires DuplexFence; use open_session_legacy for compatibility")
        if fence.session_id in self._sessions:
            raise ValueError(f"Duplex session already exists: {fence.session_id}")
        session = DuplexSessionRuntimeState(
            fence=fence,
            session_mode=session_mode,
            capabilities=capabilities or DuplexRuntimeCapabilities(),
            session_config=dict(session_config or {}),
        )
        self._sessions[fence.session_id] = session
        return session

    def open_session_legacy(self, session_id: str, **kwargs: Any) -> DuplexSessionRuntimeState:
        """Deprecated compatibility path for callers without typed identity."""
        return self.open_session(DuplexFence(session_id), **kwargs)

    def get(self, session_id: str) -> DuplexSessionRuntimeState | None:
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> DuplexSessionRuntimeState:
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"Unknown duplex session: {session_id}")
        return session

    def close_session(self, fence: DuplexFence) -> DuplexSessionRuntimeState | None:
        if not isinstance(fence, DuplexFence):
            raise TypeError("close_session requires DuplexFence; use close_session_legacy for compatibility")
        session = self._sessions.pop(fence.session_id, None)
        if session is not None:
            session.close(fence)
        return session

    def close_session_legacy(self, session_id: str) -> DuplexSessionRuntimeState | None:
        """Deprecated compatibility path for callers without typed identity."""
        session = self.get(session_id)
        return None if session is None else self.close_session(session.fence)

    def close_sessions_for_request_ids(self, request_ids: list[str]) -> dict[str, list[str]]:
        request_id_set = set(request_ids)
        closed: dict[str, list[str]] = {}
        for session_id, session in list(self._sessions.items()):
            stale = session.stage_request_ids()
            if request_id_set.isdisjoint(stale):
                continue
            self._sessions.pop(session_id, None)
            session.close()
            closed[session_id] = stale
        return closed


def duplex_data_plane_request_info(result: dict[str, object]) -> tuple[str | None, int | None]:
    stage_results = result.get("stage_results")
    if not isinstance(stage_results, list):
        return None, None
    for item in stage_results:
        if not isinstance(item, dict):
            continue
        inner = item.get("result")
        if not isinstance(inner, dict) or inner.get("data_plane_append") is not True:
            continue
        request_id = inner.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        response_stage_id = inner.get("response_stage_id")
        return request_id, response_stage_id if isinstance(response_stage_id, int) else None
    return None, None


def duplex_data_plane_request_ids(result: dict[str, object]) -> list[str]:
    """All data-plane request ids from an append result, in stage order.

    A two-stage native duplex append (Stage0 listen/speak -> Stage1 TTS) can
    open more than one data-plane request: Stage1 audio arrives on the session
    stream under its own request id, not Stage0's. The port must track every
    data-plane id so :meth:`OmniDuplexEnginePort.events` surfaces Stage1 audio
    instead of dropping it. ``duplex_data_plane_request_info`` returns only the
    first (Stage0) id and is kept for the scheduler-reserve callers that need
    the primary request.
    """
    stage_results = result.get("stage_results")
    if not isinstance(stage_results, list):
        return []
    request_ids: list[str] = []
    for item in stage_results:
        if not isinstance(item, dict):
            continue
        inner = item.get("result")
        if not isinstance(inner, dict) or inner.get("data_plane_append") is not True:
            continue
        request_id = inner.get("request_id")
        if isinstance(request_id, str) and request_id and request_id not in request_ids:
            request_ids.append(request_id)
    return request_ids


def duplex_resource_request_id(fence: DuplexFence, role: str) -> str:
    if not role or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in role
    ):
        raise ValueError(f"invalid duplex resource role: {role!r}")
    return f"duplex-{fence.session_id}-e{fence.epoch}-{role}"


_DUPLEX_CHUNK_SAMPLES = 16000
_DUPLEX_SAMPLES_PER_AUDIO_TOKEN = 1600


def _duplex_pcm_sample_count(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    audio = payload.get("audio") or payload.get("data")
    if payload.get("format") != "pcm_f32le" or not isinstance(audio, str):
        return None
    try:
        raw = b64decode(audio, validate=True)
    except (BinasciiError, ValueError):
        return None
    return len(raw) // 4


def duplex_payload_is_exact_chunks(payload: object) -> bool:
    sample_count = _duplex_pcm_sample_count(payload)
    return bool(sample_count) and sample_count % _DUPLEX_CHUNK_SAMPLES == 0


def duplex_first_append_unit_count(payload: object) -> int | None:
    sample_count = _duplex_pcm_sample_count(payload)
    if not sample_count or sample_count % _DUPLEX_CHUNK_SAMPLES != 0:
        return None
    return max(1, sample_count // _DUPLEX_CHUNK_SAMPLES - 1)


def duplex_scheduler_token_budget(payload: object, *, default: int = 64) -> int:
    sample_count = _duplex_pcm_sample_count(payload)
    if sample_count is None:
        return max(1, int(default))
    sample_count = max(1, sample_count)
    if sample_count % _DUPLEX_CHUNK_SAMPLES == 0:
        units = sample_count // _DUPLEX_CHUNK_SAMPLES
        return units * (2 + _DUPLEX_CHUNK_SAMPLES // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN)
    return max(16, min(768, sample_count // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8))


def duplex_first_append_context_reserve(session_config: object) -> int:
    if not isinstance(session_config, dict):
        return 48
    sources: list[dict[str, Any]] = [session_config]
    if isinstance(session_config.get("extra_body"), dict):
        sources.append(session_config["extra_body"])
    for source in sources:
        exact = source.get("duplex_first_append_context_tokens")
        if isinstance(exact, int) and exact >= 0:
            return exact
    reserve = 48
    for source in sources:
        ref = source.get("ref_audio_data")
        if not isinstance(ref, str) or not ref:
            continue
        try:
            raw = b64decode(ref, validate=True)
        except (BinasciiError, ValueError):
            continue
        reserve += max(0, (len(raw) // 4) // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8)
        break
    return reserve


def duplex_new_user_turn_prefix_reserve(session_config: object, *, variant: object = None) -> int:
    if not isinstance(session_config, dict):
        return 0
    sources: list[dict[str, Any]] = [session_config]
    if isinstance(session_config.get("extra_body"), dict):
        sources.append(session_config["extra_body"])
    if isinstance(variant, str) and variant:
        for source in sources:
            by_variant = source.get("duplex_new_user_turn_prefix_tokens_by_variant")
            if isinstance(by_variant, dict) and isinstance(by_variant.get(variant), int):
                return max(0, by_variant[variant])
    for source in sources:
        exact = source.get("duplex_new_user_turn_prefix_tokens")
        if isinstance(exact, int) and exact >= 0:
            return exact
    return 0


def _duplex_force_listen_count(extra_body: object) -> int:
    raw = extra_body.get("force_listen_count") if isinstance(extra_body, dict) else None
    try:
        return 0 if raw is None else max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def build_duplex_data_plane_prompt(
    *,
    request_id: str,
    fence: DuplexFence,
    session_config: dict[str, Any],
    seq: int,
    turn_seq: int,
    mode: DuplexInputMode,
    payload: object,
    final: bool,
) -> dict[str, Any]:
    """Plan the token-shaped scheduler admission for the current Omni port."""
    token_budget = duplex_scheduler_token_budget(payload)
    if seq <= 1:
        context_reserve = duplex_first_append_context_reserve(session_config)
        token_budget += context_reserve
        first_units = duplex_first_append_unit_count(payload)
        if first_units is not None:
            token_budget = context_reserve + first_units * 12 - 1
    if (
        seq > 1
        and duplex_payload_is_exact_chunks(payload)
        and not (isinstance(payload, dict) and payload.get("new_user_turn") is True)
    ):
        token_budget += 1
    if isinstance(payload, dict) and payload.get("new_user_turn") is True:
        token_budget += duplex_new_user_turn_prefix_reserve(
            session_config,
            variant=payload.get("new_user_turn_prefix_variant"),
        )
    if final and duplex_payload_is_exact_chunks(payload):
        token_budget += 12
    extra_body = session_config.get("extra_body")
    raw_token_id = session_config.get("duplex_scheduler_token_id")
    if raw_token_id is None and isinstance(extra_body, dict):
        raw_token_id = extra_body.get("duplex_scheduler_token_id")
    try:
        token_id = max(0, int(raw_token_id))
    except (TypeError, ValueError):
        token_id = 0
    force_listen_count = _duplex_force_listen_count(extra_body)
    if (
        force_listen_count > 0
        and turn_seq <= force_listen_count
        and isinstance(payload, dict)
        and payload.get("force_listen") is not True
    ):
        payload = {**payload, "force_listen": True}
    return {
        "prompt_token_ids": [token_id] * token_budget,
        "model_intermediate_buffer": {
            "request_id": request_id,
            "global_request_id": [fence.session_id],
            "duplex": {
                "fence": fence,
                "session_id": fence.session_id,
                "epoch": fence.epoch,
                "seq": seq,
                "turn_id": fence.turn_id,
                "response_seq": fence.response_seq,
                "turn_seq": turn_seq,
                "mode": mode.value,
                "payload": payload,
                "final": final,
                "data_plane": True,
                "session_config": dict(session_config),
                "scheduler_token_budget": token_budget,
                "scheduler_token_id": token_id,
            },
        },
    }


class _AsyncOmniEngine(Protocol):
    async def open_duplex_session_fenced_async(self, fence: DuplexFence, **kwargs: object) -> dict[str, object]: ...

    async def append_duplex_input_fenced_async(self, fence: DuplexFence, **kwargs: object) -> dict[str, object]: ...

    async def signal_duplex_turn_fenced_async(self, fence: DuplexFence, **kwargs: object) -> dict[str, object]: ...

    async def close_duplex_session_fenced_async(self, fence: DuplexFence, **kwargs: object) -> dict[str, object]: ...

    async def get_duplex_output_async(self, session_id: str) -> object: ...


class DuplexOutputFenceError(DuplexFenceError):
    def __init__(self, output: object) -> None:
        self.output = output
        super().__init__(f"{type(output).__name__} is missing a DuplexFence")


OutputMapper = Callable[[object, DuplexFence], EngineEvent | Iterable[EngineEvent]]


class OmniDuplexEnginePort:
    """Typed full-duplex port over the current AsyncOmniEngine controls."""

    def __init__(
        self,
        engine: _AsyncOmniEngine,
        *,
        capabilities: DuplexRuntimeCapabilities | None = None,
        session_config: dict[str, object] | None = None,
        input_mode: DuplexInputMode = DuplexInputMode.APPEND_AUDIO_CHUNK,
        output_mapper: OutputMapper | None = None,
        timeout: float | None = 10.0,
    ) -> None:
        self._engine = engine
        self._capabilities = capabilities or DuplexRuntimeCapabilities(input_modes={input_mode})
        self._session_config = dict(session_config or {})
        self._input_mode = input_mode
        self._output_mapper = output_mapper or self._default_output_mapper
        self._timeout = timeout
        self._sessions = DuplexSessionRuntimeManager()
        self._request_ids: set[str] = set()
        self._pending_accept_fences: set[DuplexFence] = set()
        self._session_id: str | None = None
        self._closed = False

    async def reserve(self, command: ReserveResponse) -> None:
        await self._ensure_open(command.fence)

    async def append(self, command: AppendToEngine) -> None:
        await self._ensure_open(command.fence)
        payload = self._append_payload(command)
        result = await self._engine.append_duplex_input_fenced_async(
            command.fence,
            mode=self._input_mode.value,
            payload=payload,
            final=command.final,
            expected_epoch=command.fence.epoch,
            timeout=self._timeout,
        )
        # Register every data-plane request id, not just Stage0's: Stage1 TTS
        # audio arrives on the session stream under its own id and would be
        # dropped by events() if only the primary (Stage0) id were tracked.
        for request_id in duplex_data_plane_request_ids(result):
            self._request_ids.add(request_id)
        # A committed turn (final append) must surface EngineAppendAccepted so
        # the reducer can advance TURN_COMMITTED -> AWAITING_MODEL before any
        # model output arrives; events() emits it just ahead of that turn's
        # first mapped output.
        if command.final:
            self._pending_accept_fences.add(command.fence)

    async def cancel(self, fence: DuplexFence) -> None:
        await self._signal(fence, "response.cancel", {})

    async def reset(self, command: ResetStage1) -> None:
        await self._signal(command.fence, "reset_stage1", {})

    async def rebuild(self, command: RebuildStage0Context) -> None:
        await self._signal(
            command.fence,
            "rebuild_stage0",
            {
                "committed_history": command.committed_history,
                "committed_playback_position": command.committed_playback_position,
            },
        )

    async def close(self, fence: DuplexFence) -> None:
        await self._engine.close_duplex_session_fenced_async(
            fence,
            reason="runtime_close",
            timeout=self._timeout,
        )
        self._sessions.close_session(fence)
        self._closed = True

    async def events(self) -> AsyncIterator[EngineEvent]:
        while not self._closed:
            if self._session_id is None:
                # The runtime starts consuming events concurrently with input,
                # but the engine session opens lazily on the first append. Wait
                # for it rather than failing the session on this benign race.
                await asyncio.sleep(0.005)
                continue
            output = await self._engine.get_duplex_output_async(self._session_id)
            if output is None:
                await asyncio.sleep(0)
                continue
            request_id = getattr(output, "request_id", None)
            if not isinstance(request_id, str) or request_id not in self._request_ids:
                continue
            fence = getattr(output, "fence", None)
            if not isinstance(fence, DuplexFence):
                raise DuplexOutputFenceError(output)
            if fence.session_id != self._session_id:
                raise DuplexFenceMismatchError(DuplexFence(self._session_id), fence)
            raw = getattr(output, "engine_outputs", output)
            normalized = self._normalize_engine_output(raw, output)
            mapped = self._output_mapper(normalized, fence)
            events = mapped if isinstance(mapped, Iterable) and not hasattr(mapped, "fence") else (mapped,)
            # Surface the turn's acceptance exactly once, before its first model
            # output, so the reducer leaves TURN_COMMITTED for AWAITING_MODEL.
            if fence in self._pending_accept_fences:
                self._pending_accept_fences.discard(fence)
                yield EngineAppendAccepted(fence=fence)
            for event in events:
                event_fence = getattr(event, "fence", None)
                if not isinstance(event_fence, DuplexFence):
                    raise DuplexOutputFenceError(event)
                if event_fence != fence:
                    raise DuplexFenceMismatchError(fence, event_fence)
                yield event

    @staticmethod
    def _extract_mm_output(raw: object) -> dict:
        """Pull the multimodal_output mapping out of an engine output.

        Mirrors the Track B serving extraction chain: the duplex data-plane
        output nests the normalized native result (is_listen / audio_data /
        text-marks) under ``multimodal_output`` — on the output itself, its
        first completion, or the wrapped ``request_output``.
        """
        from collections.abc import Mapping

        if isinstance(raw, Mapping):
            return dict(raw)
        mm = getattr(raw, "multimodal_output", None)
        if isinstance(mm, Mapping) and mm:
            return dict(mm)
        outs = getattr(raw, "outputs", None)
        comp = outs[0] if isinstance(outs, list) and outs else None
        if comp is not None:
            mm = getattr(comp, "multimodal_output", None)
            if isinstance(mm, Mapping) and mm:
                return dict(mm)
        inner = getattr(raw, "request_output", None)
        if inner is not None and inner is not raw:
            mm = getattr(inner, "multimodal_output", None)
            if isinstance(mm, Mapping) and mm:
                return dict(mm)
            inner_outs = getattr(inner, "outputs", None)
            inner_comp = inner_outs[0] if isinstance(inner_outs, list) and inner_outs else None
            if inner_comp is not None:
                mm = getattr(inner_comp, "multimodal_output", None)
                if isinstance(mm, Mapping) and mm:
                    return dict(mm)
        return {}

    def _normalize_engine_output(self, raw: object, output_msg: object) -> object:
        """Normalize an engine output into the flat dict ``map_output`` expects.

        Only real engine request-output wrappers (which carry the native result
        under ``multimodal_output`` / ``outputs`` / ``request_output``) are
        flattened. Already-mapped domain events (identified by a ``fence``
        attribute) and plain mappings are passed through untouched so the
        mapper's own dispatch still applies.
        """
        from collections.abc import Mapping

        if hasattr(raw, "fence") or isinstance(raw, Mapping):
            return raw
        has_payload = (
            getattr(raw, "multimodal_output", None) is not None
            or getattr(raw, "outputs", None) is not None
            or getattr(raw, "request_output", None) is not None
        )
        if not has_payload:
            return raw
        result = self._extract_mm_output(raw)
        if "is_listen" not in result:
            decision = result.get("duplex_native_decision")
            if result.get("model_listen") is True or decision == "listen":
                result["is_listen"] = True
            elif decision == "speak" or result.get("audio_data"):
                result["is_listen"] = False
        # Surface the first completion's text / token ids / stop reason: the
        # model event adapter needs them to classify listen vs speak (the native
        # decision keys off the trailing token id and stop reason) and to dedup
        # the per-segment transcript.
        completion = None
        for candidate in (raw, getattr(raw, "request_output", None)):
            outs = getattr(candidate, "outputs", None)
            comp = outs[0] if isinstance(outs, list) and outs else None
            if comp is not None:
                completion = comp
                break
        if completion is not None:
            if not result.get("text"):
                text = getattr(completion, "text", "")
                if isinstance(text, str) and text:
                    result["text"] = text
            if "token_ids" not in result:
                token_ids = getattr(completion, "token_ids", None)
                if token_ids is not None:
                    result["token_ids"] = list(token_ids)
            if "stop_reason" not in result:
                result["stop_reason"] = getattr(completion, "stop_reason", None)
        if "finished" not in result:
            result["finished"] = bool(getattr(output_msg, "finished", False))
        return result

    async def _ensure_open(self, fence: DuplexFence) -> None:
        if self._session_id is not None and self._session_id != fence.session_id:
            raise ValueError("OmniDuplexEnginePort supports one runtime session")
        session = self._sessions.get(fence.session_id)
        if session is not None:
            session.accept_fence(fence)
            return
        await self._engine.open_duplex_session_fenced_async(
            fence,
            session_mode=SessionMode.DUPLEX.value,
            capabilities=self._capabilities.as_dict(),
            session_config=self._session_config,
            timeout=self._timeout,
        )
        self._sessions.open_session(
            fence,
            capabilities=self._capabilities,
            session_config=self._session_config,
        )
        self._session_id = fence.session_id

    async def _signal(self, fence: DuplexFence, event: str, payload: dict[str, object]) -> None:
        await self._engine.signal_duplex_turn_fenced_async(
            fence,
            event=event,
            payload=payload,
            timeout=self._timeout,
        )

    def _append_payload(self, command: AppendToEngine) -> object:
        if command.chunk is None:
            return {"final": command.final}
        data = command.chunk.data
        if command.chunk.modality == "audio" and isinstance(data, bytes):
            # Mirror the Track B PCM-buffer payload exactly: type/format/rate +
            # force_listen + is_speech. Missing "type"/"force_listen" desyncs the
            # model's audio handling and yields NaN logits. new_user_turn is NOT
            # set (the opening utterance is not a new user turn).
            return {
                "type": "audio",
                "audio": b64encode(data).decode("ascii"),
                "format": "pcm_f32le",
                "sample_rate_hz": int(self._session_config.get("sample_rate_hz", 16000)),
                "force_listen": False,
                "is_speech": True,
            }
        return {"data": data, "modality": command.chunk.modality}

    @staticmethod
    def _default_output_mapper(output: object, fence: DuplexFence) -> EngineEvent:
        if hasattr(output, "fence"):
            return output  # type: ignore[return-value]
        error = getattr(output, "error", None)
        if isinstance(error, str) and error:
            return EngineFailed(error, fence=fence)
        raise TypeError("Omni duplex model output requires a model-specific output_mapper")


__all__ = [
    "DuplexAdapterPattern",
    "DuplexInputAppend",
    "DuplexInputMode",
    "DuplexOutputFenceError",
    "DuplexPlaybackCommitCursor",
    "DuplexRuntimeCapabilities",
    "DuplexSessionRuntimeManager",
    "DuplexSessionRuntimeState",
    "DuplexSignalSource",
    "DuplexStageBinding",
    "OmniDuplexEnginePort",
    "SessionMode",
    "build_duplex_data_plane_prompt",
    "duplex_data_plane_request_info",
    "duplex_data_plane_request_ids",
    "duplex_first_append_context_reserve",
    "duplex_first_append_unit_count",
    "duplex_new_user_turn_prefix_reserve",
    "duplex_payload_is_exact_chunks",
    "duplex_resource_request_id",
    "duplex_scheduler_token_budget",
]
