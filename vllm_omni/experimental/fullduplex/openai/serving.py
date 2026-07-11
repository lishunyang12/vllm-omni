from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import os
import time
from collections.abc import AsyncGenerator, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.engine.omni import duplex_data_plane_request_info
from vllm_omni.experimental.fullduplex.minicpmo45 import (
    MiniCPMO45NativeDuplexServingAdapter,
    MiniCPMO45PcmAppendBuffer,
)
from vllm_omni.experimental.fullduplex.minicpmo45.policy import (
    MiniCPMO45DuplexPolicy,
)
from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexCapabilities,
    DuplexCommittedInput,
    DuplexOverlapPolicy,
    DuplexPlaybackCommitPolicy,
    DuplexSession,
    DuplexSessionConfig,
    DuplexSessionRegistry,
    DuplexSessionState,
    DuplexTurnController,
    DuplexTurnEventType,
    DuplexTurnState,
)
from vllm_omni.experimental.fullduplex.openai.realtime_session import (
    REALTIME_OUTPUT_AUDIO_FORMATS,
    NativeRealtimeSessionProtocol,
)
from vllm_omni.experimental.fullduplex.openai.websocket import (
    DuplexWebSocketActor,
    is_control_event,
    is_input_event,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

logger = init_logger(__name__)

_DEFAULT_CONFIG_TIMEOUT_S = 10.0
_DEFAULT_IDLE_TIMEOUT_S = 300.0
_MAX_EVENT_BYTES = 15 * 1024 * 1024


def should_enable_duplex_endpoint(
    stage_configs: list | None,
    *,
    config_path: str | None = None,
) -> bool:
    """Enable duplex routes only for deployments that explicitly opt in."""
    if stage_configs:
        for stage in stage_configs:
            session_mode = (
                stage.get("session_mode") if isinstance(stage, dict) else getattr(stage, "session_mode", None)
            )
            if session_mode == "duplex":
                return True
    if config_path:
        try:
            from omegaconf import OmegaConf

            raw_config = OmegaConf.load(config_path)
            session_mode = raw_config.get("session_mode") if hasattr(raw_config, "get") else None
            if session_mode == "duplex":
                return True
        except Exception as exc:
            logger.warning("Failed to inspect duplex session_mode from %s: %s", config_path, exc)
    return False


@dataclass
class DuplexAppendTaskMeta:
    epoch: int
    mode: str
    final: bool
    response_bound: bool


@dataclass
class LegacyDuplexSessionActor:
    """Session-scoped actor state for duplex websocket execution.

    This owns the queues and background tasks that make input, output, and
    control independently cancellable at the serving layer. Core KV lease is
    intentionally not modeled here; scheduler/KV ownership remains in the
    engine runner.
    """

    websocket: WebSocket
    output_queue: asyncio.Queue[dict[str, object] | None] = field(default_factory=asyncio.Queue)
    input_queue: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)
    control_queue: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)
    event_queue: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)
    session: DuplexSession | None = None
    outbound_protocol: NativeRealtimeSessionProtocol | None = None
    native_append_tasks: dict[asyncio.Task[None], DuplexAppendTaskMeta] = field(default_factory=dict)
    active_response_task: asyncio.Task[None] | None = None
    runtime_opened: bool = False
    runtime_closed: bool = False
    closing: bool = False
    lifecycle_state: str = "opening"
    close_reason: str | None = None
    stale_output_dropped: int = 0
    control_events_seen: int = 0
    input_events_seen: int = 0
    cancel_count: int = 0
    overlap_speech_ms: int = 0
    last_response_id: str | None = None
    _next_event_seq: int = 0
    _deferred_events: list[dict[str, object]] = field(default_factory=list)

    def transition(self, state: str, *, reason: str | None = None) -> None:
        self.lifecycle_state = state
        if reason is not None:
            self.close_reason = reason
        if self.session is None:
            return
        if state in {"closing", "closed"}:
            self.session.mark_closing()
        elif state == "listening":
            self.session.turn_state = DuplexTurnState.USER_SPEAKING
        elif state == "generating":
            self.session.turn_state = DuplexTurnState.ASSISTANT_GENERATING

    async def enqueue_event(self, event: dict[str, object]) -> None:
        event["_duplex_actor_seq"] = self._next_event_seq
        self._next_event_seq += 1
        event_type = event.get("type")
        if event_type in {"__timeout__", "__disconnect__"}:
            self.control_events_seen += 1
            await self.control_queue.put(event)
        elif isinstance(event_type, str) and OmniDuplexSessionHandler._is_duplex_control_event(event_type):
            self.control_events_seen += 1
            await self.control_queue.put(event)
        elif isinstance(event_type, str) and OmniDuplexSessionHandler._is_duplex_input_event(event_type):
            self.input_events_seen += 1
            if self.output_generation_in_flight():
                event["_duplex_overlap_candidate"] = True
            await self.input_queue.put(event)
        else:
            await self.event_queue.put(event)

    def output_generation_in_flight(self) -> bool:
        if self.session is None:
            return False
        if self.assistant_playback_active():
            return True
        if self.lifecycle_state == "generating":
            return True
        if self.session.active_response_id is not None or self.session.active_request_id is not None:
            return True
        if self.active_response_task is not None and not self.active_response_task.done():
            return True
        return self.has_response_bound_append_tasks()

    def assistant_playback_active(self) -> bool:
        if self.session is None:
            return False
        if self.session.config.playback_commit_policy != DuplexPlaybackCommitPolicy.ACK_ONLY.value:
            return False
        return self.session.playback.sent_ms > self.session.playback.committed_ms

    async def next_event(self) -> dict[str, object]:
        while True:
            ready: list[tuple[int, int, dict[str, object]]] = [
                (self._actor_event_priority(event), self._actor_event_seq(event), event)
                for event in self._deferred_events
            ]
            self._deferred_events.clear()
            for queue in (self.control_queue, self.event_queue, self.input_queue):
                try:
                    event = queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
                ready.append((self._actor_event_priority(event), self._actor_event_seq(event), event))
            if ready:
                ready.sort(key=lambda item: (item[0], item[1]))
                selected = ready[0][2]
                self._deferred_events.extend(event for _, _, event in ready[1:])
                selected.pop("_duplex_actor_seq", None)
                return selected

            control_task = asyncio.create_task(self.control_queue.get())
            event_task = asyncio.create_task(self.event_queue.get())
            input_task = asyncio.create_task(self.input_queue.get())
            tasks = {control_task, event_task, input_task}
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*pending)

            ready = []
            for task in (control_task, event_task, input_task):
                if task in done:
                    event = task.result()
                    ready.append((self._actor_event_priority(event), self._actor_event_seq(event), event))
            for queue in (self.control_queue, self.event_queue, self.input_queue):
                while True:
                    try:
                        event = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    ready.append((self._actor_event_priority(event), self._actor_event_seq(event), event))
            if not ready:
                continue
            ready.sort(key=lambda item: (item[0], item[1]))
            selected = ready[0][2]
            self._deferred_events.extend(event for _, _, event in ready[1:])
            selected.pop("_duplex_actor_seq", None)
            return selected

    @staticmethod
    def _actor_event_seq(event: dict[str, object]) -> int:
        value = event.get("_duplex_actor_seq")
        return int(value) if isinstance(value, int) else 0

    @staticmethod
    def _actor_event_priority(event: dict[str, object]) -> int:
        event_type = event.get("type")
        if event_type in {"__timeout__", "__disconnect__"}:
            return 0
        if event_type in {
            "response.cancel",
            "barge_in",
            "input_audio_buffer.clear",
            "output_audio_buffer.clear",
        }:
            return 0
        if event_type == "input.cancel":
            return 2
        if event_type in {"session.close", "close_session"}:
            return 3
        if isinstance(event_type, str) and OmniDuplexSessionHandler._is_duplex_input_event(event_type):
            return 2
        if isinstance(event_type, str) and OmniDuplexSessionHandler._is_duplex_control_event(event_type):
            return 1
        return 1

    async def send_json(self, payload: dict[str, object]) -> None:
        await self.output_queue.put(payload)

    async def writer_loop(self) -> None:
        while True:
            payload = await self.output_queue.get()
            try:
                if payload is None:
                    return
                raw_realtime = payload.pop("_realtime_raw", False) is True
                if not raw_realtime and self._is_stale_model_output(payload):
                    self.stale_output_dropped += 1
                    continue
                try:
                    if raw_realtime:
                        await self.websocket.send_json(payload)
                    elif self.outbound_protocol is not None:
                        for realtime_payload in self.outbound_protocol.encode_outbound_event(payload):
                            await self.websocket.send_json(realtime_payload)
                    else:
                        await self.websocket.send_json(payload)
                except (WebSocketDisconnect, RuntimeError):
                    return
            finally:
                self.output_queue.task_done()

    def _is_stale_model_output(self, payload: dict[str, object]) -> bool:
        if self.session is None:
            return False
        event_type = payload.get("type")
        if event_type not in {
            "response.created",
            "response.listen",
            "response.speak",
            "response.output_item.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_audio.delta",
            "response.audio.delta",
            "response.output_audio.done",
            "response.audio.done",
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.text.delta",
            "response.text.done",
            "response.message",
            "response.output_item.done",
            "response.content_part.done",
            "response.done",
            "runtime.control",
        }:
            return False
        if (self.closing or self.session.state == DuplexSessionState.CLOSED) and event_type not in {
            "response.listen",
            "runtime.control",
        }:
            return True
        epoch = payload.get("epoch")
        return isinstance(epoch, int) and epoch != self.session.epoch

    def track_append_task(
        self,
        task: asyncio.Task[None],
        *,
        epoch: int,
        mode: str,
        final: bool,
        response_bound: bool,
    ) -> None:
        self.native_append_tasks[task] = DuplexAppendTaskMeta(
            epoch=epoch,
            mode=mode,
            final=final,
            response_bound=response_bound,
        )
        task.add_done_callback(self.native_append_tasks.pop)

    def has_response_bound_append_tasks(self) -> bool:
        return any(meta.response_bound for meta in self.native_append_tasks.values())

    async def cancel_append_tasks(
        self,
        timeout_s: float = 0.25,
        *,
        response_bound_only: bool = False,
    ) -> bool:
        if not self.native_append_tasks:
            return False
        tasks = [
            task for task, meta in self.native_append_tasks.items() if not response_bound_only or meta.response_bound
        ]
        if not tasks:
            return False
        for task in tasks:
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_s)
        except asyncio.TimeoutError:
            # Keep the websocket control path responsive. The task callbacks
            # still remove completed tasks from the tracking set.
            pass
        return True

    def drain_input_queue(self) -> int:
        drained = 0
        kept: list[dict[str, object]] = []
        for event in self._deferred_events:
            event_type = event.get("type")
            if isinstance(event_type, str) and OmniDuplexSessionHandler._is_duplex_input_event(event_type):
                drained += 1
            else:
                kept.append(event)
        self._deferred_events = kept
        while True:
            try:
                self.input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            drained += 1
        return drained


DuplexSessionActor = DuplexWebSocketActor


class OmniDuplexSessionHandler:
    """WebSocket handler for RFC-style full-duplex session control.

    This owns the serving-side session actor, prioritized control/input queues,
    barge-in epoch, turn controller, and playback commit state. Generic sessions
    can still fall back to chat requests, while MiniCPM-o 4.5 native sessions
    route audio appends through scheduler data-plane stage requests. It
    deliberately does not claim core persistent KV lease support.
    """

    def __init__(
        self,
        *,
        chat_service: OmniOpenAIServingChat,
        config_timeout_s: float = _DEFAULT_CONFIG_TIMEOUT_S,
        idle_timeout_s: float = _DEFAULT_IDLE_TIMEOUT_S,
    ) -> None:
        self._chat_service = chat_service
        self._config_timeout_s = config_timeout_s
        self._idle_timeout_s = idle_timeout_s
        self._registry = DuplexSessionRegistry(
            DuplexCapabilities(
                supports_model_native_turn_policy=False,
                supports_input_append=False,
                supports_replace_latest_chunk=True,
                supports_reencode_context=True,
                supports_turn_commit_only=True,
                supports_kv_lease=False,
            )
        )
        self._turn_controller = DuplexTurnController()
        self._data_plane_audio_offsets: dict[str, int] = {}
        # request_id -> cumulative text cursor already exposed with audio.
        # Auto-response streams keep it across segment boundaries.
        self._data_plane_sent_segment_texts: dict[str, str] = {}
        self._data_plane_segment_text_metadata_request_ids: set[str] = set()
        self._data_plane_pending_audio_without_text: dict[str, list[dict[str, object]]] = {}
        self._data_plane_turns_with_text: set[str] = set()
        self._auto_response_waiting_for_speech: set[str] = set()
        self._auto_response_new_turn_prefix_variants: dict[str, str] = {}
        # TTS segment completion and model-turn completion are independent.
        # A turn can finish in a later metadata-only output after its final
        # audio segment has already closed.
        self._data_plane_tts_eos_done: set[str] = set()
        self._data_plane_turn_eos_done: set[str] = set()
        self._data_plane_terminal_request_ids: set[str] = set()
        self._native_data_plane_tasks: dict[str, asyncio.Task[None]] = {}
        # session_id -> (response_id, silence continuation units already sent)
        self._native_response_continuations: dict[str, tuple[str, int]] = {}
        self._response_conversation_modes: dict[str, str] = {}

    async def handle_session(
        self,
        websocket: WebSocket,
        *,
        realtime_protocol: NativeRealtimeSessionProtocol | None = None,
    ) -> None:
        await websocket.accept()
        actor = DuplexSessionActor(websocket, outbound_protocol=realtime_protocol)
        if realtime_protocol is not None:

            async def send_realtime_raw(payload: dict[str, object]) -> None:
                raw_payload = dict(payload)
                raw_payload["_realtime_raw"] = True
                await actor.send_json(raw_payload)

            realtime_protocol.bind_sender(send_realtime_raw)
        session: DuplexSession | None = None
        native_audio_buffer = MiniCPMO45PcmAppendBuffer()
        native_response_emitted = False
        native_input_since_commit = False
        native_speech_since_commit = False
        native_committed_audio_payload: dict[str, object] | None = None
        native_deferred_response_create = False

        async def send_json(payload: dict[str, object]) -> None:
            nonlocal native_input_since_commit, native_response_emitted, native_speech_since_commit
            nonlocal native_committed_audio_payload
            nonlocal native_deferred_response_create
            payload_type = payload.get("type")
            deferred_overlap_payload: dict[str, object] | None = None
            if not actor._is_stale_model_output(payload):
                if payload_type == "response.created":
                    response_id = payload.get("response_id")
                    if isinstance(response_id, str) and response_id:
                        actor.last_response_id = response_id
                    actor.transition("generating")
                elif payload_type == "response.listen":
                    actor.transition("open")
                elif payload_type in {"response.done", "audio.cancelled", "input.cancelled"}:
                    actor.transition("open")
                elif payload_type == "session.closed":
                    actor.transition("closed", reason=actor.close_reason or str(payload.get("reason") or "closed"))
                if payload_type in {
                    "response.done",
                    "response.listen",
                    "audio.cancelled",
                    "input.cancelled",
                    "session.closed",
                }:
                    terminal_status = payload.get("status")
                    terminal_status_details = payload.get("status_details")
                    if terminal_status is None and isinstance(terminal_status_details, dict):
                        terminal_status = terminal_status_details.get("type")
                    can_promote_overlap = payload_type in {
                        "response.done",
                        "response.listen",
                    } and terminal_status not in {"cancelled", "failed"}
                    if can_promote_overlap and actor.overlap_speech_ms > 0:
                        should_promote_overlap = (
                            payload_type in {"response.done", "response.listen"}
                            and session is not None
                            and session.state == DuplexSessionState.OPEN
                            and self._uses_native_input_append(session)
                            and native_audio_buffer.has_pending()
                            and actor.overlap_speech_ms > session.config.overlap_short_ack_ms
                        )
                        if should_promote_overlap and session is not None:
                            deferred_overlap_payload = native_audio_buffer.flush(
                                chunk_period_ms=session.capabilities.chunk_period_ms or 1000
                            )
                            native_input_since_commit = deferred_overlap_payload is not None
                            native_response_emitted = False
                            if realtime_protocol is not None:
                                native_committed_audio_payload = deferred_overlap_payload
                                native_deferred_response_create = True
                                deferred_overlap_payload = None
                        else:
                            had_pending_overlap_audio = native_audio_buffer.has_pending()
                            native_audio_buffer.clear()
                            native_input_since_commit = False
                            native_speech_since_commit = False
                            if had_pending_overlap_audio and realtime_protocol is not None:
                                await realtime_protocol.discard_pending_input_audio(
                                    audio_end_ms=actor.overlap_speech_ms
                                )
                            if payload_type in {"audio.cancelled", "input.cancelled", "session.closed"}:
                                native_committed_audio_payload = None
                                native_deferred_response_create = False
                    actor.overlap_speech_ms = 0
                    if (
                        can_promote_overlap
                        and native_deferred_response_create
                        and native_committed_audio_payload is not None
                    ):
                        deferred_overlap_payload = native_committed_audio_payload
                        native_committed_audio_payload = None
                        native_deferred_response_create = False
                        native_input_since_commit = False
                        native_speech_since_commit = False
            await actor.send_json(payload)
            if deferred_overlap_payload is not None and session is not None and not actor.closing:
                await start_native_append(deferred_overlap_payload, final=True, precreate_response=True)

        writer_task = asyncio.create_task(actor.writer_loop(), name="duplex-session-writer")
        reader_task: asyncio.Task[None] | None = None

        async def read_event_loop() -> None:
            assert session is not None
            try:
                while not actor.closing:
                    raw = await self._receive_text(
                        websocket,
                        session.config.idle_timeout_s,
                        realtime_protocol=realtime_protocol,
                    )
                    if raw is None:
                        await actor.enqueue_event({"type": "__timeout__"})
                        return
                    if len(raw.encode("utf-8")) > _MAX_EVENT_BYTES:
                        await send_json({"type": "error", "error": "Duplex event too large", "code": "event_too_large"})
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        await send_json({"type": "error", "error": "Invalid JSON event", "code": "invalid_json"})
                        continue
                    if not isinstance(event, dict):
                        await send_json(
                            {
                                "type": "error",
                                "error": "Duplex event must be a JSON object",
                                "code": "bad_event",
                            }
                        )
                        continue
                    event_type = event.get("type")
                    if not isinstance(event_type, str):
                        await send_json(
                            {"type": "error", "error": "Duplex event missing string type", "code": "bad_event"}
                        )
                        continue
                    await actor.enqueue_event(event)
            except WebSocketDisconnect:
                await actor.enqueue_event({"type": "__disconnect__"})

        async def next_actor_event() -> dict[str, object]:
            return await actor.next_event()

        async def flush_native_audio_buffer(*, send_json) -> tuple[bool, bool]:
            if session is None:
                return True, False
            # Drain in whole-chunk payloads: the buffer's first emission is
            # capped at one chunk (worker first-unit window), so a long first
            # commit may need several appends to reach the model.
            result: tuple[bool, bool] | None = None
            while True:
                flushed = native_audio_buffer.flush(chunk_period_ms=session.capabilities.chunk_period_ms or 1000)
                if flushed is None:
                    break
                append_epoch = session.epoch
                result = await self._append_runtime_input(
                    session,
                    flushed,
                    final=not native_audio_buffer.has_pending(),
                    send_json=send_json,
                    mode="append_audio_chunk",
                    expected_epoch=append_epoch,
                )
                if result[0] is False:
                    return result
            return result if result is not None else (True, False)

        def native_response_in_progress() -> bool:
            nonlocal native_response_emitted
            if session is None:
                return False
            if native_response_emitted and session.active_response_id is not None:
                return True
            if native_response_emitted and session.active_response_id is None:
                native_response_emitted = False
            if session.active_request_id is not None:
                return True
            if actor.active_response_task is not None and not actor.active_response_task.done():
                return True
            if actor.has_response_bound_append_tasks():
                return True
            if session.session_id in self._native_data_plane_tasks:
                return True
            return actor.lifecycle_state == "generating"

        async def start_native_append(payload: object, *, final: bool, precreate_response: bool = False) -> None:
            if session is None:
                return
            append_epoch = session.epoch
            append_turn_id = self._payload_turn_id(payload)
            if append_turn_id is None:
                append_turn_id = session.turn_id
            request_id = self._native_stage0_request_id(session, append_epoch)
            if final or precreate_response:
                session.active_request_id = request_id
            if precreate_response:
                session.active_response_turn_id = append_turn_id
            if final:
                actor.transition("generating")
            if precreate_response and session.active_response_id is None:
                response_id = session.begin_response(turn_id=append_turn_id)
                self._remember_response_conversation_mode(session, response_id)
                await send_json(
                    self._response_created_payload(
                        session,
                        response_id,
                        epoch=append_epoch,
                    )
                )

            async def _run() -> None:
                nonlocal native_response_emitted
                try:
                    append_ok, emitted_response = await self._append_runtime_input(
                        session,
                        payload,
                        final=final,
                        send_json=send_json,
                        mode="append_audio_chunk",
                        expected_epoch=append_epoch,
                    )
                    if not append_ok and session.state == DuplexSessionState.CLOSED:
                        actor.runtime_closed = True
                        return
                    if emitted_response and self._uses_native_input_append(session):
                        native_response_emitted = True
                        native_audio_buffer.clear()
                    elif not emitted_response and session.epoch == append_epoch:
                        if session.active_request_id == self._native_stage0_request_id(session, append_epoch):
                            session.active_request_id = None
                        if final:
                            actor.transition("open")
                        await send_json(self._turn_controller.signal(session, DuplexTurnEventType.USER_STARTED.value))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("Native duplex append task failed: %s", exc)
                    await self._send_runtime_error(send_json, "runtime_append_task_failed", exc, session=session)
                    if session.state != DuplexSessionState.CLOSED:
                        actor.closing = True
                        actor.transition("closing", reason="runtime_append_task_failed")
                        if await self._close_runtime_session(
                            session,
                            reason="runtime_append_task_failed",
                            send_json=send_json,
                        ):
                            actor.runtime_closed = True
                            session.close()
                            actor.transition("closed", reason="runtime_append_task_failed")
                            await send_json(
                                {
                                    "type": "session.closed",
                                    "session_id": session.session_id,
                                    "reason": "runtime_append_task_failed",
                                }
                            )

            await _run()

        async def start_runtime_append(
            payload: object,
            *,
            final: bool,
            mode: str = "append_tokens",
        ) -> None:
            """Schedule non-native runtime appends without blocking WS input.

            Native MiniCPM-o appends use ``start_native_append`` because they can
            create data-plane response streams. Generic runtime appends still
            need the same control-plane isolation so cancel/close can preempt a
            slow engine append.
            """
            if session is None:
                return
            append_epoch = session.epoch

            async def _run() -> None:
                try:
                    append_ok, emitted_response = await self._append_runtime_input(
                        session,
                        payload,
                        final=final,
                        send_json=send_json,
                        mode=mode,
                        expected_epoch=append_epoch,
                    )
                    if not append_ok:
                        if session.state == DuplexSessionState.CLOSED:
                            actor.runtime_closed = True
                        return
                    if not emitted_response and session.epoch == append_epoch:
                        await send_json(self._turn_controller.signal(session, DuplexTurnEventType.USER_STARTED.value))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("Duplex runtime append task failed: %s", exc)
                    await self._send_runtime_error(send_json, "runtime_append_task_failed", exc, session=session)

            task = asyncio.create_task(_run())
            actor.track_append_task(
                task,
                epoch=append_epoch,
                mode=mode,
                final=final,
                response_bound=final,
            )

        def signal_runtime_background(event_name: str, payload: dict[str, object]) -> None:
            if session is None:
                return
            captured_fence = DuplexFence(
                session.session_id,
                epoch=session.epoch,
                turn_id=session.turn_id,
            )
            asyncio.create_task(
                self._signal_runtime_session(
                    session,
                    event_name,
                    payload,
                    send_json,
                    fence=captured_fence,
                )
            )

        try:
            session = await self._open_session(
                websocket,
                send_json,
                realtime_protocol=realtime_protocol,
            )
            if session is None:
                return
            if realtime_protocol is not None:
                session.config.playback_commit_policy = DuplexPlaybackCommitPolicy.ACK_ONLY.value
            actor.session = session
            actor.transition("opening")
            open_result = await self._open_runtime_session(session, send_json)
            if open_result is False:
                return
            actor.runtime_opened = True
            actor.transition("open")
            created_payload: dict[str, object] = {
                "type": "session.created",
                "session": session.as_public_dict(),
            }
            if isinstance(open_result, dict):
                created_payload["runtime_control"] = self._redact_runtime_control_result(open_result)
            await send_json(created_payload)
            reader_task = asyncio.create_task(read_event_loop(), name="duplex-session-reader")

            while True:
                event = await next_actor_event()
                event_type = event.get("type")

                if event_type == "__timeout__":
                    actor.closing = True
                    actor.transition("closing", reason="timeout")
                    native_audio_buffer.clear()
                    native_response_emitted = False
                    native_input_since_commit = False
                    native_speech_since_commit = False
                    native_committed_audio_payload = None
                    native_deferred_response_create = False
                    await actor.cancel_append_tasks()
                    await self._cancel_native_data_plane_stream(session)
                    await self._cancel_active_response(
                        session,
                        actor.active_response_task,
                        send_json,
                        reason="timeout",
                        notify=True,
                    )
                    actor.active_response_task = None
                    actor.runtime_closed = await self._close_runtime_session(
                        session,
                        reason="timeout",
                        send_json=send_json,
                    )
                    if not actor.runtime_closed:
                        return
                    session.close()
                    actor.transition("closed", reason="timeout")
                    await send_json(
                        {
                            "type": "session.closed",
                            "session_id": session.session_id,
                            "reason": "timeout",
                        }
                    )
                    return

                if event_type == "__disconnect__":
                    return

                if not isinstance(event_type, str):
                    continue

                if event_type in {"session.close", "close_session"}:
                    if self._uses_native_input_append(session) and native_audio_buffer.has_pending():
                        flushed = native_audio_buffer.flush(
                            chunk_period_ms=session.capabilities.chunk_period_ms or 1000
                        )
                        if flushed is not None:
                            await start_native_append(flushed, final=True)
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(actor.output_queue.join(), timeout=1.0)
                    actor.closing = True
                    actor.transition("closing", reason="session_close")
                    if session.state == DuplexSessionState.CLOSED:
                        actor.runtime_closed = True
                        return
                    native_audio_buffer.clear()
                    native_response_emitted = False
                    native_input_since_commit = False
                    native_speech_since_commit = False
                    native_committed_audio_payload = None
                    native_deferred_response_create = False
                    await actor.cancel_append_tasks()
                    await self._cancel_native_data_plane_stream(session)
                    await self._cancel_active_response(
                        session,
                        actor.active_response_task,
                        send_json,
                        reason="session_close",
                    )
                    actor.runtime_closed = await self._close_runtime_session(
                        session,
                        reason="session_close",
                        send_json=send_json,
                    )
                    actor.active_response_task = None
                    if not actor.runtime_closed:
                        return
                    await send_json({"type": "session.closed", "session_id": session.session_id})
                    session.close()
                    actor.transition("closed", reason="session_close")
                    return

                if event_type == "input_audio_buffer.clear":
                    native_audio_buffer.clear()
                    native_input_since_commit = False
                    native_speech_since_commit = False
                    native_committed_audio_payload = None
                    native_deferred_response_create = False
                    drained = actor.drain_input_queue()
                    cancelled = session.cancel_pending_input()
                    await send_json(
                        {
                            "type": "input_audio_buffer.cleared",
                            "session_id": session.session_id,
                            "epoch": session.epoch,
                            "drained_input_events": drained,
                            "cancelled": cancelled,
                        }
                    )
                    continue

                if event_type in {"input.cancel", "response.cancel", "barge_in", "output_audio_buffer.clear"}:
                    cancel_reason = (
                        "output_audio_buffer_clear" if event_type == "output_audio_buffer.clear" else "barge_in"
                    )
                    if event_type == "response.cancel":
                        requested_response_id = event.get("response_id")
                        has_active_response_work = (
                            session.active_response_id is not None
                            or session.active_request_id is not None
                            or (actor.active_response_task is not None and not actor.active_response_task.done())
                            or actor.has_response_bound_append_tasks()
                            or session.session_id in self._native_data_plane_tasks
                            or actor.assistant_playback_active()
                        )
                        if (
                            isinstance(requested_response_id, str)
                            and session.active_response_id is not None
                            and requested_response_id != session.active_response_id
                        ):
                            await send_json(
                                {
                                    "type": "error",
                                    "session_id": session.session_id,
                                    "code": "response_not_active",
                                    "error": f"Response is not active: {requested_response_id}",
                                }
                            )
                            continue
                        if not has_active_response_work:
                            if realtime_protocol is not None and isinstance(requested_response_id, str):
                                continue
                            await send_json(
                                {
                                    "type": "error",
                                    "session_id": session.session_id,
                                    "code": "response_not_active",
                                    "error": "response.cancel requires an active response",
                                }
                            )
                            continue
                    actor.cancel_count += 1
                    actor.transition("cancelling", reason="barge_in")
                    had_native_unbuffered_append = (
                        self._uses_native_input_append(session)
                        and native_input_since_commit
                        and not native_audio_buffer.has_pending()
                    )
                    playback_was_active = actor.assistant_playback_active()
                    if event_type in {"input.cancel", "barge_in"}:
                        native_audio_buffer.clear()
                        native_input_since_commit = False
                        native_committed_audio_payload = None
                        native_deferred_response_create = False
                        actor.drain_input_queue()
                    native_response_emitted = False
                    had_native_append = await actor.cancel_append_tasks(
                        response_bound_only=event_type in {"response.cancel", "output_audio_buffer.clear"},
                    )
                    had_native_stream = session.session_id in self._native_data_plane_tasks
                    cancelled = await self._cancel_active_response(
                        session,
                        actor.active_response_task,
                        send_json,
                        reason=cancel_reason,
                        rebuild_runtime_history_after_truncate=True,
                    )
                    had_native_stream = await self._cancel_native_data_plane_stream(session) or had_native_stream
                    if not cancelled and (had_native_stream or had_native_append or had_native_unbuffered_append):
                        old_epoch = session.epoch
                        old_response_id = session.active_response_id
                        committed_ms = session.playback.committed_ms
                        self._commit_played_response_history(session, old_response_id, committed_ms)
                        if not await self._signal_runtime_history_rebuild_after_truncate(
                            session,
                            old_response_id,
                            committed_ms,
                            reason=cancel_reason,
                            send_json=send_json,
                        ):
                            continue
                        new_epoch, old_playback = self._advance_barge_in_epoch(session)
                        await send_json(
                            {
                                "type": "audio.cancelled",
                                "session_id": session.session_id,
                                "response_id": old_response_id,
                                "reason": cancel_reason,
                                "cancelled_epoch": old_epoch,
                                "epoch": new_epoch,
                                "committed_ms": committed_ms,
                                "playback": old_playback,
                            }
                        )
                        cancelled = True
                    if not cancelled and playback_was_active:
                        old_epoch = session.epoch
                        committed_ms = session.playback.committed_ms
                        self._commit_played_response_history(session, actor.last_response_id, committed_ms)
                        if not await self._signal_runtime_history_rebuild_after_truncate(
                            session,
                            actor.last_response_id,
                            committed_ms,
                            reason=cancel_reason,
                            send_json=send_json,
                        ):
                            continue
                        new_epoch, old_playback = self._advance_barge_in_epoch(session)
                        await send_json(
                            {
                                "type": "audio.cancelled",
                                "session_id": session.session_id,
                                "response_id": actor.last_response_id,
                                "reason": cancel_reason,
                                "cancelled_epoch": old_epoch,
                                "epoch": new_epoch,
                                "committed_ms": committed_ms,
                                "playback": old_playback,
                            }
                        )
                        cancelled = True
                    if not cancelled and event_type == "response.cancel":
                        old_epoch = session.epoch
                        old_response_id = session.active_response_id
                        committed_ms = session.playback.committed_ms
                        self._commit_played_response_history(session, old_response_id, committed_ms)
                        if not await self._signal_runtime_history_rebuild_after_truncate(
                            session,
                            old_response_id,
                            committed_ms,
                            reason=cancel_reason,
                            send_json=send_json,
                        ):
                            continue
                        new_epoch, old_playback = self._advance_barge_in_epoch(session)
                        await send_json(
                            {
                                "type": "audio.cancelled",
                                "session_id": session.session_id,
                                "response_id": old_response_id,
                                "reason": cancel_reason,
                                "cancelled_epoch": old_epoch,
                                "epoch": new_epoch,
                                "committed_ms": committed_ms,
                                "playback": old_playback,
                            }
                        )
                        cancelled = True
                    if not cancelled and event_type == "output_audio_buffer.clear":
                        old_playback = session.playback.as_dict()
                        committed_ms = session.playback.committed_ms
                        session.clear_playback_cursor()
                        await send_json(
                            {
                                "type": "audio.cancelled",
                                "session_id": session.session_id,
                                "response_id": session.active_response_id,
                                "reason": cancel_reason,
                                "cancelled_epoch": session.epoch,
                                "epoch": session.epoch,
                                "committed_ms": committed_ms,
                                "playback": old_playback,
                            }
                        )
                        cancelled = True
                    if not cancelled:
                        await self._cancel_pending_input(session, send_json, reason="barge_in")
                    if not await self._signal_runtime_session(session, "barge_in", event, send_json):
                        continue
                    actor.active_response_task = None
                    actor.transition("open")
                    continue

                if event_type in {"turn.signal", "signal_turn"}:
                    turn_event = event.get("event")
                    if isinstance(turn_event, str):
                        if turn_event == "session.update":
                            payload = event.get("payload")
                            if not isinstance(payload, dict):
                                await send_json(
                                    {
                                        "type": "error",
                                        "session_id": session.session_id,
                                        "code": "bad_event",
                                        "error": "session.update requires a session payload",
                                    }
                                )
                                continue
                            update_error = self._apply_session_update(session, payload)
                            if update_error is not None:
                                await send_json(update_error)
                                continue
                            if not await self._signal_runtime_session(session, turn_event, event, send_json):
                                continue
                            await send_json(
                                {
                                    "type": "session.updated",
                                    "session": session.as_public_dict(),
                                }
                            )
                            continue
                        if turn_event == "conversation.item.create":
                            payload = event.get("payload")
                            item = payload.get("item") if isinstance(payload, dict) else None
                            message = self._realtime_item_to_history_message(item)
                            item_id = item.get("id") if isinstance(item, dict) else None
                            if message is not None:
                                session.history.append(message)
                                session.register_history_item(item_id if isinstance(item_id, str) else None, message)
                            if not await self._signal_runtime_session(session, turn_event, event, send_json):
                                continue
                            await send_json(
                                {
                                    "type": "conversation.item.created",
                                    "session_id": session.session_id,
                                    "item": item,
                                    "created": message is not None,
                                }
                            )
                            continue
                        if turn_event == "conversation.item.delete":
                            payload = event.get("payload")
                            item_id = payload.get("item_id") if isinstance(payload, dict) else None
                            deleted = session.delete_history_item(item_id) if isinstance(item_id, str) else False
                            if not await self._signal_runtime_session(session, turn_event, event, send_json):
                                continue
                            await send_json(
                                {
                                    "type": "conversation.item.deleted",
                                    "session_id": session.session_id,
                                    "item_id": item_id,
                                    "deleted": deleted,
                                }
                            )
                            continue
                        if turn_event == "conversation.item.truncate":
                            payload = event.get("payload")
                            item_id = payload.get("item_id") if isinstance(payload, dict) else None
                            audio_end_ms = payload.get("audio_end_ms") if isinstance(payload, dict) else None
                            truncated = (
                                session.truncate_history_item(
                                    item_id,
                                    audio_end_ms=int(audio_end_ms) if isinstance(audio_end_ms, int | float) else 0,
                                )
                                if isinstance(item_id, str)
                                else False
                            )
                            signal_event = dict(event)
                            if isinstance(payload, dict):
                                signal_payload = dict(payload)
                                signal_payload["history"] = list(session.history)
                                signal_payload["playback"] = session.playback.as_dict()
                                signal_event["payload"] = signal_payload
                            else:
                                signal_event["payload"] = {
                                    "history": list(session.history),
                                    "playback": session.playback.as_dict(),
                                }
                            if not await self._signal_runtime_session(session, turn_event, signal_event, send_json):
                                continue
                            await send_json(
                                {
                                    "type": "conversation.item.truncated",
                                    "session_id": session.session_id,
                                    "item_id": item_id,
                                    "content_index": (
                                        payload.get("content_index", 0) if isinstance(payload, dict) else 0
                                    ),
                                    "audio_end_ms": audio_end_ms,
                                    "truncated": truncated,
                                }
                            )
                            continue
                        if not await self._signal_runtime_session(session, turn_event, event, send_json):
                            continue
                        await send_json(self._turn_controller.signal(session, turn_event, event))
                    else:
                        await send_json({"type": "error", "error": "turn.signal requires event", "code": "bad_event"})
                    continue

                if event_type in {"playback.ack", "audio.playback_ack"}:
                    await self._handle_playback_ack(session, event, send_json)
                    continue

                if event_type in {"input.text.append", "input_text.append", "push_text"}:
                    actor.transition("listening")
                    text = event.get("text")
                    if not isinstance(text, str):
                        await send_json(
                            {
                                "type": "error",
                                "error": "input.text.append requires text",
                                "code": "bad_event",
                            }
                        )
                        continue
                    if self._uses_native_input_append(session):
                        await send_json(
                            {
                                "type": "error",
                                "error": "MiniCPM-o native duplex currently accepts audio append only",
                                "code": "native_text_append_unsupported",
                            }
                        )
                        continue
                    else:
                        session.append_text(text)
                    if session.capabilities.supports_input_append:
                        await start_runtime_append(text, final=False, mode="append_tokens")
                    continue

                if event_type in {"input.audio.append", "input_audio_buffer.append", "push_chunk"}:
                    actor.transition("listening")
                    audio = event.get("audio") or event.get("data")
                    if not isinstance(audio, str):
                        await send_json(
                            {
                                "type": "error",
                                "error": "input.audio.append requires audio",
                                "code": "bad_event",
                            }
                        )
                        continue
                    if event_type == "input_audio_buffer.append":
                        fmt = event.get("format") if isinstance(event.get("format"), str) else "pcm16"
                        default_sample_rate_hz = 16000
                    else:
                        fmt = event.get("format") if isinstance(event.get("format"), str) else "wav"
                        default_sample_rate_hz = None
                    sr_raw = event.get("sample_rate_hz") or event.get("sample_rate")
                    sample_rate_hz = int(sr_raw) if isinstance(sr_raw, int | float) else default_sample_rate_hz
                    audio, fmt, sample_rate_hz = NativeRealtimeSessionProtocol._convert_realtime_input_audio_with_rate(
                        audio,
                        fmt,
                        sample_rate_hz=sample_rate_hz,
                    )
                    if isinstance(fmt, str) and fmt.lower() in {"pcm16", "pcm_s16le", "s16le"}:
                        await send_json(
                            {
                                "type": "error",
                                "error": "input_audio_buffer.append pcm16 audio could not be decoded",
                                "code": "bad_audio",
                            }
                        )
                        continue
                    force_listen = bool(event.get("force_listen", False))
                    payload = {
                        "type": "audio",
                        "audio": audio,
                        "format": fmt,
                        "sample_rate_hz": sample_rate_hz,
                        "force_listen": force_listen,
                    }
                    # Speech/silence tag for the Stage0 turn-ended latch.
                    payload["is_speech"] = self._input_looks_like_speech(event, payload, session=session)
                    defer_native_append = False
                    buffer_overlap_audio = True
                    if self._uses_native_input_append(session):
                        # Full-duplex: continuous mic chunks must FEED the ongoing
                        # stage0 stream (the model owns speak/listen), not be routed
                        # through the discrete-response overlap/barge-in policy.
                        overlap_active = (not self._session_auto_responds(session)) and (
                            native_response_in_progress()
                            or (event.get("_duplex_overlap_candidate") is True and actor.output_generation_in_flight())
                        )
                        if overlap_active:
                            decision = self._overlap_decision(session, actor, event, payload)
                            await self._emit_overlap_decision(send_json, session, decision)
                            action = decision.get("action")
                            if action == "drop":
                                if realtime_protocol is not None:
                                    await realtime_protocol.discard_pending_input_audio(
                                        audio_end_ms=self._input_audio_duration_ms(event, payload)
                                    )
                                actor.transition("generating")
                                continue
                            if action == "listen":
                                buffer_overlap_audio = bool(decision.get("buffer_audio", True))
                                defer_native_append = bool(decision.get("defer_runtime_append", True))
                                if not buffer_overlap_audio and realtime_protocol is not None:
                                    await realtime_protocol.discard_pending_input_audio(
                                        audio_end_ms=self._input_audio_duration_ms(event, payload)
                                    )
                                payload["force_listen"] = True
                                actor.transition("generating")
                            else:
                                event["force_barge_in"] = True
                                self._mark_barge_in_new_user_turn_payload(payload)
                                playback_was_active = actor.assistant_playback_active()
                                buffer_overlap_audio = True
                                defer_native_append = False
                                native_audio_buffer.clear_force_listen()
                                actor.overlap_speech_ms = 0
                                native_response_emitted = False
                                native_input_since_commit = False
                                native_speech_since_commit = False
                                await actor.cancel_append_tasks()
                                had_native_stream = session.session_id in self._native_data_plane_tasks
                                cancelled = await self._cancel_active_response(
                                    session,
                                    actor.active_response_task,
                                    send_json,
                                    reason="barge_in",
                                    rebuild_runtime_history_after_truncate=True,
                                )
                                had_native_stream = (
                                    await self._cancel_native_data_plane_stream(session) or had_native_stream
                                )
                                if not cancelled and had_native_stream:
                                    old_epoch = session.epoch
                                    old_response_id = session.active_response_id
                                    committed_ms = session.playback.committed_ms
                                    self._commit_played_response_history(session, old_response_id, committed_ms)
                                    if not await self._signal_runtime_history_rebuild_after_truncate(
                                        session,
                                        old_response_id,
                                        committed_ms,
                                        reason="barge_in",
                                        send_json=send_json,
                                    ):
                                        continue
                                    new_epoch, old_playback = self._advance_barge_in_epoch(session)
                                    await send_json(
                                        {
                                            "type": "audio.cancelled",
                                            "session_id": session.session_id,
                                            "response_id": old_response_id,
                                            "reason": "barge_in",
                                            "cancelled_epoch": old_epoch,
                                            "epoch": new_epoch,
                                            "committed_ms": committed_ms,
                                            "playback": old_playback,
                                        }
                                    )
                                    cancelled = True
                                if not cancelled and playback_was_active:
                                    old_epoch = session.epoch
                                    committed_ms = session.playback.committed_ms
                                    self._commit_played_response_history(session, actor.last_response_id, committed_ms)
                                    if not await self._signal_runtime_history_rebuild_after_truncate(
                                        session,
                                        actor.last_response_id,
                                        committed_ms,
                                        reason="barge_in",
                                        send_json=send_json,
                                    ):
                                        continue
                                    new_epoch, old_playback = self._advance_barge_in_epoch(session)
                                    await send_json(
                                        {
                                            "type": "audio.cancelled",
                                            "session_id": session.session_id,
                                            "response_id": actor.last_response_id,
                                            "reason": "barge_in",
                                            "cancelled_epoch": old_epoch,
                                            "epoch": new_epoch,
                                            "committed_ms": committed_ms,
                                            "playback": old_playback,
                                        }
                                    )
                                    cancelled = True
                                if not await self._signal_runtime_session(session, "barge_in", event, send_json):
                                    continue
                                actor.active_response_task = None
                        elif not self._session_auto_responds(session) and not self._input_looks_like_speech(
                            event, payload, session=session
                        ):
                            # Turn-mode only: skip silent chunks so they don't open a
                            # response. In auto-respond (full-duplex) mode the model owns
                            # the speak/listen decision and MUST receive silence units --
                            # the official model typically starts speaking during the
                            # silence after a question.
                            await send_json(
                                {
                                    "type": "response.listen",
                                    "session_id": session.session_id,
                                    "epoch": session.epoch,
                                    "reason": "silence_or_noise",
                                }
                            )
                            continue
                        new_user_turn_payload = self._mark_auto_response_new_user_turn_payload(
                            session,
                            actor,
                            event,
                            payload,
                        )
                        if new_user_turn_payload:
                            native_audio_buffer.clear_force_listen()
                        if self._should_skip_auto_response_waiting_silence(session, actor, event, payload):
                            continue
                        if not new_user_turn_payload and self._should_force_listen_for_auto_response_overlap(
                            session, actor, event, payload
                        ):
                            # Auto-response keeps a long-lived native Stage0 stream.
                            # While assistant audio is still active, silence from the
                            # browser should advance the model in listen mode rather
                            # than letting the same stream start another assistant
                            # segment. Speech/barge-in chunks remain model-owned.
                            payload["force_listen"] = True
                        if not buffer_overlap_audio:
                            continue
                        session.mark_user_input_activity()
                        native_input_since_commit = True
                        native_speech_since_commit = native_speech_since_commit or self._input_looks_like_speech(
                            event, payload, session=session
                        )
                        try:
                            buffered_payload = native_audio_buffer.append(
                                payload,
                                chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
                                allow_emit=(
                                    not defer_native_append
                                    and (
                                        realtime_protocol is None
                                        or event_type != "input_audio_buffer.append"
                                        # Full-duplex: emit each ~chunk_period of audio so the model
                                        # runs per-chunk generation (speak/listen) without an explicit
                                        # response.create, matching the official duplex_generate loop.
                                        or self._session_auto_responds(session)
                                    )
                                ),
                            )
                        except ValueError as exc:
                            await send_json({"type": "error", "error": str(exc), "code": "bad_event"})
                            continue
                        if buffered_payload is None:
                            continue
                        payload = buffered_payload
                    else:
                        session.append_audio(audio, fmt=fmt, sample_rate_hz=sample_rate_hz)
                    if self._uses_native_input_append(session):
                        await start_native_append(payload, final=False)
                        continue
                    if session.capabilities.supports_input_append:
                        await start_runtime_append(payload, final=False, mode="append_audio_chunk")
                    continue

                if event_type in {"input.commit", "input_audio_buffer.commit", "response.create"}:
                    if event_type == "input_audio_buffer.commit" and event.get("is_speech") is False:
                        native_input_since_commit = False
                        native_speech_since_commit = False
                        native_audio_buffer.clear()
                        native_committed_audio_payload = None
                        native_deferred_response_create = False
                        await send_json(
                            {
                                "type": "input.committed",
                                "session_id": session.session_id,
                                "turn_id": session.turn_id,
                                "epoch": session.epoch,
                                "empty": True,
                                "is_speech": False,
                                "no_response": True,
                            }
                        )
                        await send_json(
                            {
                                "type": "response.listen",
                                "session_id": session.session_id,
                                "epoch": session.epoch,
                                "reason": "silence_or_noise",
                            }
                        )
                        continue
                    should_create_response = event_type == "response.create" or bool(
                        event.get("response_create", event_type == "input.commit")
                    )
                    if event_type == "response.create":
                        response_payload = event.get("response")
                        if isinstance(response_payload, dict):
                            self._apply_response_create_options(session, response_payload)
                    if self._uses_native_input_append(session) and event_type == "input_audio_buffer.commit":
                        has_pending_native_audio = (
                            native_input_since_commit
                            or native_audio_buffer.has_pending()
                            or native_committed_audio_payload is not None
                        )
                        if (
                            not has_pending_native_audio
                            and not actor.native_append_tasks
                            and session.session_id not in self._native_data_plane_tasks
                        ):
                            await send_json(
                                {
                                    "type": "error",
                                    "session_id": session.session_id,
                                    "epoch": session.epoch,
                                    "code": "input_audio_buffer_empty",
                                    "error": "input_audio_buffer.commit requires a non-empty input audio buffer.",
                                }
                            )
                            continue
                        if native_response_in_progress() and actor.overlap_speech_ms > 0:
                            if actor.overlap_speech_ms <= session.config.overlap_short_ack_ms:
                                native_audio_buffer.clear()
                                native_input_since_commit = False
                                native_speech_since_commit = False
                                native_committed_audio_payload = None
                                native_deferred_response_create = False
                                if realtime_protocol is not None:
                                    await realtime_protocol.discard_pending_input_audio(
                                        audio_end_ms=actor.overlap_speech_ms
                                    )
                                await send_json(
                                    {
                                        "type": "input.committed",
                                        "session_id": session.session_id,
                                        "turn_id": session.turn_id,
                                        "epoch": session.epoch,
                                        "empty": True,
                                        "is_speech": False,
                                        "overlap_ack": True,
                                        "no_response": True,
                                    }
                                )
                                actor.overlap_speech_ms = 0
                                session.config.extra_body.pop("realtime_response_conversation", None)
                                continue

                            deferred_payload = native_audio_buffer.flush(
                                chunk_period_ms=session.capabilities.chunk_period_ms or 1000
                            )
                            if deferred_payload is not None:
                                if native_committed_audio_payload is not None:
                                    deferred_payload = self._merge_native_audio_payloads(
                                        native_committed_audio_payload,
                                        deferred_payload,
                                    )
                                native_committed_audio_payload = deferred_payload
                                native_deferred_response_create = should_create_response
                                native_input_since_commit = False
                                native_speech_since_commit = False
                                signal_runtime_background("input.commit", event)
                                committed = self._commit_native_audio_input(
                                    session,
                                    realtime_item_id=event.get("realtime_item_id"),
                                    transcript=event.get("transcript"),
                                )
                                committed_payload = self._native_audio_committed_payload(
                                    session,
                                    committed=committed,
                                    realtime_item_id=event.get("realtime_item_id"),
                                    transcript=event.get("transcript"),
                                )
                                committed_payload["overlap_deferred"] = True
                                committed_payload["response_create_deferred"] = should_create_response
                                await send_json(committed_payload)
                                continue
                        if (
                            self._session_auto_responds(session)
                            and native_speech_since_commit
                            and session.active_response_id is None
                        ):
                            committed_input = native_audio_buffer.commit(
                                chunk_period_ms=session.capabilities.chunk_period_ms or 1000
                            )
                            final_payload = committed_input.payload
                            if native_committed_audio_payload is not None:
                                if final_payload is not None:
                                    final_payload = self._merge_native_audio_payloads(
                                        native_committed_audio_payload,
                                        final_payload,
                                    )
                                else:
                                    final_payload = native_committed_audio_payload
                                native_committed_audio_payload = None
                                native_deferred_response_create = False
                            native_input_since_commit = False
                            native_speech_since_commit = False
                            signal_runtime_background("input.commit", event)
                            data_plane_turn_id = session.turn_id
                            committed = self._commit_native_audio_input(
                                session,
                                realtime_item_id=event.get("realtime_item_id"),
                                transcript=event.get("transcript"),
                            )
                            await send_json(
                                self._native_audio_committed_payload(
                                    session,
                                    committed=committed,
                                    realtime_item_id=event.get("realtime_item_id"),
                                    transcript=event.get("transcript"),
                                )
                            )
                            if final_payload is not None:
                                await start_native_append(
                                    {
                                        **final_payload,
                                        "duplex_turn_id": data_plane_turn_id,
                                    },
                                    final=True,
                                    precreate_response=False,
                                )
                            continue
                    if self._uses_native_input_append(session) and event_type == "response.create":
                        if (
                            native_response_in_progress()
                            or actor.native_append_tasks
                            or session.session_id in self._native_data_plane_tasks
                        ):
                            if session.active_response_id is None and (
                                session.active_request_id is not None
                                or actor.native_append_tasks
                                or session.session_id in self._native_data_plane_tasks
                            ):
                                continue
                            await send_json(
                                {
                                    "type": "error",
                                    "session_id": session.session_id,
                                    "epoch": session.epoch,
                                    "code": "response_already_active",
                                    "error": "response.create cannot start while another response is active.",
                                }
                            )
                            session.config.extra_body.pop("realtime_response_conversation", None)
                            continue
                        if native_committed_audio_payload is not None:
                            committed_payload = native_committed_audio_payload
                            native_committed_audio_payload = None
                            native_input_since_commit = False
                            native_speech_since_commit = False
                            native_deferred_response_create = False
                            await start_native_append(committed_payload, final=True, precreate_response=True)
                            continue
                        await send_json(
                            {
                                "type": "error",
                                "session_id": session.session_id,
                                "epoch": session.epoch,
                                "code": "response_create_without_input",
                                "error": "MiniCPM-o native duplex response.create requires committed audio input.",
                            }
                        )
                        session.config.extra_body.pop("realtime_response_conversation", None)
                        continue
                    if self._uses_native_input_append(session) and not native_response_in_progress():
                        flushed = native_audio_buffer.flush(
                            chunk_period_ms=session.capabilities.chunk_period_ms or 1000
                        )
                        if native_committed_audio_payload is not None:
                            if flushed is not None:
                                flushed = self._merge_native_audio_payloads(
                                    native_committed_audio_payload,
                                    flushed,
                                )
                            else:
                                flushed = native_committed_audio_payload
                            native_committed_audio_payload = None
                            native_deferred_response_create = False
                        if flushed is not None:
                            if self._should_force_listen_for_short_commit(session, event, flushed):
                                flushed = dict(flushed)
                                flushed["force_listen"] = True
                            native_input_since_commit = False
                            if event_type != "response.create":
                                signal_runtime_background("input.commit", event)
                            committed = self._commit_native_audio_input(
                                session,
                                realtime_item_id=event.get("realtime_item_id"),
                                transcript=event.get("transcript"),
                            )
                            await send_json(
                                self._native_audio_committed_payload(
                                    session,
                                    committed=committed,
                                    realtime_item_id=event.get("realtime_item_id"),
                                    transcript=event.get("transcript"),
                                )
                            )
                            if should_create_response:
                                await start_native_append(flushed, final=True, precreate_response=True)
                            else:
                                native_committed_audio_payload = flushed
                                native_deferred_response_create = False
                            continue
                        append_ok, emitted_response = True, False
                        actor.active_response_task = self._native_data_plane_tasks.get(
                            session.session_id,
                            actor.active_response_task,
                        )
                        if not append_ok:
                            if session.state == DuplexSessionState.CLOSED:
                                actor.runtime_closed = True
                                return
                            continue
                        if emitted_response:
                            native_response_emitted = True
                            native_audio_buffer.clear()
                            continue
                    if self._uses_native_input_append(session) and event_type in {
                        "input_audio_buffer.commit",
                        "input.commit",
                    }:
                        signal_runtime_background("input.commit", event)
                    else:
                        signal_runtime_background("input.commit", event)
                    native_had_uncommitted_audio = self._uses_native_input_append(session) and (
                        native_input_since_commit
                        or native_audio_buffer.has_pending()
                        or native_committed_audio_payload is not None
                    )
                    committed = session.commit_user_input()
                    if self._uses_native_input_append(session) and event_type in {
                        "input_audio_buffer.commit",
                        "input.commit",
                    }:
                        native_input_since_commit = False
                        native_speech_since_commit = False
                    if committed is None and event_type != "response.create":
                        if self._uses_native_input_append(session):
                            committed = (
                                self._commit_native_audio_input(
                                    session,
                                    realtime_item_id=event.get("realtime_item_id"),
                                    transcript=event.get("transcript"),
                                )
                                if native_had_uncommitted_audio
                                else None
                            )
                            await send_json(
                                self._native_audio_committed_payload(
                                    session,
                                    committed=committed,
                                    realtime_item_id=event.get("realtime_item_id"),
                                    transcript=event.get("transcript"),
                                )
                            )
                        else:
                            await send_json(
                                {
                                    "type": "input.committed",
                                    "session_id": session.session_id,
                                    "empty": True,
                                }
                            )
                        continue
                    if committed is not None:
                        realtime_item_id = event.get("realtime_item_id")
                        if isinstance(realtime_item_id, str):
                            session.register_history_item(realtime_item_id, committed.message)
                        await send_json(
                            self._input_committed_payload(
                                session,
                                committed,
                                realtime_item_id=realtime_item_id,
                            )
                        )
                    if not should_create_response:
                        continue
                    if actor.active_response_task is not None and not actor.active_response_task.done():
                        await self._cancel_active_response(
                            session,
                            actor.active_response_task,
                            send_json,
                            reason="new_user_turn",
                        )
                    actor.active_response_task = asyncio.create_task(self._run_response(session, send_json))
                    actor.transition("generating")
                    continue

                await send_json(
                    {
                        "type": "error",
                        "error": f"Unknown duplex event: {event_type}",
                        "code": "unknown_event",
                    }
                )

        except WebSocketDisconnect:
            logger.info("Duplex session disconnected")
        except Exception as exc:
            logger.exception("Duplex session failed: %s", exc)
            with suppress(Exception):
                await send_json({"type": "error", "error": str(exc), "code": "internal_error"})
        finally:
            if session is not None:
                actor.closing = True
                actor.transition("closing", reason=actor.close_reason or "disconnect")
                if reader_task is not None and not reader_task.done():
                    reader_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await reader_task
                await actor.cancel_append_tasks()
                await self._cancel_native_data_plane_stream(session)
                await self._cancel_active_response(
                    session,
                    actor.active_response_task,
                    send_json,
                    reason="disconnect",
                    notify=False,
                )
                if actor.runtime_opened and not actor.runtime_closed and session.state != DuplexSessionState.CLOSED:
                    await self._close_runtime_session(session, reason="disconnect")
                self._cleanup_duplex_session_state(session)
                self._registry.close(session.session_id)
            await actor.output_queue.put(None)
            with suppress(Exception):
                await asyncio.wait_for(actor.output_queue.join(), timeout=2.0)
            if not writer_task.done():
                writer_task.cancel()
            with suppress(asyncio.CancelledError):
                await writer_task

    async def handle_realtime_session(self, websocket: WebSocket) -> None:
        await self.handle_session(
            websocket,
            realtime_protocol=NativeRealtimeSessionProtocol(websocket.query_params),
        )

    @staticmethod
    def _is_duplex_control_event(event_type: str) -> bool:
        return is_control_event(event_type)

    @staticmethod
    def _is_duplex_input_event(event_type: str) -> bool:
        return is_input_event(event_type)

    @staticmethod
    def _advance_barge_in_epoch(session: DuplexSession) -> tuple[int, dict[str, int]]:
        old_playback = session.playback.as_dict()
        new_epoch = session.barge_in()
        session.clear_playback_cursor()
        return new_epoch, old_playback

    @staticmethod
    def _commit_played_response_history(
        session: DuplexSession,
        response_id: str | None,
        committed_ms: int,
    ) -> None:
        if not response_id or committed_ms < 0:
            return
        session.truncate_history_item(f"item_{response_id}", audio_end_ms=committed_ms)

    async def _signal_runtime_history_rebuild_after_truncate(
        self,
        session: DuplexSession,
        response_id: str | None,
        committed_ms: int,
        *,
        reason: str,
        send_json=None,
    ) -> bool:
        profile_logs = os.environ.get("MINICPMO45_PROFILE_LOGS") == "1"
        if not response_id or committed_ms < 0:
            if profile_logs:
                logger.info(
                    "skip duplex runtime history rebuild after truncate: session=%s "
                    "response_id=%s committed_ms=%s reason=%s",
                    session.session_id,
                    response_id,
                    committed_ms,
                    reason,
                )
            return True
        signal_event = {
            "type": "turn.signal",
            "event": "conversation.item.truncate",
            "payload": {
                "item_id": f"item_{response_id}",
                "audio_end_ms": int(committed_ms),
                "history": list(session.history),
                "playback": session.playback.as_dict(),
                "reason": reason,
            },
        }
        if profile_logs:
            logger.info(
                "signal duplex runtime history rebuild after truncate: session=%s "
                "response_id=%s committed_ms=%s reason=%s history_len=%d",
                session.session_id,
                response_id,
                committed_ms,
                reason,
                len(session.history),
            )
        result = await self._signal_runtime_session(
            session,
            "conversation.item.truncate",
            signal_event,
            send_json,
        )
        if profile_logs:
            logger.info(
                "duplex runtime history rebuild after truncate result: session=%s response_id=%s ok=%s",
                session.session_id,
                response_id,
                result,
            )
        return result

    def _remember_response_conversation_mode(self, session: DuplexSession, response_id: str) -> None:
        mode = session.config.extra_body.pop("realtime_response_conversation", None)
        if isinstance(mode, str):
            self._response_conversation_modes[response_id] = mode.strip().lower()

    def _should_commit_response_to_history(self, response_id: str | None) -> bool:
        if response_id is None:
            return True
        return self._response_conversation_modes.pop(response_id, "auto") != "none"

    def _response_created_payload(
        self,
        session: DuplexSession,
        response_id: str,
        *,
        epoch: int,
        request_id: str | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "response.created",
            "session_id": session.session_id,
            "response_id": response_id,
            "epoch": epoch,
            "modalities": list(session.config.modalities),
        }
        if request_id is not None:
            payload["request_id"] = request_id
        metadata = session.config.extra_body.pop("realtime_response_metadata", None)
        if not isinstance(metadata, dict):
            metadata = session.config.extra_body.get("realtime_metadata")
        if isinstance(metadata, dict):
            payload["metadata"] = dict(metadata)
        conversation = session.config.extra_body.get("realtime_response_conversation")
        if isinstance(conversation, str):
            payload["conversation"] = conversation
        prompt = session.config.extra_body.pop("realtime_response_prompt", None)
        if isinstance(prompt, dict):
            payload["prompt"] = dict(prompt)
        return payload

    def _overlap_decision(
        self,
        session: DuplexSession,
        actor: DuplexSessionActor,
        event: dict[str, object],
        payload: dict[str, object],
    ) -> dict[str, object]:
        """Classify input that arrives while assistant audio is active.

        This is a serving-side policy. The model still owns listen/speak
        decisions for normal chunks; overlap policy only decides whether the
        current assistant response should be interrupted before the new audio is
        appended.
        """
        duration_ms = self._input_audio_duration_ms(event, payload)
        is_speech = self._input_looks_like_speech(event, payload, session=session)
        explicit = event.get("overlap_action") or event.get("overlap")
        if isinstance(explicit, str):
            normalized = explicit.strip().lower()
            if normalized in {"barge_in", "interrupt", "cancel"}:
                return {
                    "action": "barge_in",
                    "reason": "client_overlap_action",
                    "duration_ms": duration_ms,
                    "buffer_audio": True,
                }
            if normalized in {"listen", "continue", "continue_output", "ack"}:
                actor.overlap_speech_ms = 0
                return {
                    "action": "listen",
                    "reason": "client_overlap_action",
                    "duration_ms": duration_ms,
                    "buffer_audio": (
                        normalized == "listen" and is_speech and duration_ms > session.config.overlap_short_ack_ms
                    ),
                    "defer_runtime_append": True,
                }
            if normalized in {"drop", "ignore", "silence"}:
                actor.overlap_speech_ms = 0
                return {
                    "action": "drop",
                    "reason": "client_overlap_action",
                    "duration_ms": duration_ms,
                    "buffer_audio": False,
                }

        if bool(event.get("force_barge_in", False)):
            return {
                "action": "barge_in",
                "reason": "client_force_barge_in",
                "duration_ms": duration_ms,
                "buffer_audio": True,
            }
        if self._session_auto_responds(session):
            # Full-duplex: continuous input while the model is speaking is the
            # normal case (the client streams the mic non-stop). Buffer it as the
            # next turn's input and let the model finish its current response,
            # instead of auto-cancelling on every overlapping chunk. Explicit
            # barge-in (force_barge_in / overlap_action above) still interrupts.
            actor.overlap_speech_ms = 0
            return {
                "action": "listen",
                "reason": "auto_response_continuous",
                "duration_ms": duration_ms,
                "buffer_audio": is_speech,
                "defer_runtime_append": True,
            }
        if bool(event.get("force_listen", False)):
            actor.overlap_speech_ms = 0
            return {
                "action": "listen",
                "reason": "client_force_listen",
                "duration_ms": duration_ms,
                "buffer_audio": is_speech,
                "defer_runtime_append": True,
            }

        policy = session.config.overlap_policy
        if not is_speech:
            if actor.overlap_speech_ms <= 0:
                actor.overlap_speech_ms = 0
            return {
                "action": "drop",
                "reason": "silence_or_noise",
                "duration_ms": duration_ms,
                "overlap_speech_ms": actor.overlap_speech_ms,
                "buffer_audio": False,
            }

        if self._is_short_ack_transcript_hint(event, payload):
            actor.overlap_speech_ms = 0
            return {
                "action": "listen",
                "reason": "short_ack_transcript",
                "duration_ms": duration_ms,
                "overlap_speech_ms": actor.overlap_speech_ms,
                "buffer_audio": False,
                "defer_runtime_append": True,
            }

        if policy == DuplexOverlapPolicy.LISTEN_ONLY.value:
            actor.overlap_speech_ms += max(0, duration_ms)
            return {
                "action": "listen",
                "reason": "policy_listen_only",
                "duration_ms": duration_ms,
                "overlap_speech_ms": actor.overlap_speech_ms,
                "buffer_audio": True,
                "defer_runtime_append": True,
            }

        actor.overlap_speech_ms += max(0, duration_ms)
        if policy == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value:
            return {
                "action": "barge_in",
                "reason": "policy_barge_in_on_speech",
                "duration_ms": duration_ms,
                "overlap_speech_ms": actor.overlap_speech_ms,
                "buffer_audio": True,
            }

        if (
            duration_ms <= session.config.overlap_short_ack_ms
            and actor.overlap_speech_ms <= session.config.overlap_short_ack_ms
        ):
            return {
                "action": "listen",
                "reason": "short_ack",
                "duration_ms": duration_ms,
                "overlap_speech_ms": actor.overlap_speech_ms,
                "buffer_audio": True,
                "defer_runtime_append": True,
            }
        if actor.overlap_speech_ms >= session.config.overlap_barge_in_ms:
            return {
                "action": "barge_in",
                "reason": "long_overlap_speech",
                "duration_ms": duration_ms,
                "overlap_speech_ms": actor.overlap_speech_ms,
                "buffer_audio": True,
            }
        return {
            "action": "listen",
            "reason": "accumulating_overlap_speech",
            "duration_ms": duration_ms,
            "overlap_speech_ms": actor.overlap_speech_ms,
            "buffer_audio": True,
            "defer_runtime_append": True,
        }

    @staticmethod
    def _is_short_ack_transcript_hint(event: dict[str, object], payload: dict[str, object]) -> bool:
        raw_text = event.get("transcript") or event.get("text") or payload.get("transcript") or payload.get("text")
        if not isinstance(raw_text, str):
            return False
        normalized = raw_text.strip().lower()
        if not normalized:
            return False
        compact = "".join(ch for ch in normalized if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        if compact in {
            "嗯",
            "嗯嗯",
            "对",
            "对的",
            "好",
            "好的",
            "继续",
            "继续说",
            "可以",
            "是的",
            "yes",
            "yeah",
            "yep",
            "ok",
            "okay",
            "continue",
            "goon",
            "right",
        }:
            return True
        return normalized in {"go on", "keep going", "please continue"}

    @staticmethod
    def _input_audio_duration_ms(event: dict[str, object], payload: dict[str, object]) -> int:
        for key in ("duration_ms", "audio_duration_ms"):
            value = event.get(key)
            if isinstance(value, int | float):
                return max(0, int(value))
        fmt = payload.get("format")
        sample_rate_hz = payload.get("sample_rate_hz")
        audio = payload.get("audio")
        if fmt == "pcm_f32le" and isinstance(sample_rate_hz, int) and sample_rate_hz > 0 and isinstance(audio, str):
            try:
                raw = base64.b64decode(audio, validate=True)
            except (binascii.Error, ValueError):
                return 0
            return int((len(raw) // 4) * 1000 / sample_rate_hz)
        return 0

    @staticmethod
    def _merge_native_audio_payloads(
        first: dict[str, object],
        second: dict[str, object],
    ) -> dict[str, object]:
        if first.get("format") != "pcm_f32le" or second.get("format") != "pcm_f32le":
            return second
        first_rate = first.get("sample_rate_hz")
        second_rate = second.get("sample_rate_hz")
        if not isinstance(first_rate, int) or not isinstance(second_rate, int) or first_rate != second_rate:
            return second
        first_audio = first.get("audio")
        second_audio = second.get("audio")
        if not isinstance(first_audio, str) or not isinstance(second_audio, str):
            return second
        try:
            first_raw = base64.b64decode(first_audio, validate=True)
            second_raw = base64.b64decode(second_audio, validate=True)
        except (binascii.Error, ValueError):
            return second
        merged = dict(second)
        merged["audio"] = base64.b64encode(first_raw + second_raw).decode("ascii")
        merged["sample_rate_hz"] = first_rate
        merged["force_listen"] = bool(first.get("force_listen", False)) or bool(second.get("force_listen", False))
        merged.pop("force_speak", None)
        merged["is_speech"] = bool(first.get("is_speech", False)) or bool(second.get("is_speech", False))
        if bool(first.get("new_user_turn", False)) or bool(second.get("new_user_turn", False)):
            merged["new_user_turn"] = True
            prefix_variant = first.get("new_user_turn_prefix_variant") or second.get("new_user_turn_prefix_variant")
            if isinstance(prefix_variant, str):
                merged["new_user_turn_prefix_variant"] = prefix_variant
        return merged

    @classmethod
    def _should_force_listen_for_short_commit(
        cls,
        session: DuplexSession,
        event: dict[str, object],
        payload: dict[str, object],
    ) -> bool:
        """Keep very short committed Realtime chunks in listen mode.

        Realtime VAD can emit a commit for a short pause even when the user has
        not actually yielded the turn. For MiniCPM-o native duplex, make that
        policy explicit by steering the scheduler path to the model listen
        token instead of letting a sub-second chunk start a response.
        """
        if event.get("force_listen") is True or payload.get("force_listen") is True:
            return True
        if event.get("force_barge_in") is True:
            return False
        if event.get("response_create") is not True:
            return False
        duration_ms = cls._input_audio_duration_ms(event, payload)
        return 0 < duration_ms <= session.config.overlap_short_ack_ms

    def _should_force_listen_for_auto_response_overlap(
        self,
        session: DuplexSession,
        actor: DuplexSessionActor,
        event: dict[str, object],
        payload: dict[str, object],
    ) -> bool:
        if not self._session_auto_responds(session):
            return False
        if event.get("force_barge_in") is True:
            return False
        if event.get("force_listen") is True or payload.get("force_listen") is True:
            return True
        looks_like_speech = self._input_looks_like_speech(event, payload, session=session)
        if session.session_id in self._auto_response_waiting_for_speech:
            return False
        if actor.assistant_playback_active():
            return True
        if looks_like_speech:
            self._auto_response_waiting_for_speech.discard(session.session_id)
            self._auto_response_new_turn_prefix_variants.pop(session.session_id, None)
            return False
        return False

    def _should_skip_auto_response_waiting_silence(
        self,
        session: DuplexSession,
        actor: DuplexSessionActor,
        event: dict[str, object],
        payload: dict[str, object],
    ) -> bool:
        if not self._session_auto_responds(session):
            return False
        if session.session_id not in self._auto_response_waiting_for_speech:
            return False
        if event.get("force_barge_in") is True:
            return False
        if event.get("force_listen") is True:
            return False
        return not self._input_looks_like_speech(event, payload, session=session)

    def _mark_auto_response_waiting_for_speech(
        self,
        session: DuplexSession,
        *,
        prefix_variant: str,
    ) -> None:
        if not self._session_auto_responds(session):
            return
        self._auto_response_waiting_for_speech.add(session.session_id)
        self._auto_response_new_turn_prefix_variants[session.session_id] = prefix_variant

    @staticmethod
    def _mark_barge_in_new_user_turn_payload(payload: dict[str, object]) -> None:
        payload["new_user_turn"] = True
        payload.setdefault(
            "new_user_turn_prefix_variant",
            MiniCPMO45DuplexPolicy.NEW_USER_TURN_PREFIX_INTERRUPTED_TTS,
        )

    def _mark_auto_response_new_user_turn_payload(
        self,
        session: DuplexSession,
        actor: DuplexSessionActor,
        event: dict[str, object],
        payload: dict[str, object],
    ) -> bool:
        profile_logs = os.environ.get("MINICPMO45_PROFILE_LOGS") == "1"
        if not self._session_auto_responds(session):
            return False
        if event.get("force_listen") is True:
            if profile_logs:
                logger.info(
                    "MiniCPM-o auto-response skip new user turn: session=%s reason=event_force_listen",
                    session.session_id,
                )
            return False
        force_barge_in = event.get("force_barge_in") is True
        if force_barge_in:
            prefix_variant = self._auto_response_new_turn_prefix_variants.pop(
                session.session_id,
                MiniCPMO45DuplexPolicy.NEW_USER_TURN_PREFIX_INTERRUPTED_TTS,
            )
        else:
            if session.session_id not in self._auto_response_waiting_for_speech:
                if profile_logs:
                    logger.info(
                        "MiniCPM-o auto-response skip new user turn: session=%s reason=waiting_latch_unset "
                        "payload_flags={is_speech:%s,force_listen:%s}",
                        session.session_id,
                        payload.get("is_speech"),
                        payload.get("force_listen"),
                    )
                return False
            if actor.assistant_playback_active():
                if profile_logs:
                    logger.info(
                        "MiniCPM-o auto-response skip new user turn: session=%s reason=playback_active "
                        "payload_flags={is_speech:%s,force_listen:%s}",
                        session.session_id,
                        payload.get("is_speech"),
                        payload.get("force_listen"),
                    )
                return False
            if not self._input_looks_like_speech(event, payload, session=session):
                if profile_logs:
                    logger.info(
                        "MiniCPM-o auto-response skip new user turn: session=%s reason=input_not_speech "
                        "payload_flags={is_speech:%s,force_listen:%s}",
                        session.session_id,
                        payload.get("is_speech"),
                        payload.get("force_listen"),
                    )
                return False
            prefix_variant = self._auto_response_new_turn_prefix_variants.pop(session.session_id, None)
        self._auto_response_waiting_for_speech.discard(session.session_id)
        payload["force_listen"] = False
        if prefix_variant:
            payload.setdefault("new_user_turn_prefix_variant", prefix_variant)
        payload["new_user_turn"] = True
        if profile_logs:
            logger.info(
                "MiniCPM-o auto-response mark new user turn: session=%s "
                "playback_active=%s payload_flags={is_speech:%s,force_listen:%s,"
                "force_barge_in:%s,new_user_turn:%s,prefix_variant:%s}",
                session.session_id,
                actor.assistant_playback_active(),
                payload.get("is_speech"),
                payload.get("force_listen"),
                event.get("force_barge_in"),
                payload.get("new_user_turn"),
                payload.get("new_user_turn_prefix_variant"),
            )
        return True

    @staticmethod
    def _input_looks_like_speech(
        event: dict[str, object],
        payload: dict[str, object],
        *,
        session: DuplexSession,
    ) -> bool:
        for key in ("is_speech", "speech"):
            value = event.get(key)
            if isinstance(value, bool):
                return value
        vad = event.get("vad")
        if isinstance(vad, dict):
            value = vad.get("is_speech")
            if isinstance(value, bool):
                return value
            probability = vad.get("speech_probability", vad.get("probability"))
            if isinstance(probability, int | float):
                return float(probability) >= 0.5
        probability = event.get("speech_probability")
        if isinstance(probability, int | float):
            return float(probability) >= 0.5

        fmt = payload.get("format")
        audio = payload.get("audio")
        if fmt in {"pcm_f32le", "pcm16"} and isinstance(audio, str):
            try:
                raw = base64.b64decode(audio, validate=True)
            except (binascii.Error, ValueError):
                return True
            if fmt == "pcm_f32le":
                if len(raw) < 4 or len(raw) % 4 != 0:
                    return True
                samples = np.frombuffer(raw, dtype=np.float32)
            else:
                if len(raw) < 2 or len(raw) % 2 != 0:
                    return True
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size == 0:
                return False
            rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
            return rms >= session.config.overlap_silence_rms
        return True

    @staticmethod
    def _input_explicitly_non_speech(event: dict[str, object]) -> bool:
        for key in ("is_speech", "speech"):
            value = event.get(key)
            if isinstance(value, bool):
                return not value
        vad = event.get("vad")
        if isinstance(vad, dict):
            value = vad.get("is_speech")
            if isinstance(value, bool):
                return not value
            probability = vad.get("speech_probability", vad.get("probability"))
            if isinstance(probability, int | float):
                return float(probability) < 0.5
        probability = event.get("speech_probability")
        return isinstance(probability, int | float) and float(probability) < 0.5

    async def _emit_overlap_decision(
        self,
        send_json,
        session: DuplexSession,
        decision: dict[str, object],
    ) -> None:
        await send_json(
            {
                "type": "overlap.decision",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "policy": session.config.overlap_policy,
                **decision,
            }
        )

    async def _open_runtime_session(self, session: DuplexSession, send_json) -> dict[str, object] | bool:
        open_session = getattr(self._chat_service.engine_client, "open_duplex_session_async", None)
        if not callable(open_session):
            return True
        try:
            open_kwargs = {
                "session_mode": "duplex",
                "capabilities": session.capabilities.as_dict(),
                "session_config": session.config.as_dict(),
                "timeout": self._runtime_control_timeout_s(session),
            }
            if self._callable_accepts_keyword(open_session, "fence"):
                open_kwargs["fence"] = DuplexFence(
                    session.session_id,
                    epoch=session.epoch,
                    turn_id=session.turn_id,
                )
            result = await open_session(session.session_id, **open_kwargs)
        except Exception as exc:
            logger.exception("Failed to open duplex runtime session: %s", exc)
            await self._send_runtime_error(send_json, "runtime_open_failed", exc, session=session)
            return False
        if (
            isinstance(result, dict)
            and session.capabilities.implementation_level == "model_native_duplex"
            and self._runtime_control_failed(result)
        ):
            await send_json(
                {
                    "type": "error",
                    "error": "Native duplex runtime is not available for this session",
                    "code": ("runtime_open_unsupported" if result.get("unsupported_count") else "runtime_open_failed"),
                    "runtime_control": self._redact_runtime_control_result(result),
                }
            )
            return False
        return result if isinstance(result, dict) else True

    async def _append_runtime_input(
        self,
        session: DuplexSession,
        payload: object,
        *,
        final: bool,
        send_json,
        mode: str = "append_tokens",
        expected_epoch: int | None = None,
    ) -> tuple[bool, bool]:
        if not session.capabilities.supports_input_append:
            return True, False
        append_input = getattr(self._chat_service.engine_client, "append_duplex_input_async", None)
        if not callable(append_input):
            return True, False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return True, False
        try:
            append_kwargs = {
                "mode": mode,
                "payload": payload,
                "final": final,
                "timeout": self._runtime_control_timeout_s(session),
            }
            if expected_epoch is not None and self._callable_accepts_keyword(append_input, "expected_epoch"):
                append_kwargs["expected_epoch"] = expected_epoch
            if self._callable_accepts_keyword(append_input, "fence"):
                payload_turn_id = self._payload_turn_id(payload)
                append_kwargs["fence"] = DuplexFence(
                    session.session_id,
                    epoch=session.epoch,
                    turn_id=(
                        payload_turn_id
                        if payload_turn_id is not None
                        else (
                            session.active_response_turn_id
                            if session.active_response_turn_id is not None
                            else session.turn_id
                        )
                    ),
                )
            if self._callable_accepts_keyword(append_input, "collect_outputs"):
                append_kwargs["collect_outputs"] = False
            result = await append_input(session.session_id, **append_kwargs)
        except Exception as exc:
            logger.exception("Failed to append duplex runtime input: %s", exc)
            await self._send_runtime_error(send_json, "runtime_append_failed", exc, session=session)
            return False, False
        if isinstance(result, dict) and self._runtime_control_failed(result):
            await self._send_runtime_control_error(
                send_json,
                "runtime_append_failed",
                "Duplex runtime append failed",
                result,
                session=session,
            )
            return False, False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return True, False
        await self._send_runtime_control_if_needed(send_json, result, session=session)
        request_id, response_stage_id = (
            self._data_plane_request_info(result) if isinstance(result, dict) else (None, None)
        )
        if request_id is not None:
            self._data_plane_terminal_request_ids.discard(request_id)
            self._data_plane_tts_eos_done.discard(request_id)
            self._data_plane_turn_eos_done.discard(request_id)
        if os.environ.get("MINICPMO45_PROFILE_LOGS") == "1" and isinstance(result, dict):
            dp_outputs = result.get("data_plane_outputs")
            logger.info(
                "[append-result] request_info=%s n_dp_outputs=%s keys=%s",
                (request_id, response_stage_id),
                len(dp_outputs) if isinstance(dp_outputs, list) else None,
                sorted(result.keys()),
            )
        close_reason, emitted_response = await self._send_native_duplex_events(
            send_json,
            result,
            session=session,
            expected_epoch=expected_epoch,
        )
        # A resumable data-plane append remains active even when its persistent
        # drain task already exists. Treating only a newly-created drain as
        # activity clears active_request_id after later appends and prevents a
        # terminal TTS segment from scheduling the next model decision.
        emitted_response = emitted_response or request_id is not None
        if close_reason is None and await self._start_native_data_plane_stream_task(
            send_json,
            result,
            session=session,
            expected_epoch=expected_epoch,
        ):
            emitted_response = True
        if close_reason is not None:
            if not await self._close_runtime_session(session, reason=close_reason, send_json=send_json):
                return False, emitted_response
            session.close()
            await send_json(
                {
                    "type": "session.closed",
                    "session_id": session.session_id,
                    "reason": close_reason,
                }
            )
            return False, emitted_response
        return True, emitted_response

    @staticmethod
    def _callable_accepts_keyword(fn, name: str) -> bool:
        try:
            params = inspect.signature(fn).parameters.values()
        except (TypeError, ValueError):
            return True
        return any(param.kind == inspect.Parameter.VAR_KEYWORD or param.name == name for param in params)

    async def _start_native_data_plane_stream_task(
        self,
        send_json,
        result: object,
        *,
        session: DuplexSession,
        expected_epoch: int | None = None,
    ) -> bool:
        request_id, _ = self._data_plane_request_info(result)
        profile_logs = os.environ.get("MINICPMO45_PROFILE_LOGS") == "1"
        if request_id is None or self._data_plane_outputs_finished(result):
            if profile_logs:
                logger.info(
                    "[drain-start] skip: request_id=%s finished=%s session=%s",
                    request_id,
                    self._data_plane_outputs_finished(result),
                    session.session_id,
                )
            return False
        session.active_request_id = request_id

        old_task = self._native_data_plane_tasks.get(session.session_id)
        if old_task is not None and not old_task.done():
            if self._session_auto_responds(session):
                # One persistent drain per session, like the official worker's
                # single synchronous loop: the resumable data-plane request id
                # is stable, and cancel/restart on every append orphans any
                # decision that lands in the swap window.
                if profile_logs:
                    logger.info(
                        "[drain-start] keep-alive: existing drain for session=%s; append request_id=%s",
                        session.session_id,
                        request_id,
                    )
                return False
            if profile_logs:
                logger.info(
                    "[drain-start] cancel-restart: session=%s new request_id=%s",
                    session.session_id,
                    request_id,
                )
            old_task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(old_task, return_exceptions=True), timeout=0.25)
            except asyncio.TimeoutError:
                pass

        async def _run() -> None:
            close_reason: str | None = None
            try:
                close_reason = await self._drain_native_data_plane_stream(
                    send_json,
                    result,
                    session=session,
                    expected_epoch=expected_epoch,
                )
                if close_reason is not None:
                    if await self._close_runtime_session(session, reason=close_reason, send_json=send_json):
                        session.close()
                        await send_json(
                            {
                                "type": "session.closed",
                                "session_id": session.session_id,
                                "reason": close_reason,
                            }
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Failed to drain duplex data-plane stream: %s", exc)
                await self._send_runtime_error(send_json, "runtime_data_plane_stream_failed", exc, session=session)
                if session.state != DuplexSessionState.CLOSED:
                    close_reason = "runtime_data_plane_stream_failed"
                    if await self._close_runtime_session(session, reason=close_reason, send_json=send_json):
                        session.close()
                        await send_json(
                            {
                                "type": "session.closed",
                                "session_id": session.session_id,
                                "reason": close_reason,
                            }
                        )
            finally:
                current = self._native_data_plane_tasks.get(session.session_id)
                if current is task:
                    self._native_data_plane_tasks.pop(session.session_id, None)
                if close_reason is None and not self._session_auto_responds(session):
                    # Auto-respond sessions keep one resumable stage-1 stream
                    # whose audio accumulates across speak units; the offset
                    # must survive drain-task turnover or every unit re-sends
                    # the reply audio from the start.
                    self._data_plane_audio_offsets.pop(request_id, None)
                    self._data_plane_sent_segment_texts.pop(request_id, None)
            if close_reason is None:
                self._maybe_continue_native_response(send_json, session=session, expected_epoch=expected_epoch)

        if profile_logs:
            logger.info(
                "[drain-start] started: session=%s request_id=%s",
                session.session_id,
                request_id,
            )
        task = asyncio.create_task(_run())
        self._native_data_plane_tasks[session.session_id] = task
        return True

    # One model unit (1 s at 16 kHz) of pcm_f32le silence, matching the
    # official full-duplex behavior where the microphone keeps streaming
    # silence while the assistant speaks; replies span multiple units.
    _NATIVE_SILENCE_UNIT_PAYLOAD_AUDIO = base64.b64encode(bytes(16000 * 4)).decode("ascii")
    _NATIVE_RESPONSE_MAX_CONTINUATION_UNITS = 8

    def _native_silence_unit_payload(self) -> dict[str, object]:
        return {
            "type": "audio",
            "audio": self._NATIVE_SILENCE_UNIT_PAYLOAD_AUDIO,
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        }

    def _native_response_continuations_remaining(self, session: DuplexSession, response_id: str) -> bool:
        if self._session_auto_responds(session):
            return True
        prev_response_id, count = self._native_response_continuations.get(session.session_id, (response_id, 0))
        if prev_response_id != response_id:
            count = 0
        return count < self._NATIVE_RESPONSE_MAX_CONTINUATION_UNITS

    def _maybe_continue_native_response(
        self,
        send_json,
        *,
        session: DuplexSession,
        expected_epoch: int | None,
    ) -> None:
        """Keep an open spoken response going with silence units.

        The model generates one unit per append; official replies span
        several units, so while a response is still open after a segment
        finishes, append a silence unit (the official microphone-keeps-
        streaming beat) until the model ends the turn or the cap is reached.
        """
        response_id = session.active_response_id
        if response_id is None or session.state == DuplexSessionState.CLOSED:
            self._native_response_continuations.pop(session.session_id, None)
            return
        request_id = session.active_request_id
        if request_id is None:
            self._native_response_continuations.pop(session.session_id, None)
            return
        if expected_epoch is not None and session.epoch != expected_epoch:
            return
        prev_response_id, count = self._native_response_continuations.get(session.session_id, (response_id, 0))
        if prev_response_id != response_id:
            count = 0
        auto_response = self._session_auto_responds(session)
        if not auto_response and count >= self._NATIVE_RESPONSE_MAX_CONTINUATION_UNITS:
            return
        self._native_response_continuations[session.session_id] = (response_id, count + 1)
        payload = self._native_silence_unit_payload()
        payload["duplex_turn_id"] = (
            session.active_response_turn_id if session.active_response_turn_id is not None else session.turn_id
        )

        async def _continue() -> None:
            try:
                if (
                    session.state == DuplexSessionState.CLOSED
                    or session.active_response_id != response_id
                    or session.active_request_id != request_id
                    or (expected_epoch is not None and session.epoch != expected_epoch)
                ):
                    return
                await self._append_runtime_input(
                    session,
                    payload,
                    final=False,
                    send_json=send_json,
                    mode="append_audio_chunk",
                    expected_epoch=expected_epoch,
                )
            except Exception as exc:
                logger.exception("Failed to continue duplex native response: %s", exc)

        asyncio.create_task(_continue())

    async def _cancel_native_data_plane_stream(self, session: DuplexSession) -> bool:
        task = self._native_data_plane_tasks.pop(session.session_id, None)
        if task is None or task.done():
            return False
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=0.25)
        except asyncio.TimeoutError:
            # Keep barge-in/cancel responsive. The task still carries the old
            # expected_epoch and late model output is filtered by the writer.
            pass
        return True

    async def _signal_runtime_session(
        self,
        session: DuplexSession,
        event: str,
        payload: dict[str, object] | None = None,
        send_json=None,
        *,
        fence: DuplexFence | None = None,
    ) -> bool:
        signal_turn = getattr(self._chat_service.engine_client, "signal_duplex_turn_async", None)
        if not callable(signal_turn):
            return True
        try:
            signal_kwargs = {
                "event": event,
                "payload": payload,
                "timeout": self._runtime_control_timeout_s(session),
            }
            if self._callable_accepts_keyword(signal_turn, "fence"):
                signal_kwargs["fence"] = fence or DuplexFence(
                    session.session_id, epoch=session.epoch, turn_id=session.turn_id
                )
            result = await signal_turn(session.session_id, **signal_kwargs)
        except Exception as exc:
            logger.exception("Failed to signal duplex runtime session: %s", exc)
            if send_json is not None:
                await self._send_runtime_error(send_json, "runtime_signal_failed", exc, session=session)
            return False
        if isinstance(result, dict) and self._runtime_signal_failed(result):
            logger.warning(
                "Duplex runtime signal failed: session=%s event=%s result=%s",
                session.session_id,
                event,
                self._redact_runtime_control_result(result),
            )
            if send_json is not None:
                await self._send_runtime_control_error(
                    send_json,
                    "runtime_signal_failed",
                    "Duplex runtime signal failed",
                    result,
                    session=session,
                )
            return False
        await self._send_runtime_control_if_needed(send_json, result, session=session)
        return True

    async def _close_runtime_session(self, session: DuplexSession, *, reason: str, send_json=None) -> bool:
        close_session = getattr(self._chat_service.engine_client, "close_duplex_session_async", None)
        if not callable(close_session):
            return True
        try:
            close_kwargs = {
                "reason": reason,
                "timeout": self._runtime_control_timeout_s(session),
            }
            if self._callable_accepts_keyword(close_session, "fence"):
                close_kwargs["fence"] = DuplexFence(
                    session.session_id,
                    epoch=session.epoch,
                    turn_id=session.turn_id,
                )
            result = await close_session(session.session_id, **close_kwargs)
        except Exception as exc:
            logger.exception("Failed to close duplex runtime session: %s", exc)
            if send_json is not None:
                await self._send_runtime_error(send_json, "runtime_close_failed", exc, session=session)
            return False
        if isinstance(result, dict) and self._runtime_control_failed(result):
            if send_json is not None:
                await self._send_runtime_control_error(
                    send_json,
                    "runtime_close_failed",
                    "Duplex runtime close failed",
                    result,
                    session=session,
                )
            return False
        await self._send_runtime_control_if_needed(send_json, result, session=session)
        return True

    @staticmethod
    def _runtime_control_timeout_s(session: DuplexSession) -> float:
        raw = session.config.extra_body.get("duplex_control_timeout_s") or session.config.extra_body.get(
            "runtime_control_timeout_s"
        )
        if isinstance(raw, int | float) and raw > 0:
            return float(raw)
        if session.capabilities.implementation_level == "model_native_duplex":
            return 60.0
        return 10.0

    async def _send_runtime_control_if_needed(
        self,
        send_json,
        result: object,
        *,
        session: DuplexSession,
    ) -> None:
        if send_json is None:
            return
        if not isinstance(result, dict):
            return
        if not result.get("unsupported_count") and not result.get("error_count") and not result.get("passive_count"):
            return
        await send_json(
            {
                "type": "runtime.control",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "result": self._redact_runtime_control_result(result),
            }
        )

    @staticmethod
    def _runtime_control_failed(result: dict[str, object]) -> bool:
        if result.get("ok") is False:
            return True
        for key in ("unsupported_count", "error_count"):
            value = result.get(key)
            if isinstance(value, int | float) and value > 0:
                return True
        return False

    @classmethod
    def _runtime_signal_failed(cls, result: dict[str, object]) -> bool:
        error_count = result.get("error_count")
        if isinstance(error_count, int | float) and error_count > 0:
            return True
        if result.get("ok") is not False:
            return False
        return not cls._runtime_signal_has_data_plane_ack(result)

    @classmethod
    def _runtime_signal_has_data_plane_ack(cls, value: object) -> bool:
        if isinstance(value, dict):
            if value.get("data_plane_signal") is True and value.get("supported") is True:
                return True
            return any(cls._runtime_signal_has_data_plane_ack(child) for child in value.values())
        if isinstance(value, list | tuple):
            return any(cls._runtime_signal_has_data_plane_ack(item) for item in value)
        return False

    @classmethod
    def _redact_runtime_control_result(cls, value: object) -> object:
        if isinstance(value, dict):
            redacted = {
                key: cls._redact_runtime_control_result(child)
                for key, child in value.items()
                if key
                not in {
                    "stage_handoff",
                    "tts_handoff",
                    "omni_payload",
                    "tts_hidden_states",
                    "tts_token_ids",
                    "traceback",
                    "experimental_worker_control_rpc",
                    "experimental_eager_decoder",
                }
            }
            native_result = redacted.get("native_result")
            if isinstance(native_result, dict):
                if native_result.get("requires_stage_handoff") is True:
                    native_result.pop("requires_stage_handoff", None)
                if native_result.get("requires_tts_stage") is True:
                    native_result.pop("requires_tts_stage", None)
            return redacted
        if isinstance(value, list | tuple):
            return [cls._redact_runtime_control_result(item) for item in value]
        return value

    async def _send_native_duplex_events(
        self,
        send_json,
        result: object,
        *,
        session: DuplexSession,
        expected_epoch: int | None = None,
    ) -> tuple[str | None, bool]:
        if send_json is None:
            return None, False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return None, False
        close_reason: str | None = None
        emitted_response = False
        request_id, _ = self._data_plane_request_info(result)
        if request_id is not None and request_id in self._data_plane_terminal_request_ids:
            if os.environ.get("MINICPMO45_PROFILE_LOGS") == "1":
                logger.info("native data-plane output ignored after terminal request_id=%s", request_id)
            return None, False
        if request_id is not None and session.active_request_id is None:
            session.active_request_id = request_id
        for native_result in self._data_plane_native_results(result, session=session):
            close_reason_for_result, did_emit = await self._send_one_native_duplex_event(
                send_json,
                native_result,
                session=session,
                expected_epoch=expected_epoch,
            )
            emitted_response = emitted_response or did_emit
            close_reason = close_reason or close_reason_for_result
            if expected_epoch is not None and session.epoch != expected_epoch:
                return None, emitted_response
        for native_result in self._iter_native_duplex_results(result):
            close_reason_for_result, did_emit = await self._send_one_native_duplex_event(
                send_json,
                native_result,
                session=session,
                expected_epoch=expected_epoch,
            )
            emitted_response = emitted_response or did_emit
            close_reason = close_reason or close_reason_for_result
            if expected_epoch is not None and session.epoch != expected_epoch:
                return None, emitted_response
        return close_reason, emitted_response

    async def _drain_native_data_plane_stream(
        self,
        send_json,
        result: object,
        *,
        session: DuplexSession,
        expected_epoch: int | None = None,
    ) -> str | None:
        request_id, response_stage_id = self._data_plane_request_info(result)
        if request_id is None:
            return None
        if self._data_plane_outputs_finished(result):
            return None
        collect_outputs = getattr(
            self._chat_service.engine_client,
            "collect_duplex_data_plane_outputs_async",
            None,
        )
        if not callable(collect_outputs):
            return None

        profile_logs = os.environ.get("MINICPMO45_PROFILE_LOGS") == "1"
        close_reason: str | None = None
        empty_polls = 0
        while close_reason is None:
            if request_id in self._data_plane_terminal_request_ids:
                if profile_logs:
                    logger.info("[drain] exit terminal data-plane request: request_id=%s", request_id)
                return None
            if self._session_auto_responds(session):
                active_request_id = session.active_request_id
                if active_request_id is not None and active_request_id != request_id:
                    if profile_logs:
                        logger.info(
                            "[drain] exit inactive auto-response request: request_id=%s active_request_id=%s",
                            request_id,
                            active_request_id,
                        )
                    return None
            if expected_epoch is not None and session.epoch != expected_epoch:
                if profile_logs:
                    logger.info("[drain] exit epoch-change: request_id=%s", request_id)
                return None
            if session.state == DuplexSessionState.CLOSED:
                if profile_logs:
                    logger.info("[drain] exit session-closed: request_id=%s", request_id)
                return None
            outputs = await collect_outputs(
                request_id,
                response_stage_id=response_stage_id,
                timeout=self._runtime_control_timeout_s(session),
            )
            if profile_logs:
                logger.info(
                    "[drain] collect: request_id=%s n_outputs=%d finished=%s",
                    request_id,
                    len(outputs) if outputs else 0,
                    bool(outputs) and bool(getattr(outputs[-1], "finished", False)),
                )
            if expected_epoch is not None and session.epoch != expected_epoch:
                return None
            if not outputs:
                # An empty poll means no output arrived within one control
                # window, not that the stream is over. Exiting here orphans a
                # decision that lands moments later: it would sit queued until
                # the NEXT append starts a fresh drain task, adding one full
                # chunk of latency to every model decision.
                empty_polls += 1
                if empty_polls >= 3:
                    return None
                continue
            empty_polls = 0
            drain_result = {
                "data_plane_outputs": outputs,
            }
            close_reason, emitted_response = await self._send_native_duplex_events(
                send_json,
                drain_result,
                session=session,
                expected_epoch=expected_epoch,
            )
            if (
                self._data_plane_outputs_finished(drain_result)
                and emitted_response
                and not self._session_auto_responds(session)
            ):
                # Auto-respond sessions are resumable: every segment ends
                # with finished=True but the stream continues with the next
                # audio chunk. Exiting here would make each decision wait for
                # the next append to start a fresh drain task (one full chunk
                # of added latency); keep draining until epoch change,
                # session close, or cancellation by a newer drain task.
                return close_reason
            # A batch without a client-visible event (e.g. the generic
            # stage-0 final-output message that precedes the listen/speak
            # decision in the same segment) must not terminate the drain;
            # the real decision is still in flight.
        return close_reason

    @staticmethod
    def _data_plane_outputs_finished(result: object) -> bool:
        if not isinstance(result, dict):
            return False
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list) or not outputs:
            return False
        return bool(getattr(outputs[-1], "finished", False))

    @staticmethod
    def _data_plane_request_info(result: object) -> tuple[str | None, int | None]:
        if not isinstance(result, dict):
            return None, None
        return duplex_data_plane_request_info(result)

    async def _send_one_native_duplex_event(
        self,
        send_json,
        native_result: dict[str, object],
        *,
        session: DuplexSession,
        expected_epoch: int | None = None,
    ) -> tuple[str | None, bool]:
        close_reason: str | None = None
        emitted_response = False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return close_reason, emitted_response
        if native_result.get("passive_stage") is True:
            return close_reason, emitted_response
        data_plane_request_id = native_result.get("data_plane_request_id")
        if isinstance(data_plane_request_id, str) and data_plane_request_id in self._data_plane_terminal_request_ids:
            if os.environ.get("MINICPMO45_PROFILE_LOGS") == "1":
                logger.info("native event ignored after terminal data-plane request_id=%s", data_plane_request_id)
            return close_reason, emitted_response
        active_request_matches = session.active_request_id == data_plane_request_id or (
            self._session_auto_responds(session) and session.active_request_id is None
        )
        if isinstance(data_plane_request_id, str) and not active_request_matches:
            if os.environ.get("MINICPMO45_PROFILE_LOGS") == "1":
                logger.info(
                    "native event ignored: stale data-plane request_id=%s active_request_id=%s",
                    data_plane_request_id,
                    session.active_request_id,
                )
            return close_reason, emitted_response
        kv_cache_length = native_result.get("kv_cache_length")
        if isinstance(native_result.get("error_code"), str):
            response_id = session.active_response_id
            await send_json(
                {
                    "type": "error",
                    "code": native_result.get("error_code"),
                    "session_id": session.session_id,
                    "epoch": session.epoch,
                    "error": str(native_result.get("error") or "Duplex native data-plane error"),
                }
            )
            if response_id is not None:
                session.end_response(commit_text=False)
                await send_json(
                    {
                        "type": "response.done",
                        "session_id": session.session_id,
                        "response_id": response_id,
                        "epoch": session.epoch,
                        "committed": False,
                        "status": "failed",
                        "status_details": {
                            "type": "failed",
                            "reason": native_result.get("error_code"),
                        },
                        "playback": session.playback.as_dict(),
                    }
                )
            return close_reason, True
        if self._native_result_missing_stage_role(session, native_result):
            await send_json(
                {
                    "type": "error",
                    "code": "runtime_native_stage_role_required",
                    "session_id": session.session_id,
                    "epoch": session.epoch,
                    "error": "MiniCPM-o native duplex results must include explicit stage_role.",
                }
            )
            return close_reason, True
        if self._native_result_requires_runner_kv(session, native_result):
            await send_json(
                {
                    "type": "error",
                    "code": "runtime_native_runner_kv_required",
                    "session_id": session.session_id,
                    "epoch": session.epoch,
                    "error": ("MiniCPM-o native duplex Stage0 output must be scheduler/KV-backed by the model runner."),
                    "vllm_omni": {
                        "runtime_impl": native_result.get("runtime_impl", ""),
                    },
                }
            )
            return close_reason, True
        if native_result.get("requires_stage_handoff") is True or native_result.get("requires_tts_stage") is True:
            # Stage0 announces a talker handoff before Stage1 has produced
            # client-visible audio. A single client input stream may contain
            # several model turns, including terminal-only turns. Reserve the
            # protocol response only when Stage1 emits text/audio so empty
            # model turns cannot create empty responses or steal later audio.
            return close_reason, False
        is_listen = native_result.get("is_listen")
        if native_result.get("is_buffering") is True or native_result.get("prefill_success") is False:
            if native_result.get("data_plane_request_id") == session.active_request_id:
                session.active_request_id = None
            payload = {
                "type": "response.listen",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "kv_cache_length": kv_cache_length,
                "reason": native_result.get("reason") or "buffering",
                "model_listen": False,
                "buffering": True,
            }
            self._attach_native_runtime_metadata(payload, native_result)
            await send_json(payload)
            return close_reason, emitted_response
        if is_listen is True:
            non_terminal_auto_listen = (
                self._session_auto_responds(session)
                and session.active_response_id is not None
                and session.active_request_id is not None
                and native_result.get("end_of_turn") is not True
            )
            if non_terminal_auto_listen:
                self._maybe_continue_native_response(
                    send_json,
                    session=session,
                    expected_epoch=expected_epoch,
                )
                return close_reason, emitted_response
            session.turn_state = DuplexTurnState.IDLE
            auto_response = self._session_auto_responds(session)
            if not auto_response and data_plane_request_id == session.active_request_id:
                session.active_request_id = None
            model_listen = native_result.get("model_listen")
            if not isinstance(model_listen, bool):
                model_listen = native_result.get("reason") in {None, "", "model_listen"}
            response_id = session.active_response_id
            if isinstance(data_plane_request_id, str):
                self._data_plane_terminal_request_ids.add(data_plane_request_id)
            emitted_response = True
            payload = {
                "type": "response.listen",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "kv_cache_length": kv_cache_length,
                "reason": native_result.get("reason") or "model_listen",
                "model_listen": model_listen,
            }
            self._attach_native_runtime_metadata(payload, native_result)
            if os.environ.get("MINICPMO45_PROFILE_LOGS") == "1":
                logger.info("native event sent: response.listen t=%.3f", time.monotonic())
            await send_json(payload)
            if native_result.get("abort_data_plane_request") is True and isinstance(data_plane_request_id, str):
                await self._abort_request_background(
                    session,
                    data_plane_request_id,
                    send_json,
                    notify=False,
                )
            if response_id is not None:
                if not self._session_auto_responds(session) and self._native_response_continuations_remaining(
                    session, response_id
                ):
                    # The official model often listens for a silence beat or
                    # two before answering; keep the response open and give it
                    # the next silence unit as a decision point.
                    self._maybe_continue_native_response(
                        send_json,
                        session=session,
                        expected_epoch=expected_epoch,
                    )
                    return close_reason, emitted_response
                session.end_response(commit_text=False, preserve_request=auto_response)
                if auto_response:
                    self._mark_auto_response_waiting_for_speech(
                        session,
                        prefix_variant=MiniCPMO45DuplexPolicy.NEW_USER_TURN_PREFIX_CLEAN_RESPONSE_DONE,
                    )
                await send_json(
                    {
                        "type": "response.done",
                        "session_id": session.session_id,
                        "response_id": response_id,
                        "epoch": session.epoch,
                        "committed": False,
                        "playback": session.playback.as_dict(),
                    }
                )
            if self._is_native_context_full(session, kv_cache_length):
                close_reason = "context_full"
            return close_reason, emitted_response

        text = native_result.get("text")
        audio = native_result.get("audio_data", native_result.get("audio"))
        end_of_turn = bool(native_result.get("end_of_turn", False))
        has_text = isinstance(text, str) and bool(text)
        has_audio = isinstance(audio, str) and bool(audio)
        if not has_text and not has_audio and not end_of_turn:
            tts_segment_ended = (
                native_result.get("stage_role") == "tts" and native_result.get("abort_data_plane_request") is True
            )
            if tts_segment_ended and self._session_auto_responds(session):
                self._maybe_continue_native_response(
                    send_json,
                    session=session,
                    expected_epoch=expected_epoch,
                )
            return close_reason, emitted_response
        if end_of_turn and not has_text and not has_audio and session.active_response_id is None:
            if isinstance(data_plane_request_id, str):
                if not self._session_auto_responds(session) and data_plane_request_id == session.active_request_id:
                    session.active_request_id = None
                if not self._session_auto_responds(session):
                    self._data_plane_terminal_request_ids.add(data_plane_request_id)
            model_turn_id = self._coerce_data_plane_int(native_result.get("model_turn_id"))
            if model_turn_id is not None:
                session.complete_model_turn(model_turn_id)
            return close_reason, emitted_response
        emitted_response = True
        response_created = False
        response_id = session.active_response_id
        if response_id is None:
            model_turn_id = self._coerce_data_plane_int(native_result.get("model_turn_id"))
            response_id = session.begin_response(turn_id=model_turn_id)
            self._remember_response_conversation_mode(session, response_id)
            response_created = True
            await send_json(
                self._response_created_payload(
                    session,
                    response_id,
                    epoch=session.epoch,
                )
            )
            speak_payload = {
                "type": "response.speak",
                "session_id": session.session_id,
                "response_id": response_id,
                "epoch": session.epoch,
                "text": text if isinstance(text, str) else "",
                "end_of_turn": end_of_turn,
                "model_speak": True,
                "kv_cache_length": kv_cache_length,
            }
            self._attach_native_runtime_metadata(speak_payload, native_result)
            await send_json(speak_payload)
        previous_sent_ms = session.playback.sent_ms
        text_chars_before_append = len("".join(session.assistant_text_buffer))
        if isinstance(text, str):
            session.append_assistant_text(text)
        duration_ms = native_result.get("audio_duration_ms")
        text_chars = len("".join(session.assistant_text_buffer))
        mark_duration_ms = None
        mark_text_chars: int | None = text_chars
        if native_result.get("audio_text_mark") is False:
            mark_text_chars = None
        if isinstance(duration_ms, int | float):
            mark_duration_ms = int(duration_ms)
            if native_result.get("audio_duration_is_cumulative") is not True:
                mark_duration_ms += session.playback.sent_ms
        audio_text_marks = native_result.get("audio_text_marks")
        audio_text_marks = self._normalize_native_audio_text_marks(
            audio_text_marks if isinstance(audio_text_marks, list) else None,
            audio_offset_ms=(
                0
                if native_result.get("audio_text_marks_are_cumulative") is True
                or native_result.get("audio_duration_is_cumulative") is True
                else previous_sent_ms
            ),
            text_offset_chars=(
                0 if native_result.get("audio_text_marks_are_cumulative") is True else text_chars_before_append
            ),
        )
        session.mark_audio_sent(
            mark_duration_ms,
            text_chars=mark_text_chars if mark_duration_ms is not None else None,
            audio_text_marks=audio_text_marks,
        )
        payload = {
            "type": "response.output_audio.delta",
            "session_id": session.session_id,
            "response_id": response_id,
            "epoch": session.epoch,
            "text": text if isinstance(text, str) else "",
            "audio": audio if isinstance(audio, str) else "",
            "format": (
                native_result.get("audio_format")
                if isinstance(native_result.get("audio_format"), str)
                else session.config.response_format
            ),
            "end_of_turn": end_of_turn,
            "model_speak": True,
            "kv_cache_length": kv_cache_length,
        }
        if mark_duration_ms is not None:
            payload["audio_duration_ms"] = mark_duration_ms
        if audio_text_marks:
            payload["audio_text_marks"] = audio_text_marks
        elif mark_duration_ms is not None and mark_text_chars is not None:
            payload["audio_text_marks"] = [
                {
                    "text_chars": max(0, int(mark_text_chars)),
                    "audio_end_ms": max(0, int(mark_duration_ms)),
                }
            ]
        payload["playback"] = session.playback.as_dict()
        sample_rate_hz = native_result.get("sample_rate_hz") or native_result.get("audio_sample_rate_hz")
        if isinstance(sample_rate_hz, int | float) and int(sample_rate_hz) > 0:
            payload["sample_rate_hz"] = int(sample_rate_hz)
        self._attach_native_runtime_metadata(payload, native_result)
        if os.environ.get("MINICPMO45_PROFILE_LOGS") == "1":
            logger.info("native event sent: audio.delta t=%.3f", time.monotonic())
        await send_json(payload)
        if (
            not end_of_turn
            and native_result.get("stage_role") == "tts"
            and native_result.get("abort_data_plane_request") is True
            and self._session_auto_responds(session)
        ):
            self._maybe_continue_native_response(
                send_json,
                session=session,
                expected_epoch=expected_epoch,
            )
        if end_of_turn:
            data_plane_request_id = native_result.get("data_plane_request_id")
            if isinstance(data_plane_request_id, str) and not self._session_auto_responds(session):
                self._data_plane_audio_offsets.pop(data_plane_request_id, None)
                self._data_plane_sent_segment_texts.pop(data_plane_request_id, None)
            if isinstance(data_plane_request_id, str):
                if not self._session_auto_responds(session) and data_plane_request_id == session.active_request_id:
                    session.active_request_id = None
                if not self._session_auto_responds(session):
                    self._data_plane_terminal_request_ids.add(data_plane_request_id)
            should_commit = self._should_commit_response_to_history(response_id)
            committed_message = session.end_response(
                commit_text=should_commit,
                preserve_request=self._session_auto_responds(session),
            )
            model_turn_id = self._coerce_data_plane_int(native_result.get("model_turn_id"))
            if model_turn_id is not None:
                session.complete_model_turn(model_turn_id)
            if self._session_auto_responds(session):
                self._mark_auto_response_waiting_for_speech(
                    session,
                    prefix_variant=MiniCPMO45DuplexPolicy.NEW_USER_TURN_PREFIX_CLEAN_RESPONSE_DONE,
                )
            if should_commit:
                session.register_history_item(f"item_{response_id}", committed_message)
            await send_json(
                {
                    "type": "response.done",
                    "session_id": session.session_id,
                    "response_id": response_id,
                    "epoch": session.epoch,
                    "committed": committed_message is not None,
                    "playback": session.playback.as_dict(),
                }
            )
        elif response_created:
            session.turn_state = DuplexTurnState.ASSISTANT_PLAYING
        if self._is_native_context_full(session, kv_cache_length):
            close_reason = "context_full"
        return close_reason, emitted_response

    @staticmethod
    def _normalize_native_audio_text_marks(
        audio_text_marks: list[object] | None,
        *,
        audio_offset_ms: int,
        text_offset_chars: int,
    ) -> list[dict[str, int]] | None:
        if not audio_text_marks:
            return None
        normalized: list[dict[str, int]] = []
        for raw_mark in audio_text_marks:
            if not isinstance(raw_mark, dict):
                continue
            raw_text_chars = raw_mark.get("text_chars")
            raw_audio_end_ms = raw_mark.get("audio_end_ms", raw_mark.get("audio_ms"))
            if not isinstance(raw_text_chars, int | float) or not isinstance(raw_audio_end_ms, int | float):
                continue
            normalized.append(
                {
                    "text_chars": max(0, int(raw_text_chars) + int(text_offset_chars)),
                    "audio_end_ms": max(0, int(raw_audio_end_ms) + int(audio_offset_ms)),
                }
            )
        return normalized or None

    def _data_plane_native_results(self, result: object, *, session: DuplexSession | None = None):
        if not isinstance(result, dict):
            return
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list):
            return
        for output in outputs:
            yield from self._native_results_from_data_plane_output(output, session=session)

    def _native_results_from_data_plane_output(self, output: object, *, session: DuplexSession | None = None):
        data_plane_request_id = getattr(output, "request_id", None)
        if not isinstance(data_plane_request_id, str) or not data_plane_request_id:
            data_plane_request_id = None
        outputs = getattr(output, "outputs", None)
        completion = outputs[0] if isinstance(outputs, list) and outputs else None
        text = getattr(completion, "text", "") if completion is not None else ""
        mm_output = getattr(output, "multimodal_output", None)
        if not isinstance(mm_output, Mapping):
            mm_output = getattr(completion, "multimodal_output", {}) if completion is not None else {}
        if not mm_output:
            inner_output = getattr(output, "request_output", None)
            if inner_output is not None and inner_output is not output:
                inner_mm_output = getattr(inner_output, "multimodal_output", None)
                if isinstance(inner_mm_output, Mapping) and inner_mm_output:
                    mm_output = inner_mm_output
                else:
                    inner_outputs = getattr(inner_output, "outputs", None)
                    inner_completion = inner_outputs[0] if isinstance(inner_outputs, list) and inner_outputs else None
                    inner_completion_mm_output = (
                        getattr(inner_completion, "multimodal_output", None) if inner_completion is not None else None
                    )
                    if completion is None and inner_completion is not None:
                        completion = inner_completion
                        text = getattr(inner_completion, "text", "") or text
                    if isinstance(inner_completion_mm_output, Mapping):
                        mm_output = inner_completion_mm_output
        if isinstance(mm_output, Mapping):
            mm_output = dict(mm_output)
        else:
            mm_output = {}
        output_turn_id = self._data_plane_output_turn_id(mm_output)
        output_epoch = self._data_plane_output_epoch(mm_output)
        stale_turn = False
        stale_epoch = False
        if session is not None:
            expected_turn_id = session.active_response_turn_id
            stale_turn = (
                expected_turn_id is not None and output_turn_id is not None and output_turn_id != expected_turn_id
            )
            if expected_turn_id is None and output_turn_id is not None:
                stale_turn = output_turn_id < session.turn_id
            stale_epoch = output_epoch is not None and output_epoch != session.epoch
        if session is not None and self._session_auto_responds(session) and (stale_turn or stale_epoch):
            if os.environ.get("MINICPMO45_PROFILE_LOGS") == "1":
                logger.info(
                    "drop stale duplex data-plane output: request_id=%s output_turn_id=%s "
                    "session_turn_id=%s active_response_turn_id=%s output_epoch=%s session_epoch=%s",
                    data_plane_request_id,
                    output_turn_id,
                    session.turn_id,
                    session.active_response_turn_id,
                    output_epoch,
                    session.epoch,
                )
            return
        mm_text = self._data_plane_llm_output_text(mm_output)
        if mm_text:
            # Stage-1 audio outputs can expose cumulative completion.text while
            # carrying the exact text for this audio segment in metadata.
            text = mm_text
            if data_plane_request_id is not None:
                self._data_plane_segment_text_metadata_request_ids.add(data_plane_request_id)
        elif (
            data_plane_request_id is not None
            and data_plane_request_id in self._data_plane_segment_text_metadata_request_ids
        ):
            # Once a data-plane request has explicit segment text metadata,
            # keep transcript sourcing there. TTS-stage completion.text is
            # cumulative and will duplicate transcripts if mixed back in.
            text = ""
        profile_logs = os.environ.get("MINICPMO45_PROFILE_LOGS") == "1"
        finished = bool(getattr(output, "finished", False))
        tts_is_last_chunk = self._data_plane_bool_metadata(
            mm_output,
            ("tts_is_last_chunk",),
            default=False,
        )
        token_ids = self._data_plane_completion_token_ids(completion)
        native_decision = self._data_plane_native_decision(
            completion,
            mm_output,
            token_ids=token_ids,
            finished=finished,
        )
        if native_decision == "listen":
            # A model-listen wrapper carries the thinker's hidden states under
            # "latent", which the audio encoder's key fallback would treat as
            # a waveform: encoding it ratchets the cumulative audio offset
            # (tokens x hidden_dim samples) past the talker's real audio, and
            # every later speak unit slices to empty and is dropped. Yield the
            # listen before any audio work so the offset is never touched.
            if profile_logs:
                logger.info(
                    "duplex data-plane output: request_id=%s finished=%s "
                    "native_decision=listen mm_keys=%s (audio encode skipped)",
                    getattr(output, "request_id", None),
                    finished,
                    sorted(mm_output.keys()),
                )
            yield {
                "supported": True,
                "stage_role": "llm",
                "is_listen": True,
                "model_listen": True,
                "listen_source": "model_listen",
                "data_plane_request_id": data_plane_request_id,
                "end_of_turn": False,
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
                "runtime_impl": "scheduler_data_plane",
                "owned_runtime": False,
                "experimental_worker_control_rpc": False,
                "runner_local_payload_ref": False,
            }
            return
        # Stage 1 exposes a cumulative waveform for the lifetime of its
        # resumable scheduler request, including across model-owned turns.
        # Keep the audio cursor request-scoped; keying it by turn replays the
        # complete previous waveform whenever the turn id advances.
        audio_cursor_key = data_plane_request_id
        turn_cursor_key = self._data_plane_text_cursor_key(data_plane_request_id, output_turn_id)
        raw_audio = next((mm_output[k] for k in ("audio", "model_outputs", "latent") if k in mm_output), None)
        raw_audio_samples = self._audio_num_samples(raw_audio)
        offset_before = self._data_plane_audio_offsets.get(audio_cursor_key) if audio_cursor_key is not None else None
        audio_chunks = list(
            self._encode_data_plane_audio_chunks_with_duration(
                mm_output,
                request_id=audio_cursor_key,
                response_format=(session.config.response_format if session is not None else "wav"),
                speed=(session.config.speed if session is not None else None),
            )
        )
        stage_turn_end = self._data_plane_bool_metadata(
            mm_output,
            ("turn_end", "end_of_turn"),
            default=False,
        )
        # Terminal de-duplication is turn-scoped even though cumulative audio
        # slicing is request-scoped.
        tts_eos_key = turn_cursor_key or data_plane_request_id or ""
        stage_tts_eos = (
            session is not None
            and self._session_auto_responds(session)
            and 151645 in token_ids
            and not audio_chunks
            and raw_audio_samples is not None
            and (raw_audio_samples == 0 or raw_audio_samples == offset_before)
            and tts_eos_key not in self._data_plane_tts_eos_done
        )
        tts_segment_end = bool(tts_is_last_chunk or stage_tts_eos) and (
            not tts_eos_key or tts_eos_key not in self._data_plane_tts_eos_done
        )
        if tts_segment_end and tts_eos_key:
            self._data_plane_tts_eos_done.add(tts_eos_key)
        stage_turn_end_new = bool(stage_turn_end) and (
            not tts_eos_key or tts_eos_key not in self._data_plane_turn_eos_done
        )
        if stage_turn_end_new and tts_eos_key:
            self._data_plane_turn_eos_done.add(tts_eos_key)
        unit_end_of_turn = stage_turn_end_new or (
            finished and not (session is not None and self._session_auto_responds(session))
        )
        if profile_logs:
            logger.info(
                "duplex data-plane output: request_id=%s finished=%s "
                "text_len=%d audio_chunks=%d native_decision=%s mm_keys=%s "
                "raw_audio_samples=%s offset_before=%s offset_after=%s "
                "token_ids=%s tts_is_last_chunk=%s stage_tts_eos=%s "
                "stage_turn_end=%s stage_turn_end_new=%s tts_segment_end=%s",
                getattr(output, "request_id", None),
                finished,
                len(text) if isinstance(text, str) else 0,
                len(audio_chunks),
                native_decision,
                sorted(mm_output.keys()),
                raw_audio_samples,
                offset_before,
                (self._data_plane_audio_offsets.get(audio_cursor_key) if audio_cursor_key is not None else None),
                token_ids,
                tts_is_last_chunk,
                stage_tts_eos,
                stage_turn_end,
                stage_turn_end_new,
                tts_segment_end,
            )
        if audio_chunks:
            # The talker streams several cumulative-audio batches per handed
            # text, INCLUDING continuation units that re-run it with the
            # same text past finished=True boundaries (every engine segment
            # ends finished); official results carry per-unit deltas, so
            # attach only text not yet delivered, comparing by content.
            # Deliberately no reset at segment or response boundaries: any
            # reset re-attaches the text on the next same-text continuation
            # batch and duplicates the transcript.
            delta_text = self._data_plane_segment_text_delta(
                data_plane_request_id,
                text,
                turn_id=(
                    output_turn_id if output_turn_id is not None else (session.turn_id if session is not None else None)
                ),
            )
            last_idx = len(audio_chunks) - 1
            sample_rate_hz = self._data_plane_sample_rate_hz(mm_output)
            audio_text_marks = self._data_plane_audio_text_marks(mm_output)
            fallback_audio_text_marks: list[list[dict[str, int]] | None] = []
            if not audio_text_marks and delta_text:
                total_duration_ms = sum(max(0, int(duration_ms)) for _, duration_ms in audio_chunks)
                cumulative_duration_ms = 0
                for _, duration_ms in audio_chunks:
                    cumulative_duration_ms += max(0, int(duration_ms))
                    if total_duration_ms <= 0:
                        fallback_audio_text_marks.append(None)
                        continue
                    text_chars = int(
                        len(delta_text) * max(0.0, min(1.0, cumulative_duration_ms / float(total_duration_ms)))
                    )
                    fallback_audio_text_marks.append(
                        [
                            {
                                "text_chars": max(0, text_chars),
                                "audio_end_ms": max(0, cumulative_duration_ms),
                            }
                        ]
                    )
            audio_results: list[dict[str, object]] = []
            for idx, (audio, duration_ms) in enumerate(audio_chunks):
                native_result = {
                    "supported": True,
                    "stage_role": "tts",
                    "is_listen": False,
                    "data_plane_request_id": data_plane_request_id,
                    "text": delta_text if idx == 0 else "",
                    "audio_data": audio,
                    "audio_format": session.config.response_format if session is not None else "wav",
                    "audio_duration_ms": duration_ms,
                    "audio_text_mark": idx == last_idx,
                    "sample_rate_hz": sample_rate_hz,
                    "end_of_turn": unit_end_of_turn and idx == last_idx,
                    "abort_data_plane_request": tts_segment_end and idx == last_idx,
                    "uses_model_runner_scheduler": True,
                    "runner_kv_backed": True,
                    "runtime_impl": "scheduler_data_plane",
                    "owned_runtime": False,
                    "experimental_worker_control_rpc": False,
                    "runner_local_payload_ref": False,
                }
                if output_turn_id is not None:
                    native_result["model_turn_id"] = output_turn_id
                if audio_text_marks and idx == last_idx:
                    native_result["audio_text_marks"] = audio_text_marks
                    native_result["audio_text_marks_are_cumulative"] = True
                elif idx < len(fallback_audio_text_marks) and fallback_audio_text_marks[idx]:
                    native_result["audio_text_marks"] = fallback_audio_text_marks[idx]
                    native_result["audio_text_marks_are_cumulative"] = True
                audio_results.append(native_result)

            pending_audio_key = turn_cursor_key or data_plane_request_id
            if session is not None and self._session_auto_responds(session):
                if delta_text and pending_audio_key is not None:
                    self._data_plane_turns_with_text.add(pending_audio_key)
                response_turn_bound = session.active_response_id is not None or (
                    session.active_response_turn_id is not None
                    and output_turn_id is not None
                    and session.active_response_turn_id == output_turn_id
                )
                turn_has_text = pending_audio_key is not None and pending_audio_key in self._data_plane_turns_with_text
                if not response_turn_bound and not turn_has_text:
                    if pending_audio_key is not None:
                        if unit_end_of_turn:
                            self._data_plane_pending_audio_without_text.pop(pending_audio_key, None)
                        else:
                            self._data_plane_pending_audio_without_text.setdefault(pending_audio_key, []).extend(
                                audio_results
                            )
                    if unit_end_of_turn:
                        terminal_result = dict(audio_results[-1])
                        terminal_result.update(
                            {
                                "audio_data": "",
                                "audio_duration_ms": 0,
                                "audio_text_mark": False,
                                "end_of_turn": True,
                            }
                        )
                        yield terminal_result
                    return
                if pending_audio_key is not None:
                    yield from self._data_plane_pending_audio_without_text.pop(pending_audio_key, [])
            yield from audio_results
            return
        if tts_segment_end:
            pending_audio_key = turn_cursor_key or data_plane_request_id
            if unit_end_of_turn and pending_audio_key is not None:
                self._data_plane_pending_audio_without_text.pop(pending_audio_key, None)
            terminal_result = {
                "supported": True,
                "stage_role": "tts",
                "is_listen": False,
                "data_plane_request_id": data_plane_request_id,
                "text": "",
                "audio_data": "",
                "audio_format": session.config.response_format if session is not None else "wav",
                "audio_text_mark": False,
                "end_of_turn": unit_end_of_turn,
                "abort_data_plane_request": True,
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
                "runtime_impl": "scheduler_data_plane",
                "owned_runtime": False,
                "experimental_worker_control_rpc": False,
                "runner_local_payload_ref": False,
            }
            if output_turn_id is not None:
                terminal_result["model_turn_id"] = output_turn_id
            yield terminal_result
            return
        if session is not None and session.active_response_id is not None and unit_end_of_turn:
            terminal_result = {
                "supported": True,
                "stage_role": "tts",
                "is_listen": False,
                "data_plane_request_id": data_plane_request_id,
                "text": "",
                "audio_data": "",
                "audio_format": session.config.response_format,
                "audio_text_mark": False,
                "end_of_turn": True,
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
                "runtime_impl": "scheduler_data_plane",
                "owned_runtime": False,
                "experimental_worker_control_rpc": False,
                "runner_local_payload_ref": False,
            }
            if output_turn_id is not None:
                terminal_result["model_turn_id"] = output_turn_id
            yield terminal_result
            return
        if data_plane_request_id is not None and session is not None and self._session_auto_responds(session):
            # Text-only flushes are not client-visible transcript segments.
            # Keep the audio-bound cursor for the next batch that carries audio.
            if unit_end_of_turn and session.active_response_id is None:
                pending_audio_key = turn_cursor_key or data_plane_request_id
                if pending_audio_key is not None:
                    self._data_plane_pending_audio_without_text.pop(pending_audio_key, None)
                self._mark_auto_response_waiting_for_speech(
                    session,
                    prefix_variant=MiniCPMO45DuplexPolicy.NEW_USER_TURN_PREFIX_LISTEN_ONLY,
                )
                return
        if (
            finished
            and session is not None
            and self._session_auto_responds(session)
            and session.active_response_id is not None
            and data_plane_request_id is not None
            and not unit_end_of_turn
        ):
            yield {
                "supported": True,
                "stage_role": "llm",
                "is_listen": True,
                "model_listen": False,
                "listen_source": "auto_response_segment_complete",
                "reason": "auto_response_segment_complete",
                "data_plane_request_id": data_plane_request_id,
                "end_of_turn": False,
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
                "runtime_impl": "scheduler_data_plane",
                "owned_runtime": False,
                "experimental_worker_control_rpc": False,
                "runner_local_payload_ref": False,
            }
            return
        if not text:
            if session is not None and self._session_auto_responds(session):
                return
            if finished:
                yield {
                    "supported": True,
                    "stage_role": "llm",
                    "is_listen": True,
                    "model_listen": False,
                    "listen_source": "data_plane_finished_without_output",
                    "reason": "data_plane_finished_without_output",
                    "data_plane_request_id": data_plane_request_id,
                    "end_of_turn": False,
                    "uses_model_runner_scheduler": True,
                    "runner_kv_backed": True,
                    "runtime_impl": "scheduler_data_plane",
                    "owned_runtime": False,
                    "experimental_worker_control_rpc": False,
                    "runner_local_payload_ref": False,
                }
            return
        if session is not None and self._session_auto_responds(session):
            return
        if session is not None and "audio" in session.config.modalities:
            yield {
                "supported": True,
                "stage_role": "tts",
                "error_code": "runtime_data_plane_text_without_audio",
                "error": "MiniCPM-o native duplex data-plane produced text without audio.",
                "data_plane_request_id": data_plane_request_id,
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
                "runtime_impl": "scheduler_data_plane",
                "owned_runtime": False,
                "experimental_worker_control_rpc": False,
                "runner_local_payload_ref": False,
            }
            return
        yield {
            "supported": True,
            "stage_role": "llm",
            "is_listen": False,
            "data_plane_request_id": data_plane_request_id,
            "text": text if isinstance(text, str) else "",
            "audio_data": "",
            "end_of_turn": unit_end_of_turn,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
            "owned_runtime": False,
            "experimental_worker_control_rpc": False,
            "runner_local_payload_ref": False,
        }

    @classmethod
    def _data_plane_native_decision(
        cls,
        completion: object,
        mm_output: dict[str, object],
        *,
        token_ids: list[int],
        finished: bool,
    ) -> str | None:
        if not finished:
            return None
        # The orchestrator stamps model-listen segments explicitly; honor the
        # marker before falling back to token heuristics (the wrapped listen
        # output does not expose completion token ids).
        if mm_output.get("duplex_native_decision") == "listen" or mm_output.get("model_listen") is True:
            return "listen"
        special = cls._data_plane_special_token_ids(mm_output)
        listen_id = special.get("listen_token_id")
        if listen_id is None:
            return None
        stop_reason = getattr(completion, "stop_reason", None) if completion is not None else None
        if cls._coerce_data_plane_int(stop_reason) == listen_id:
            return "listen"
        return "listen" if token_ids and token_ids[-1] == listen_id else None

    @classmethod
    def _data_plane_special_token_ids(cls, mm_output: dict[str, object]) -> dict[str, int]:
        out: dict[str, int] = {}
        sources: list[object] = []
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
                token_id = cls._coerce_data_plane_int(value)
                if token_id is not None and token_id >= 0:
                    out[key] = token_id
        return out

    @classmethod
    def _payload_turn_id(cls, payload: object) -> int | None:
        if not isinstance(payload, Mapping):
            return None
        return cls._coerce_data_plane_int(payload.get("duplex_turn_id"))

    @classmethod
    def _data_plane_output_turn_id(cls, mm_output: dict[str, object]) -> int | None:
        candidates: list[object] = [
            mm_output.get("duplex_turn_id"),
            mm_output.get("turn_id"),
            mm_output.get("meta.duplex_turn_id"),
            mm_output.get("meta.turn_id"),
        ]
        meta = mm_output.get("meta")
        if isinstance(meta, dict):
            candidates.extend((meta.get("duplex_turn_id"), meta.get("turn_id")))
        for value in candidates:
            turn_id = cls._coerce_data_plane_int(value)
            if turn_id is not None:
                return turn_id
        return None

    @classmethod
    def _data_plane_output_epoch(cls, mm_output: dict[str, object]) -> int | None:
        candidates: list[object] = [
            mm_output.get("duplex_epoch"),
            mm_output.get("epoch"),
            mm_output.get("meta.duplex_epoch"),
            mm_output.get("meta.epoch"),
        ]
        meta = mm_output.get("meta")
        if isinstance(meta, dict):
            candidates.extend((meta.get("duplex_epoch"), meta.get("epoch")))
        for value in candidates:
            epoch = cls._coerce_data_plane_int(value)
            if epoch is not None:
                return epoch
        return None

    @classmethod
    def _data_plane_completion_token_ids(cls, completion: object) -> list[int]:
        if completion is None:
            return []
        candidates = (
            getattr(completion, "token_ids", None),
            getattr(completion, "cumulative_token_ids", None),
        )
        for candidate in candidates:
            token_ids = cls._coerce_data_plane_int_list(candidate)
            if token_ids:
                return token_ids
        return []

    @classmethod
    def _coerce_data_plane_int_list(cls, value: object) -> list[int]:
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
            token_id = cls._coerce_data_plane_int(item)
            if token_id is not None:
                out.append(token_id)
        return out

    @staticmethod
    def _data_plane_text_cursor_key(request_id: str | None, turn_id: int | None = None) -> str | None:
        if request_id is None:
            return None
        if turn_id is None:
            return request_id
        return f"{request_id}:turn:{turn_id}"

    def _data_plane_segment_text_delta(
        self,
        request_id: str | None,
        text: object,
        *,
        turn_id: int | None = None,
    ) -> str:
        if not isinstance(text, str) or not text:
            return ""
        cursor_key = self._data_plane_text_cursor_key(request_id, turn_id)
        if cursor_key is None:
            return text
        sent_text = self._data_plane_sent_segment_texts.get(cursor_key)
        if not isinstance(sent_text, str) or not sent_text:
            delta_text = text
        elif text == sent_text:
            delta_text = ""
        elif text.startswith(sent_text):
            delta_text = text[len(sent_text) :]
        else:
            logger.warning(
                "Non-prefix duplex data-plane transcript for request_id=%s turn_id=%s; emitting full text",
                request_id,
                turn_id,
            )
            delta_text = text
        self._data_plane_sent_segment_texts[cursor_key] = text
        return delta_text

    @staticmethod
    def _data_plane_request_belongs_to_session(request_id: str, session_id: str) -> bool:
        return request_id.startswith(f"duplex-{session_id}-") or request_id.startswith(f"chatcmpl-duplex-{session_id}-")

    def _cleanup_data_plane_request_state(self, request_id: str) -> None:
        for cursor_key in list(self._data_plane_audio_offsets):
            if cursor_key == request_id or cursor_key.startswith(f"{request_id}:"):
                self._data_plane_audio_offsets.pop(cursor_key, None)
        for cursor_key in list(self._data_plane_sent_segment_texts):
            if cursor_key == request_id or cursor_key.startswith(f"{request_id}:"):
                self._data_plane_sent_segment_texts.pop(cursor_key, None)
        for cursor_key in list(self._data_plane_pending_audio_without_text):
            if cursor_key == request_id or cursor_key.startswith(f"{request_id}:"):
                self._data_plane_pending_audio_without_text.pop(cursor_key, None)
        for cursor_key in list(self._data_plane_turns_with_text):
            if cursor_key == request_id or cursor_key.startswith(f"{request_id}:"):
                self._data_plane_turns_with_text.discard(cursor_key)
        self._data_plane_segment_text_metadata_request_ids.discard(request_id)
        for terminal_key in list(self._data_plane_tts_eos_done):
            if terminal_key == request_id or terminal_key.startswith(f"{request_id}:"):
                self._data_plane_tts_eos_done.discard(terminal_key)
        for terminal_key in list(self._data_plane_turn_eos_done):
            if terminal_key == request_id or terminal_key.startswith(f"{request_id}:"):
                self._data_plane_turn_eos_done.discard(terminal_key)
        for terminal_key in list(self._data_plane_terminal_request_ids):
            if terminal_key == request_id or terminal_key.startswith(f"{request_id}:"):
                self._data_plane_terminal_request_ids.discard(terminal_key)

    def _cleanup_duplex_session_state(self, session: DuplexSession) -> None:
        session_id = session.session_id
        self._auto_response_waiting_for_speech.discard(session_id)
        self._auto_response_new_turn_prefix_variants.pop(session_id, None)
        self._native_response_continuations.pop(session_id, None)
        active_request_id = session.active_request_id
        if isinstance(active_request_id, str):
            self._cleanup_data_plane_request_state(active_request_id)
        for request_id in (
            set(self._data_plane_audio_offsets)
            | set(self._data_plane_sent_segment_texts)
            | set(self._data_plane_pending_audio_without_text)
            | set(self._data_plane_turns_with_text)
            | set(self._data_plane_segment_text_metadata_request_ids)
        ):
            if self._data_plane_request_belongs_to_session(request_id, session_id):
                self._cleanup_data_plane_request_state(request_id)
        for request_id in (
            set(self._data_plane_tts_eos_done)
            | set(self._data_plane_turn_eos_done)
            | set(self._data_plane_terminal_request_ids)
        ):
            if self._data_plane_request_belongs_to_session(request_id, session_id):
                self._cleanup_data_plane_request_state(request_id)

    @staticmethod
    def _coerce_data_plane_int(value: object) -> int | None:
        if hasattr(value, "detach"):
            try:
                value = value.detach().cpu().reshape(-1)
                if value.numel() == 0:
                    return None
                value = value[0].item()
            except Exception:
                return None
        elif hasattr(value, "reshape") and hasattr(value, "size"):
            try:
                value = value.reshape(-1)
                if int(value.size) == 0:
                    return None
                item = value[0]
                value = item.item() if hasattr(item, "item") else item
            except Exception:
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _encode_data_plane_audio_chunks(
        self,
        mm_output: dict[str, object],
        *,
        request_id: str | None = None,
        response_format: str = "wav",
        speed: float | None = None,
    ):
        audio_data = None
        for key in ("audio", "model_outputs", "latent"):
            if key in mm_output:
                audio_data = mm_output[key]
                break
        if isinstance(audio_data, list):
            for item in audio_data:
                encoded = self._encode_data_plane_audio_value(
                    item,
                    mm_output,
                    response_format=response_format,
                    speed=speed,
                )
                if encoded:
                    yield encoded
            return
        audio_data = self._slice_cumulative_data_plane_audio(request_id, audio_data)
        encoded = self._encode_data_plane_audio_value(
            audio_data,
            mm_output,
            response_format=response_format,
            speed=speed,
        )
        if encoded:
            yield encoded

    def _encode_data_plane_audio_chunks_with_duration(
        self,
        mm_output: dict[str, object],
        *,
        request_id: str | None = None,
        response_format: str = "wav",
        speed: float | None = None,
    ):
        sample_rate_hz = self._data_plane_sample_rate_hz(mm_output)
        audio_data = None
        for key in ("audio", "model_outputs", "latent"):
            if key in mm_output:
                audio_data = mm_output[key]
                break
        if isinstance(audio_data, list):
            for value in audio_data:
                encoded = self._encode_data_plane_audio_value(
                    value,
                    mm_output,
                    response_format=response_format,
                    speed=speed,
                )
                if not encoded:
                    continue
                num_samples = self._audio_num_samples(value) or 0
                duration_ms = int(num_samples * 1000 / max(1, sample_rate_hz))
                yield encoded, duration_ms
            return
        sliced = self._slice_cumulative_data_plane_audio(request_id, audio_data)
        encoded = self._encode_data_plane_audio_value(
            sliced,
            mm_output,
            response_format=response_format,
            speed=speed,
        )
        if encoded:
            num_samples = self._audio_num_samples(sliced) or 0
            duration_ms = int(num_samples * 1000 / max(1, sample_rate_hz))
            yield encoded, duration_ms

    def _slice_cumulative_data_plane_audio(
        self,
        request_id: str | None,
        audio_data: object,
    ) -> object:
        if request_id is None:
            return audio_data
        num_samples = self._audio_num_samples(audio_data)
        if num_samples is None or num_samples <= 0:
            return audio_data
        prev_samples = self._data_plane_audio_offsets.get(request_id, 0)
        if prev_samples <= 0:
            self._data_plane_audio_offsets[request_id] = num_samples
            return audio_data
        if num_samples == prev_samples:
            return None
        if num_samples < prev_samples:
            # Some stage implementations restart their cumulative waveform at
            # a model-turn boundary. Treat an actual length rollback as the
            # reset signal instead of assuming every turn restarts.
            self._data_plane_audio_offsets[request_id] = num_samples
            return audio_data
        self._data_plane_audio_offsets[request_id] = num_samples
        try:
            import numpy as np
            import torch

            if isinstance(audio_data, torch.Tensor):
                return audio_data.reshape(-1)[prev_samples:].contiguous()
            audio_array = np.asarray(audio_data, dtype=np.float32).reshape(-1)
            return audio_array[prev_samples:]
        except Exception:
            logger.exception("Failed to slice cumulative duplex audio output")
            return audio_data

    @staticmethod
    def _audio_num_samples(audio_data: object) -> int | None:
        try:
            import numpy as np
            import torch

            if isinstance(audio_data, torch.Tensor):
                return int(audio_data.numel())
            return int(np.asarray(audio_data, dtype=np.float32).size)
        except Exception:
            return None

    @classmethod
    def _data_plane_bool_metadata(
        cls,
        mm_output: dict[str, object],
        names: tuple[str, ...],
        *,
        default: bool = False,
    ) -> bool:
        def coerce(value: object) -> bool | None:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            try:
                import torch

                if isinstance(value, torch.Tensor):
                    if value.numel() == 0:
                        return None
                    return bool(value.reshape(-1)[-1].item())
            except Exception:
                pass
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return None
                return bool(value.reshape(-1)[-1].item())
            if isinstance(value, np.generic):
                return bool(value.item())
            if isinstance(value, (list, tuple)):
                return coerce(value[-1]) if value else None
            if isinstance(value, (int, float)):
                return bool(value)
            return None

        meta = mm_output.get("meta")
        for name in names:
            for key in (name, f"meta.{name}"):
                result = coerce(mm_output.get(key))
                if result is not None:
                    return result
            if isinstance(meta, dict):
                result = coerce(meta.get(name))
                if result is not None:
                    return result
        return default

    @staticmethod
    def _data_plane_sample_rate_hz(mm_output: dict[str, object]) -> int:
        sr_raw = mm_output.get("sr")
        if sr_raw is None:
            sr_raw = mm_output.get("sample_rate_hz", mm_output.get("sample_rate"))
        if sr_raw is None and isinstance(mm_output.get("meta"), dict):
            meta = mm_output["meta"]
            sr_raw = meta.get("sr") or meta.get("sample_rate_hz") or meta.get("sample_rate")
        if sr_raw is None:
            sr_raw = (
                mm_output.get("meta.sr") or mm_output.get("meta.sample_rate_hz") or mm_output.get("meta.sample_rate")
            )
        if hasattr(sr_raw, "item"):
            try:
                return int(sr_raw.item())
            except Exception:
                return 24000
        if isinstance(sr_raw, int | float):
            return int(sr_raw)
        return 24000

    @staticmethod
    def _data_plane_audio_text_marks(mm_output: dict[str, object]) -> list[dict[str, object]]:
        candidate_sources: list[object] = []
        for key in (
            "audio_text_marks",
            "text_audio_marks",
            "audio_text_alignment",
            "alignment_marks",
        ):
            candidate_sources.append(mm_output.get(key))
        meta = mm_output.get("meta")
        if isinstance(meta, dict):
            for key in (
                "audio_text_marks",
                "text_audio_marks",
                "audio_text_alignment",
                "alignment_marks",
            ):
                candidate_sources.append(meta.get(key))
        for key, value in mm_output.items():
            if (
                isinstance(key, str)
                and key.startswith("meta.")
                and key.rsplit(".", 1)[-1]
                in {
                    "audio_text_marks",
                    "text_audio_marks",
                    "audio_text_alignment",
                    "alignment_marks",
                }
            ):
                candidate_sources.append(value)

        for raw in candidate_sources:
            if not isinstance(raw, list):
                continue
            marks: list[dict[str, object]] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                text_chars = item.get("text_chars")
                audio_end_ms = item.get("audio_end_ms", item.get("audio_ms"))
                if not isinstance(text_chars, int | float) or not isinstance(audio_end_ms, int | float):
                    continue
                marks.append(
                    {
                        "text_chars": max(0, int(text_chars)),
                        "audio_end_ms": max(0, int(audio_end_ms)),
                    }
                )
            if marks:
                return marks
        return []

    @staticmethod
    def _data_plane_llm_output_text(mm_output: dict[str, object]) -> str:
        candidates: list[object] = [
            mm_output.get("llm_output_text"),
            mm_output.get("text"),
            mm_output.get("llm_output_text_utf8"),
            mm_output.get("meta.llm_output_text_utf8"),
        ]
        meta = mm_output.get("meta")
        if isinstance(meta, dict):
            candidates.extend((meta.get("llm_output_text"), meta.get("text"), meta.get("llm_output_text_utf8")))
        for key in ("meta.llm_output_text", "meta.text"):
            candidates.append(mm_output.get(key))
        for value in candidates:
            if isinstance(value, str) and value:
                return value
            decoded = OmniDuplexSessionHandler._decode_data_plane_text_tensor(value)
            if decoded:
                return decoded
            if isinstance(value, list):
                text_chunks = [item for item in value if isinstance(item, str)]
                if text_chunks:
                    return "".join(text_chunks)
        return ""

    @staticmethod
    def _decode_data_plane_text_tensor(value: object) -> str:
        if value is None:
            return ""
        try:
            if isinstance(value, np.ndarray):
                raw = value.astype(np.uint8, copy=False).reshape(-1).tobytes()
            elif hasattr(value, "detach"):
                raw = value.detach().cpu().numpy().astype(np.uint8, copy=False).reshape(-1).tobytes()
            else:
                return ""
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _encode_data_plane_audio(
        self,
        mm_output: dict[str, object],
        *,
        response_format: str = "wav",
        speed: float | None = None,
    ) -> str | None:
        for encoded in self._encode_data_plane_audio_chunks(
            mm_output,
            response_format=response_format,
            speed=speed,
        ):
            return encoded
        return None

    def _encode_data_plane_audio_value(
        self,
        audio_data: object,
        mm_output: dict[str, object],
        *,
        response_format: str = "wav",
        speed: float | None = None,
    ) -> str | None:
        if audio_data is None:
            return None
        try:
            import numpy as np
            import torch

            from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio

            if isinstance(audio_data, torch.Tensor):
                audio_tensor = audio_data.detach().cpu().float().numpy()
            else:
                audio_tensor = np.asarray(audio_data, dtype=np.float32)
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.reshape(-1)
            sample_rate = self._data_plane_sample_rate_hz(mm_output)
            audio_response = self._chat_service.create_audio(
                CreateAudio(
                    audio_tensor=audio_tensor,
                    sample_rate=sample_rate,
                    response_format=response_format,
                    speed=float(speed) if isinstance(speed, int | float) and speed > 0 else 1.0,
                    stream_format="audio",
                    base64_encode=True,
                )
            )
            return str(audio_response.audio_data)
        except Exception:
            logger.exception("Failed to encode duplex data-plane audio output")
            return None

    @staticmethod
    def _native_result_emits_model_event(native_result: dict[str, object]) -> bool:
        text = native_result.get("text")
        audio = native_result.get("audio_data", native_result.get("audio"))
        return any(
            (
                native_result.get("is_listen") is True,
                native_result.get("requires_stage_handoff") is True,
                native_result.get("requires_tts_stage") is True,
                isinstance(text, str) and bool(text),
                isinstance(audio, str) and bool(audio),
                native_result.get("end_of_turn") is True,
            )
        )

    @classmethod
    def _native_result_missing_stage_role(cls, session: DuplexSession, native_result: dict[str, object]) -> bool:
        if not session.capabilities.requires_native_stage_role:
            return False
        if not cls._native_result_emits_model_event(native_result):
            return False
        return native_result.get("stage_role") not in {"llm", "thinker", "tts", "talker"}

    @staticmethod
    def _native_result_requires_runner_kv(session: DuplexSession, native_result: dict[str, object]) -> bool:
        if not session.capabilities.requires_model_runner_kv:
            return False
        stage_role = native_result.get("stage_role")
        if stage_role not in {"llm", "thinker"}:
            return False
        if not OmniDuplexSessionHandler._native_result_emits_model_event(native_result):
            return False
        return (
            native_result.get("uses_model_runner_scheduler") is not True
            or native_result.get("runner_kv_backed") is not True
        )

    @staticmethod
    def _attach_native_runtime_metadata(payload: dict[str, object], native_result: dict[str, object]) -> None:
        metadata: dict[str, object] = {}
        runtime_impl = native_result.get("runtime_impl")
        if isinstance(runtime_impl, str) and runtime_impl:
            metadata["runtime_impl"] = runtime_impl
        owned_runtime = native_result.get("owned_runtime")
        if isinstance(owned_runtime, bool):
            metadata["owned_runtime"] = owned_runtime
        for name in (
            "uses_model_runner_scheduler",
            "runner_kv_backed",
            "per_step_tensor_handoff",
            "runner_local_payload_ref",
        ):
            value = native_result.get(name)
            if isinstance(value, bool):
                metadata[name] = value
        if metadata:
            payload["vllm_omni"] = metadata

    @staticmethod
    def _is_native_context_full(session: DuplexSession, kv_cache_length: object) -> bool:
        if not isinstance(kv_cache_length, int | float):
            return False
        raw_limit = (
            session.config.extra_body.get("duplex_context_limit")
            or session.config.extra_body.get("context_limit")
            or session.config.extra_body.get("max_context_tokens")
            or 8192
        )
        limit = int(raw_limit) if isinstance(raw_limit, int | float) else 8192
        return int(kv_cache_length) >= limit

    @classmethod
    def _iter_native_duplex_results(cls, result: object):
        if isinstance(result, dict):
            native_result = result.get("native_result")
            if isinstance(native_result, dict):
                yield native_result
            for key in ("stage_results", "result", "results"):
                value = result.get(key)
                if value is not None:
                    yield from cls._iter_native_duplex_results(value)
            return
        if isinstance(result, list | tuple):
            for item in result:
                yield from cls._iter_native_duplex_results(item)

    async def _send_runtime_error(
        self,
        send_json,
        code: str,
        exc: Exception,
        *,
        session: DuplexSession | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "type": "error",
            "code": code,
            "error": str(exc),
        }
        if session is not None:
            payload["session_id"] = session.session_id
            payload["epoch"] = session.epoch
        await send_json(payload)

    async def _send_runtime_control_error(
        self,
        send_json,
        code: str,
        message: str,
        result: dict[str, object],
        *,
        session: DuplexSession,
    ) -> None:
        await send_json(
            {
                "type": "error",
                "code": code,
                "error": message,
                "session_id": session.session_id,
                "epoch": session.epoch,
                "runtime_control": self._redact_runtime_control_result(result),
            }
        )

    async def _open_session(
        self,
        websocket: WebSocket,
        send_json,
        *,
        realtime_protocol: NativeRealtimeSessionProtocol | None = None,
    ) -> DuplexSession | None:
        raw = await self._receive_text(
            websocket,
            self._config_timeout_s,
            realtime_protocol=realtime_protocol,
        )
        if raw is None:
            await send_json({"type": "error", "error": "Timeout waiting for session.create", "code": "config_timeout"})
            return None
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            await send_json({"type": "error", "error": "Invalid JSON in session.create", "code": "invalid_json"})
            return None
        if not isinstance(event, dict) or event.get("type") not in {"session.create", "open_session", "session.config"}:
            await send_json(
                {
                    "type": "error",
                    "error": f"Expected session.create, got: {event.get('type') if isinstance(event, dict) else None}",
                    "code": "bad_event",
                }
            )
            return None

        config = DuplexSessionConfig.from_event(event)
        if config.idle_timeout_s == _DEFAULT_IDLE_TIMEOUT_S:
            config.idle_timeout_s = self._idle_timeout_s
        use_minicpmo45_native = self._use_minicpmo45_native_duplex(config)
        if use_minicpmo45_native:
            try:
                await MiniCPMO45NativeDuplexServingAdapter.prepare_session_config(
                    config,
                    model_config=getattr(self._chat_service, "model_config", None),
                )
            except ValueError as exc:
                await send_json({"type": "error", "error": str(exc), "code": "unsupported_ref_audio_path"})
                return None
        max_sessions = self._max_duplex_sessions(config)
        if max_sessions is not None and self._registry.active_count() >= max_sessions:
            await send_json(
                {
                    "type": "error",
                    "error": "Duplex session admission limit reached",
                    "code": "duplex_session_busy",
                    "active_sessions": self._registry.active_count(),
                    "max_sessions": max_sessions,
                }
            )
            return None
        session_id = event.get("session_id") if isinstance(event.get("session_id"), str) else None
        session = self._registry.create(config=config, session_id=session_id)
        if use_minicpmo45_native:
            session.capabilities = DuplexCapabilities.minicpmo45_native()
        return session

    @staticmethod
    def _max_duplex_sessions(config: DuplexSessionConfig) -> int | None:
        raw = config.extra_body.get("max_duplex_sessions") or config.extra_body.get("duplex_max_sessions")
        try:
            value = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _use_minicpmo45_native_duplex(self, config: DuplexSessionConfig) -> bool:
        return MiniCPMO45NativeDuplexServingAdapter.is_enabled(config)

    @staticmethod
    def _uses_native_input_append(session: DuplexSession) -> bool:
        return (
            session.capabilities.implementation_level == "model_native_duplex"
            and session.capabilities.supports_input_append
        )

    @staticmethod
    def _is_minicpmo45_model(model: str) -> bool:
        normalized = model.lower().replace("_", "-")
        return "minicpm-o-4-5" in normalized or "minicpmo-4-5" in normalized or "minicpmo45" in normalized

    @staticmethod
    def _native_stage0_request_id(session: DuplexSession, epoch: int) -> str:
        return f"duplex-{session.session_id}-e{epoch}-stage0"

    @staticmethod
    def _session_auto_responds(session: DuplexSession) -> bool:
        """Full-duplex / model-driven mode.

        When set, the server runs per-chunk speak-generation continuously (like
        the official MiniCPM-o ``duplex_generate`` loop) instead of waiting for an
        explicit ``response.create``: each ~chunk_period of appended audio is
        emitted and fed to the stage0 stream so the model itself decides to speak
        or listen. Signaled by the client via ``extra_body.auto_response`` (or
        ``extra_body.full_duplex``).
        """
        extra = getattr(session.config, "extra_body", None)
        if not isinstance(extra, dict):
            return False
        return extra.get("auto_response") is True or extra.get("full_duplex") is True

    async def _receive_text(
        self,
        websocket: WebSocket,
        timeout_s: float,
        *,
        realtime_protocol: NativeRealtimeSessionProtocol | None = None,
    ) -> str | None:
        try:
            if realtime_protocol is not None:
                return await asyncio.wait_for(
                    realtime_protocol.receive_internal_event_text(websocket),
                    timeout=max(0.1, timeout_s),
                )
            return await asyncio.wait_for(websocket.receive_text(), timeout=max(0.1, timeout_s))
        except asyncio.TimeoutError:
            return None

    @staticmethod
    def _input_committed_payload(
        session: DuplexSession,
        committed: DuplexCommittedInput,
        *,
        realtime_item_id: object | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "input.committed",
            "session_id": session.session_id,
            "turn_id": committed.turn_id,
            "epoch": committed.epoch,
            "history_len": len(session.history),
            "message": committed.message,
        }
        if isinstance(realtime_item_id, str) and realtime_item_id:
            payload["realtime_item_id"] = realtime_item_id
        return payload

    @staticmethod
    def _commit_native_audio_input(
        session: DuplexSession,
        *,
        realtime_item_id: object | None = None,
        transcript: object | None = None,
    ) -> DuplexCommittedInput:
        clean_transcript = transcript.strip() if isinstance(transcript, str) else None
        committed = session.commit_native_audio_input(transcript=clean_transcript or None)
        if isinstance(realtime_item_id, str) and realtime_item_id:
            session.register_history_item(realtime_item_id, committed.message)
        return committed

    @staticmethod
    def _native_audio_committed_payload(
        session: DuplexSession,
        *,
        committed: DuplexCommittedInput | None = None,
        realtime_item_id: object | None = None,
        transcript: object | None = None,
    ) -> dict[str, object]:
        message = committed.message if committed is not None else None
        if not isinstance(message, dict):
            input_audio_part: dict[str, object] = {
                "type": "audio_url",
                "audio_url": {"url": "native-duplex:input-audio"},
            }
            if isinstance(transcript, str) and transcript:
                input_audio_part["transcript"] = transcript
            message = {
                "role": "user",
                "content": [input_audio_part],
            }
        payload: dict[str, object] = {
            "type": "input.committed",
            "session_id": session.session_id,
            "turn_id": committed.turn_id if committed is not None else session.turn_id,
            "epoch": committed.epoch if committed is not None else session.epoch,
            "history_len": len(session.history),
            "native_audio": True,
            "message": message,
        }
        if isinstance(transcript, str) and transcript:
            payload["transcript"] = transcript
        if isinstance(realtime_item_id, str) and realtime_item_id:
            payload["realtime_item_id"] = realtime_item_id
        return payload

    @staticmethod
    def _apply_session_update(session: DuplexSession, payload: dict[str, object]) -> dict[str, object] | None:
        model = payload.get("model")
        audio_config = payload.get("audio")
        audio_input = audio_config.get("input") if isinstance(audio_config, dict) else None
        audio_output = audio_config.get("output") if isinstance(audio_config, dict) else None
        voice = payload.get("voice")
        if not isinstance(voice, str) and isinstance(audio_output, dict):
            voice = audio_output.get("voice")
        if isinstance(model, str) and session.config.model is not None and model != session.config.model:
            return {
                "type": "error",
                "session_id": session.session_id,
                "code": "model_update_unsupported",
                "error": "session.update cannot change model for an open realtime duplex session",
            }
        if isinstance(model, str) and session.config.model is None:
            session.config.model = model
        if isinstance(voice, str) and (session.playback.generated_ms > 0 or session.playback.sent_ms > 0):
            return {
                "type": "error",
                "session_id": session.session_id,
                "code": "voice_update_after_audio_unsupported",
                "error": "session.update cannot change voice after audio output has started",
            }
        if isinstance(payload.get("ref_audio"), str):
            return {
                "type": "error",
                "session_id": session.session_id,
                "code": "ref_audio_update_unsupported",
                "error": "session.update cannot change ref_audio after the native duplex runtime is open",
            }
        if isinstance(payload.get("instructions"), str):
            session.config.instructions = str(payload["instructions"])
        elif "instructions" in payload and payload.get("instructions") is None:
            session.config.instructions = None
        if isinstance(voice, str):
            session.config.voice = str(voice)
        elif "voice" in payload and payload.get("voice") is None:
            session.config.voice = None
        response_format = payload.get("output_audio_format") or payload.get("response_format")
        if response_format is None and isinstance(audio_config, dict):
            if isinstance(audio_output, dict):
                response_format = audio_output.get("format")
        response_format, _ = NativeRealtimeSessionProtocol._parse_realtime_audio_format(response_format)
        if isinstance(response_format, str) and response_format.lower() in REALTIME_OUTPUT_AUDIO_FORMATS:
            session.config.response_format = NativeRealtimeSessionProtocol._duplex_response_format(response_format)
        if isinstance(payload.get("temperature"), int | float):
            session.config.temperature = float(payload["temperature"])
        speed = payload.get("speed")
        if not isinstance(speed, int | float) and isinstance(audio_output, dict):
            speed = audio_output.get("speed")
        if isinstance(speed, int | float):
            session.config.speed = float(speed)
        max_tokens = (
            payload.get("max_response_output_tokens")
            if "max_response_output_tokens" in payload
            else payload.get("max_output_tokens")
            if "max_output_tokens" in payload
            else payload.get("max_tokens")
        )
        if "max_response_output_tokens" in payload or "max_output_tokens" in payload or "max_tokens" in payload:
            session.config.max_tokens = NativeRealtimeSessionProtocol.realtime_max_output_tokens(max_tokens)
        if isinstance(payload.get("overlap_policy"), str):
            session.config.overlap_policy = DuplexSessionConfig._normalize_overlap_policy(
                str(payload["overlap_policy"])
            )
        if isinstance(payload.get("overlap_short_ack_ms"), int | float):
            session.config.overlap_short_ack_ms = max(0, int(payload["overlap_short_ack_ms"]))
        if isinstance(payload.get("overlap_barge_in_ms"), int | float):
            session.config.overlap_barge_in_ms = max(0, int(payload["overlap_barge_in_ms"]))
        if isinstance(payload.get("overlap_silence_rms"), int | float):
            session.config.overlap_silence_rms = max(0.0, float(payload["overlap_silence_rms"]))
        if isinstance(payload.get("playback_commit_policy"), str):
            session.config.playback_commit_policy = DuplexSessionConfig._normalize_playback_commit_policy(
                str(payload["playback_commit_policy"])
            )
        modalities = payload.get("modalities") or payload.get("output_modalities")
        if isinstance(modalities, list) and all(isinstance(item, str) for item in modalities):
            session.config.modalities = list(modalities)
        if isinstance(payload.get("extra_body"), dict):
            session.config.extra_body.update(payload["extra_body"])
            extra = payload["extra_body"]
            if isinstance(extra.get("overlap_policy"), str):
                session.config.overlap_policy = DuplexSessionConfig._normalize_overlap_policy(
                    str(extra["overlap_policy"])
                )
            if isinstance(extra.get("playback_commit_policy"), str):
                session.config.playback_commit_policy = DuplexSessionConfig._normalize_playback_commit_policy(
                    str(extra["playback_commit_policy"])
                )
        if isinstance(payload.get("tools"), list):
            session.config.extra_body["realtime_tools"] = payload["tools"]
        elif "tools" in payload and payload.get("tools") is None:
            session.config.extra_body.pop("realtime_tools", None)
        if isinstance(payload.get("tool_choice"), str | dict):
            session.config.extra_body["realtime_tool_choice"] = payload["tool_choice"]
        elif "tool_choice" in payload and payload.get("tool_choice") is None:
            session.config.extra_body.pop("realtime_tool_choice", None)
        if isinstance(payload.get("metadata"), dict):
            session.config.extra_body["realtime_metadata"] = dict(payload["metadata"])
        elif "metadata" in payload and payload.get("metadata") is None:
            session.config.extra_body.pop("realtime_metadata", None)
        if isinstance(payload.get("include"), list):
            session.config.extra_body["realtime_include"] = list(payload["include"])
        elif "include" in payload and payload.get("include") is None:
            session.config.extra_body.pop("realtime_include", None)
        if isinstance(payload.get("prompt"), dict):
            session.config.extra_body["realtime_prompt"] = dict(payload["prompt"])
        elif "prompt" in payload and payload.get("prompt") is None:
            session.config.extra_body.pop("realtime_prompt", None)
        turn_detection = NativeRealtimeSessionProtocol._turn_detection_config(payload)
        if isinstance(turn_detection, dict):
            session.config.extra_body["realtime_turn_detection"] = dict(turn_detection)
        elif "turn_detection" in payload and payload.get("turn_detection") is None:
            session.config.extra_body.pop("realtime_turn_detection", None)
        input_audio_transcription = NativeRealtimeSessionProtocol._input_audio_transcription_config(payload)
        if isinstance(input_audio_transcription, dict):
            session.config.extra_body["realtime_input_audio_transcription"] = dict(input_audio_transcription)
        elif "input_audio_transcription" in payload and payload.get("input_audio_transcription") is None:
            session.config.extra_body.pop("realtime_input_audio_transcription", None)
        if isinstance(payload.get("input_audio_noise_reduction"), dict):
            session.config.extra_body["realtime_input_audio_noise_reduction"] = dict(
                payload["input_audio_noise_reduction"]
            )
        elif "input_audio_noise_reduction" in payload and payload.get("input_audio_noise_reduction") is None:
            session.config.extra_body.pop("realtime_input_audio_noise_reduction", None)
        if isinstance(audio_input, dict) and isinstance(audio_input.get("noise_reduction"), dict):
            session.config.extra_body["realtime_input_audio_noise_reduction"] = dict(audio_input["noise_reduction"])
        elif isinstance(audio_input, dict) and audio_input.get("noise_reduction") is None:
            session.config.extra_body.pop("realtime_input_audio_noise_reduction", None)
        if isinstance(payload.get("audio"), dict):
            session.config.extra_body["realtime_audio"] = dict(payload["audio"])
        elif "audio" in payload and payload.get("audio") is None:
            session.config.extra_body.pop("realtime_audio", None)
        if isinstance(payload.get("tracing"), str | dict):
            session.config.extra_body["realtime_tracing"] = payload["tracing"]
        elif "tracing" in payload and payload.get("tracing") is None:
            session.config.extra_body.pop("realtime_tracing", None)
        session.config.extra_body["realtime_session_payload"] = (
            NativeRealtimeSessionProtocol._json_safe_realtime_payload(payload)
        )
        session.touch()
        return None

    @staticmethod
    def _apply_response_create_options(session: DuplexSession, payload: dict[str, object]) -> None:
        """Apply per-response Realtime options to the next duplex response.

        OpenAI Realtime lets ``response.create`` override a subset of session
        fields for the requested response. vLLM-Omni still stores those knobs
        on the session config before scheduling the next response because the
        downstream chat/native paths read a single config object.
        """
        if isinstance(payload.get("instructions"), str):
            session.config.instructions = str(payload["instructions"])
        audio_config = payload.get("audio")
        audio_output = audio_config.get("output") if isinstance(audio_config, dict) else None
        voice = payload.get("voice")
        if not isinstance(voice, str) and isinstance(audio_output, dict):
            voice = audio_output.get("voice")
        if isinstance(voice, str):
            session.config.voice = str(voice)
        response_format = payload.get("output_audio_format") or payload.get("response_format")
        if response_format is None and isinstance(audio_config, dict):
            if isinstance(audio_output, dict):
                response_format = audio_output.get("format")
        response_format, _ = NativeRealtimeSessionProtocol._parse_realtime_audio_format(response_format)
        if isinstance(response_format, str) and response_format.lower() in REALTIME_OUTPUT_AUDIO_FORMATS:
            session.config.response_format = NativeRealtimeSessionProtocol._duplex_response_format(response_format)
        if isinstance(payload.get("temperature"), int | float):
            session.config.temperature = float(payload["temperature"])
        speed = payload.get("speed")
        if not isinstance(speed, int | float) and isinstance(audio_output, dict):
            speed = audio_output.get("speed")
        if isinstance(speed, int | float):
            session.config.speed = float(speed)
        max_tokens = (
            payload.get("max_response_output_tokens")
            if "max_response_output_tokens" in payload
            else payload.get("max_output_tokens")
            if "max_output_tokens" in payload
            else payload.get("max_tokens")
        )
        if "max_response_output_tokens" in payload or "max_output_tokens" in payload or "max_tokens" in payload:
            session.config.max_tokens = NativeRealtimeSessionProtocol.realtime_max_output_tokens(max_tokens)
        modalities = payload.get("modalities") or payload.get("output_modalities")
        if isinstance(modalities, list) and all(isinstance(item, str) for item in modalities):
            session.config.modalities = list(modalities)
        conversation = payload.get("conversation")
        if isinstance(conversation, str):
            session.config.extra_body["realtime_response_conversation"] = conversation
        elif "conversation" in payload and conversation is None:
            session.config.extra_body.pop("realtime_response_conversation", None)
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            session.config.extra_body["realtime_response_metadata"] = dict(metadata)
        prompt = payload.get("prompt")
        if isinstance(prompt, dict):
            session.config.extra_body["realtime_response_prompt"] = dict(prompt)
        if isinstance(payload.get("tools"), list):
            session.config.extra_body["realtime_response_tools"] = payload["tools"]
        if isinstance(payload.get("tool_choice"), str | dict):
            session.config.extra_body["realtime_response_tool_choice"] = payload["tool_choice"]
        extra_body = payload.get("extra_body")
        if isinstance(extra_body, dict):
            session.config.extra_body.update(extra_body)
        session.touch()

    @staticmethod
    def _realtime_item_to_history_message(item: object) -> dict[str, object] | None:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        if role not in {"system", "user", "assistant"}:
            return None
        content = item.get("content")
        if isinstance(content, str):
            text = content.strip()
            return {"role": role, "content": text} if text else None
        if not isinstance(content, list):
            return None
        text_chunks: list[str] = []
        audio_chunks: list[dict[str, object]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"input_text", "text", "output_text"} and isinstance(part.get("text"), str):
                text_chunks.append(str(part["text"]))
            elif part_type in {"input_audio", "audio"}:
                audio = part.get("audio") or part.get("data")
                fmt = part.get("format") if isinstance(part.get("format"), str) else "wav"
                if isinstance(audio, str) and audio:
                    audio_chunks.append(
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/{fmt};base64,{audio}",
                            },
                        }
                    )
            elif part_type in {"audio_transcript", "transcript"} and isinstance(part.get("text"), str):
                text_chunks.append(str(part["text"]))
        text = "".join(text_chunks).strip()
        if audio_chunks:
            content_items: list[dict[str, object]] = []
            if text:
                content_items.append({"type": "text", "text": text})
            content_items.extend(audio_chunks)
            return {"role": role, "content": content_items}
        if text:
            return {"role": role, "content": text}
        return None

    async def _handle_playback_ack(self, session: DuplexSession, event: dict[str, object], send_json) -> None:
        played_ms = event.get("played_ms", event.get("audio_ms", 0))
        committed_ms = event.get("committed_ms")
        if not isinstance(played_ms, int | float):
            await send_json({"type": "error", "error": "playback.ack requires played_ms", "code": "bad_event"})
            return
        committed_cursor = int(committed_ms) if isinstance(committed_ms, int | float) else int(played_ms)
        if event.get("truncate") is True:
            session.acknowledge_playback(int(played_ms), committed_cursor)
            session.truncate_playback_commit(committed_cursor)
        else:
            session.acknowledge_playback(
                int(played_ms),
                committed_cursor,
            )
        item_id = event.get("item_id")
        committed_history = False
        if isinstance(item_id, str) and item_id:
            committed_history = session.truncate_history_item(item_id, audio_end_ms=committed_cursor)
        elif session.pending_history_item_ids:
            # A plain playback ack has no OpenAI item id. Commit the only
            # uncommitted assistant candidate if the session has an unambiguous
            # pending response; otherwise wait for conversation.item.truncate.
            pending_ids = list(session.pending_history_item_ids)
            if len(pending_ids) == 1:
                item_id = pending_ids[0]
                committed_history = session.truncate_history_item(
                    item_id,
                    audio_end_ms=committed_cursor,
                )
        elif session.active_response_id is not None:
            item_id = f"item_{session.active_response_id}"
            committed_history = session.truncate_history_item(
                item_id,
                audio_end_ms=committed_cursor,
            )
        elif session.last_assistant_full_message is not None:
            response_id = event.get("response_id")
            item_id = f"item_{response_id}" if isinstance(response_id, str) and response_id else None
            if item_id is None and session.history_item_ids:
                assistant_item_ids = [
                    known_item_id
                    for known_item_id, message in session.history_item_ids.items()
                    if message.get("role") == "assistant"
                ]
                if len(assistant_item_ids) == 1:
                    item_id = assistant_item_ids[0]
            if isinstance(item_id, str) and item_id:
                committed_history = session.truncate_history_item(
                    item_id,
                    audio_end_ms=committed_cursor,
                )
        await send_json(
            {
                "type": "playback.acknowledged",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "item_id": item_id,
                "played_ms": int(played_ms),
                "committed_ms": committed_cursor,
                "truncate": event.get("truncate") is True,
                "playback": session.playback.as_dict(),
                "history_committed": committed_history,
            }
        )

    async def _cancel_active_response(
        self,
        session: DuplexSession,
        active_task: asyncio.Task[None] | None,
        send_json,
        *,
        reason: str,
        notify: bool = True,
        rebuild_runtime_history_after_truncate: bool = False,
    ) -> bool:
        has_running_task = active_task is not None and not active_task.done()
        if not has_running_task and session.active_request_id is None and session.active_response_id is None:
            return False

        old_epoch = session.epoch
        old_request_id = session.active_request_id
        old_response_id = session.active_response_id
        committed_ms = session.playback.committed_ms
        committed_message = session.end_response(
            commit_text=self._should_commit_response_to_history(old_response_id),
            playback_commit_policy=DuplexPlaybackCommitPolicy.ACK_ONLY.value,
        )
        if old_response_id is not None:
            item_id = f"item_{old_response_id}"
            if committed_message is not None:
                session.register_history_item(item_id, committed_message)
            elif committed_ms > 0:
                session.truncate_history_item(item_id, audio_end_ms=committed_ms)
            if rebuild_runtime_history_after_truncate:
                await self._signal_runtime_history_rebuild_after_truncate(
                    session,
                    old_response_id,
                    committed_ms,
                    reason=reason,
                    send_json=send_json,
                )
        new_epoch, old_playback = self._advance_barge_in_epoch(session)
        if old_request_id is not None:
            await self._abort_request_background(
                session,
                old_request_id,
                send_json,
                notify=notify,
            )
        if has_running_task and active_task is not None:
            active_task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(active_task, return_exceptions=True), timeout=0.25)
            except asyncio.TimeoutError:
                pass
        if notify:
            await send_json(
                {
                    "type": "audio.cancelled",
                    "session_id": session.session_id,
                    "response_id": old_response_id,
                    "reason": reason,
                    "cancelled_epoch": old_epoch,
                    "epoch": new_epoch,
                    "committed_ms": committed_ms,
                    "playback": old_playback,
                }
            )
        return True

    async def _abort_request_background(
        self,
        session: DuplexSession,
        request_id: str,
        send_json,
        *,
        notify: bool,
    ) -> None:
        try:
            abort_internal = getattr(self._chat_service.engine_client, "_abort_internal_requests", None)
            if callable(abort_internal):
                result = abort_internal([request_id])
            else:
                result = self._chat_service.engine_client.abort([request_id])
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.exception("Failed to abort duplex request %s: %s", request_id, exc)
            if notify and session.state != DuplexSessionState.CLOSED:
                await self._send_runtime_error(send_json, "runtime_abort_failed", exc, session=session)

    async def _cancel_pending_input(self, session: DuplexSession, send_json, *, reason: str) -> None:
        cancelled = session.cancel_pending_input()
        self._advance_barge_in_epoch(session)
        await send_json(
            {
                "type": "input.cancelled",
                "session_id": session.session_id,
                "reason": reason,
                "epoch": session.epoch,
                "cancelled": cancelled,
            }
        )

    async def _run_response(self, session: DuplexSession, send_json) -> None:
        response_id = session.begin_response()
        self._remember_response_conversation_mode(session, response_id)
        epoch = session.epoch
        request_id = f"duplex-{session.session_id}-{epoch}-{session.input_commit_seq}"
        session.active_request_id = f"chatcmpl-{request_id}"
        await send_json(
            self._response_created_payload(
                session,
                response_id,
                epoch=epoch,
                request_id=session.active_request_id,
            )
        )

        try:
            request = self._build_chat_request(session, request_id)
            result = await self._chat_service.create_chat_completion(request, raw_request=None)
            if isinstance(result, ErrorResponse):
                await send_json({"type": "error", "error": result.message, "code": result.type or "chat_error"})
                session.end_response(commit_text=False)
                return
            if hasattr(result, "__aiter__"):
                await self._drain_streaming_response(session, result, epoch, response_id, send_json)
            else:
                await self._emit_full_response(session, result, epoch, response_id, send_json)
            if session.epoch == epoch:
                should_commit = self._should_commit_response_to_history(response_id)
                committed_message = session.end_response(commit_text=should_commit)
                if should_commit:
                    session.register_history_item(f"item_{response_id}", committed_message)
                await send_json(
                    {
                        "type": "response.done",
                        "session_id": session.session_id,
                        "response_id": response_id,
                        "epoch": epoch,
                        "committed": committed_message is not None,
                        "playback": session.playback.as_dict(),
                    }
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Duplex response failed: %s", exc)
            session.end_response(commit_text=False)
            await send_json(
                {
                    "type": "error",
                    "session_id": session.session_id,
                    "response_id": response_id,
                    "error": str(exc),
                    "code": "response_error",
                }
            )

    def _build_chat_request(self, session: DuplexSession, request_id: str) -> ChatCompletionRequest:
        messages: list[dict[str, object]] = []
        if session.config.instructions:
            messages.append({"role": "system", "content": session.config.instructions})
        messages.extend(session.history)

        kwargs: dict[str, Any] = {
            "model": session.config.model or self._chat_service.model_config.model,
            "messages": messages,
            "stream": True,
        }
        if session.config.temperature is not None:
            kwargs["temperature"] = session.config.temperature
        if session.config.max_tokens is not None:
            kwargs["max_tokens"] = session.config.max_tokens
        kwargs.update(session.config.extra_body)

        request = ChatCompletionRequest(**kwargs)
        object.__setattr__(request, "modalities", session.config.modalities)
        object.__setattr__(request, "request_id", request_id)
        object.__setattr__(
            request,
            "chat_template_kwargs",
            {"use_tts_template": session.config.use_tts_template},
        )
        return request

    async def _drain_streaming_response(
        self,
        session: DuplexSession,
        result: AsyncGenerator[str, None],
        epoch: int,
        response_id: str,
        send_json,
    ) -> None:
        async for raw_chunk in result:
            if session.epoch != epoch:
                return
            for payload in self._parse_sse_payloads(raw_chunk):
                if payload == "[DONE]":
                    continue
                if isinstance(payload, dict):
                    await self._emit_chat_payload(session, payload, epoch, response_id, send_json)

    async def _emit_full_response(
        self,
        session: DuplexSession,
        result: Any,
        epoch: int,
        response_id: str,
        send_json,
    ) -> None:
        if hasattr(result, "model_dump"):
            payload = result.model_dump(mode="json", exclude_unset=True)
        else:
            payload = {"response": str(result)}
        await self._emit_chat_payload(session, payload, epoch, response_id, send_json)

    def _parse_sse_payloads(self, raw_chunk: str) -> list[dict[str, object] | str]:
        payloads: list[dict[str, object] | str] = []
        for line in raw_chunk.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                payloads.append(data)
                continue
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                logger.debug("Skipping non-JSON duplex stream payload: %s", data)
                continue
            if isinstance(parsed, dict):
                payloads.append(parsed)
        return payloads

    async def _emit_chat_payload(
        self,
        session: DuplexSession,
        payload: dict[str, object],
        epoch: int,
        response_id: str,
        send_json,
    ) -> None:
        modality = payload.get("modality")
        choices = payload.get("choices")
        if not isinstance(choices, list):
            await send_json(
                {
                    "type": "response.message",
                    "session_id": session.session_id,
                    "response_id": response_id,
                    "epoch": epoch,
                    "payload": payload,
                }
            )
            return

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            message = choice.get("message")
            content = None
            if isinstance(delta, dict):
                content = delta.get("content")
            elif isinstance(message, dict):
                content = message.get("content")

            if isinstance(content, str) and content:
                if modality == "audio":
                    session.mark_audio_sent()
                    await send_json(
                        {
                            "type": "response.output_audio.delta",
                            "session_id": session.session_id,
                            "response_id": response_id,
                            "epoch": epoch,
                            "audio": content,
                            "format": session.config.response_format,
                        }
                    )
                else:
                    session.append_assistant_text(content)
                    await send_json(
                        {
                            "type": "response.text.delta",
                            "session_id": session.session_id,
                            "response_id": response_id,
                            "epoch": epoch,
                            "delta": content,
                        }
                    )

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and modality != "audio":
                await send_json(
                    {
                        "type": "response.output_item.done",
                        "session_id": session.session_id,
                        "response_id": response_id,
                        "epoch": epoch,
                        "finish_reason": finish_reason,
                        "modality": modality,
                    }
                )
