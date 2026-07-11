from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class DuplexOverlapPolicy(str, Enum):
    AUTO = "auto"
    LISTEN_ONLY = "listen_only"
    BARGE_IN_ON_SPEECH = "barge_in_on_speech"


class DuplexPlaybackCommitPolicy(str, Enum):
    COMMIT_ALL_ON_DONE = "commit_all_on_done"
    ACK_ONLY = "ack_only"


class DuplexSessionState(str, Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class DuplexTurnState(str, Enum):
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    USER_COMMITTED = "user_committed"
    ASSISTANT_GENERATING = "assistant_generating"
    ASSISTANT_PLAYING = "assistant_playing"
    BARGE_IN = "barge_in"


class DuplexTurnEventType(str, Enum):
    USER_STARTED = "user_started"
    USER_COMMITTED = "user_committed"
    ASSISTANT_STARTED = "assistant_started"
    ASSISTANT_DONE = "assistant_done"
    BARGE_IN = "barge_in"
    PLAYBACK_ACK = "playback_ack"
    TIMEOUT = "timeout"
    CLOSE = "close"


@dataclass
class DuplexCapabilities:
    """Runtime/model capabilities exposed by the duplex serving protocol.

    These are intentionally explicit so the serving layer does not assume all
    duplex models support the same input append, rollback, or turn policy.
    ``supports_core_kv_lease`` is reserved for scheduler-owned KV lifecycle;
    model-owned decoder/TTS state must use ``supports_model_internal_state``.
    ``supports_core_resumable_request`` means the scheduler can resume the
    same request id across streaming updates, but it is not a KV lease by
    itself. Realtime support means this endpoint can speak the native Realtime
    event schema for the supported audio duplex paths while keeping model- or
    scheduler-specific limits explicit in the capability payload.
    """

    supports_session_adapter: bool = True
    supports_model_native_turn_policy: bool = False
    supports_external_turn_signal: bool = True
    supports_client_commit: bool = True
    supports_barge_in: bool = True
    supports_playback_ack: bool = True
    supports_input_append: bool = False
    supports_replace_latest_chunk: bool = True
    supports_reencode_context: bool = True
    supports_rollback_to_checkpoint: bool = False
    supports_turn_commit_only: bool = True
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
    supports_audio_truncate: bool = False
    requires_model_runner_kv: bool = False
    requires_native_stage_role: bool = False
    implementation_level: str = "serving_session_adapter"
    adapter_patterns: list[str] = field(default_factory=lambda: ["chunk_group_append"])
    input_modes: list[str] = field(default_factory=lambda: ["turn_commit_only", "reencode_context"])
    signal_sources: list[str] = field(default_factory=lambda: ["client_event", "server_policy", "model_native"])
    stage_handoff_transport: str | None = None
    chunk_period_ms: int | None = 1000
    target_barge_in_latency_ms: int | None = 1000

    @classmethod
    def minicpmo45_native(cls) -> DuplexCapabilities:
        return cls(
            supports_model_native_turn_policy=True,
            supports_input_append=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=False,
            supports_turn_commit_only=False,
            supports_kv_lease=False,
            supports_core_kv_lease=False,
            supports_model_internal_state=True,
            supports_stage_resumption=True,
            supports_scheduler_native_append=True,
            supports_core_resumable_request=True,
            supports_stage_connector_handoff=True,
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=True,
            supports_multi_session_same_replica=True,
            supports_audio_truncate=True,
            requires_model_runner_kv=True,
            requires_native_stage_role=True,
            implementation_level="model_native_duplex",
            adapter_patterns=["scheduler_data_plane"],
            input_modes=["append_audio_chunk"],
            signal_sources=["model_native", "client_event", "server_policy"],
            stage_handoff_transport="scheduler_data_plane",
            chunk_period_ms=1000,
            target_barge_in_latency_ms=1000,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "supports_session_adapter": self.supports_session_adapter,
            "supports_model_native_turn_policy": self.supports_model_native_turn_policy,
            "supports_external_turn_signal": self.supports_external_turn_signal,
            "supports_client_commit": self.supports_client_commit,
            "supports_barge_in": self.supports_barge_in,
            "supports_playback_ack": self.supports_playback_ack,
            "supports_input_append": self.supports_input_append,
            "supports_replace_latest_chunk": self.supports_replace_latest_chunk,
            "supports_reencode_context": self.supports_reencode_context,
            "supports_rollback_to_checkpoint": self.supports_rollback_to_checkpoint,
            "supports_turn_commit_only": self.supports_turn_commit_only,
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
            "supports_audio_truncate": self.supports_audio_truncate,
            "requires_model_runner_kv": self.requires_model_runner_kv,
            "requires_native_stage_role": self.requires_native_stage_role,
            "implementation_level": self.implementation_level,
            "adapter_patterns": self.adapter_patterns,
            "input_modes": self.input_modes,
            "signal_sources": self.signal_sources,
            "stage_handoff_transport": self.stage_handoff_transport,
            "chunk_period_ms": self.chunk_period_ms,
            "target_barge_in_latency_ms": self.target_barge_in_latency_ms,
        }


@dataclass
class DuplexPlaybackCursor:
    generated_ms: int = 0
    sent_ms: int = 0
    played_ms: int = 0
    committed_ms: int = 0

    def acknowledge(self, played_ms: int, committed_ms: int | None = None) -> None:
        self.played_ms = max(self.played_ms, max(0, int(played_ms)))
        if committed_ms is None:
            committed_ms = self.played_ms
        self.committed_ms = max(self.committed_ms, max(0, int(committed_ms)))

    def truncate_committed(self, committed_ms: int) -> None:
        self.committed_ms = max(0, min(max(self.sent_ms, self.generated_ms), int(committed_ms)))

    def as_dict(self) -> dict[str, int]:
        return {
            "generated_ms": self.generated_ms,
            "sent_ms": self.sent_ms,
            "played_ms": self.played_ms,
            "committed_ms": self.committed_ms,
        }


@dataclass
class DuplexAudioChunk:
    data: str
    format: str = "wav"
    sample_rate_hz: int | None = None


@dataclass
class DuplexSessionConfig:
    model: str | None = None
    modalities: list[str] = field(default_factory=lambda: ["text", "audio"])
    instructions: str | None = None
    voice: str | None = None
    ref_audio: str | None = None
    response_format: str = "wav"
    temperature: float | None = None
    max_tokens: int | None = None
    speed: float | None = None
    use_tts_template: bool = True
    idle_timeout_s: float = 300.0
    overlap_policy: str = DuplexOverlapPolicy.AUTO.value
    overlap_short_ack_ms: int = 700
    overlap_barge_in_ms: int = 1200
    overlap_silence_rms: float = 0.003
    playback_commit_policy: str = DuplexPlaybackCommitPolicy.COMMIT_ALL_ON_DONE.value
    extra_body: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "modalities": list(self.modalities),
            "instructions": self.instructions,
            "voice": self.voice,
            "ref_audio": self.ref_audio,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "speed": self.speed,
            "use_tts_template": self.use_tts_template,
            "idle_timeout_s": self.idle_timeout_s,
            "overlap_policy": self.overlap_policy,
            "overlap_short_ack_ms": self.overlap_short_ack_ms,
            "overlap_barge_in_ms": self.overlap_barge_in_ms,
            "overlap_silence_rms": self.overlap_silence_rms,
            "playback_commit_policy": self.playback_commit_policy,
            "extra_body": dict(self.extra_body),
        }

    @classmethod
    def from_event(cls, event: dict[str, object]) -> DuplexSessionConfig:
        payload = event.get("session")
        if isinstance(payload, dict):
            source = payload
        else:
            source = event

        config = cls()
        if isinstance(source.get("model"), str):
            config.model = source["model"]
        if isinstance(source.get("instructions"), str):
            config.instructions = source["instructions"]
        if isinstance(source.get("voice"), str):
            config.voice = source["voice"]
        if isinstance(source.get("ref_audio"), str):
            config.ref_audio = source["ref_audio"]
        if isinstance(source.get("response_format"), str):
            config.response_format = source["response_format"]
        if isinstance(source.get("use_tts_template"), bool):
            config.use_tts_template = bool(source["use_tts_template"])
        if isinstance(source.get("temperature"), int | float):
            config.temperature = float(source["temperature"])
        if isinstance(source.get("max_tokens"), int):
            config.max_tokens = int(source["max_tokens"])
        if isinstance(source.get("speed"), int | float):
            config.speed = float(source["speed"])
        if isinstance(source.get("idle_timeout_s"), int | float):
            config.idle_timeout_s = float(source["idle_timeout_s"])
        if isinstance(source.get("overlap_policy"), str):
            config.overlap_policy = cls._normalize_overlap_policy(source["overlap_policy"])
        if isinstance(source.get("overlap_short_ack_ms"), int | float):
            config.overlap_short_ack_ms = max(0, int(source["overlap_short_ack_ms"]))
        if isinstance(source.get("overlap_barge_in_ms"), int | float):
            config.overlap_barge_in_ms = max(0, int(source["overlap_barge_in_ms"]))
        if isinstance(source.get("overlap_silence_rms"), int | float):
            config.overlap_silence_rms = max(0.0, float(source["overlap_silence_rms"]))
        if isinstance(source.get("playback_commit_policy"), str):
            config.playback_commit_policy = cls._normalize_playback_commit_policy(source["playback_commit_policy"])
        if isinstance(source.get("modalities"), list) and all(isinstance(x, str) for x in source["modalities"]):
            config.modalities = list(source["modalities"])
        if isinstance(source.get("extra_body"), dict):
            config.extra_body = dict(source["extra_body"])
            extra = config.extra_body
            if isinstance(extra.get("overlap_policy"), str):
                config.overlap_policy = cls._normalize_overlap_policy(extra["overlap_policy"])
            if isinstance(extra.get("overlap_short_ack_ms"), int | float):
                config.overlap_short_ack_ms = max(0, int(extra["overlap_short_ack_ms"]))
            if isinstance(extra.get("overlap_barge_in_ms"), int | float):
                config.overlap_barge_in_ms = max(0, int(extra["overlap_barge_in_ms"]))
            if isinstance(extra.get("overlap_silence_rms"), int | float):
                config.overlap_silence_rms = max(0.0, float(extra["overlap_silence_rms"]))
            if isinstance(extra.get("playback_commit_policy"), str):
                config.playback_commit_policy = cls._normalize_playback_commit_policy(extra["playback_commit_policy"])
        return config

    @staticmethod
    def _normalize_overlap_policy(value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {policy.value for policy in DuplexOverlapPolicy}:
            return normalized
        return DuplexOverlapPolicy.AUTO.value

    @staticmethod
    def _normalize_playback_commit_policy(value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {policy.value for policy in DuplexPlaybackCommitPolicy}:
            return normalized
        return DuplexPlaybackCommitPolicy.COMMIT_ALL_ON_DONE.value


@dataclass
class DuplexCommittedInput:
    message: dict[str, object]
    turn_id: int
    epoch: int
    input_commit_seq: int


@dataclass
class DuplexAssistantAudioTextMark:
    text_chars: int
    audio_end_ms: int


@dataclass
class DuplexSession:
    session_id: str
    config: DuplexSessionConfig
    capabilities: DuplexCapabilities = field(default_factory=DuplexCapabilities)
    state: DuplexSessionState = DuplexSessionState.OPEN
    turn_state: DuplexTurnState = DuplexTurnState.IDLE
    epoch: int = 0
    turn_id: int = 0
    input_commit_seq: int = 0
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    history: list[dict[str, object]] = field(default_factory=list)
    pending_text: list[str] = field(default_factory=list)
    pending_audio: list[DuplexAudioChunk] = field(default_factory=list)
    active_request_id: str | None = None
    active_response_id: str | None = None
    active_response_turn_id: int | None = None
    assistant_text_buffer: list[str] = field(default_factory=list)
    assistant_audio_text_marks: list[DuplexAssistantAudioTextMark] = field(default_factory=list)
    playback: DuplexPlaybackCursor = field(default_factory=DuplexPlaybackCursor)
    history_item_ids: dict[str, dict[str, object]] = field(default_factory=dict)
    history_item_audio_text_marks: dict[str, list[DuplexAssistantAudioTextMark]] = field(default_factory=dict)
    pending_history_item_ids: dict[str, dict[str, object]] = field(default_factory=dict)
    pending_history_item_audio_text_marks: dict[str, list[DuplexAssistantAudioTextMark]] = field(default_factory=dict)
    pending_history_truncations_ms: dict[str, int] = field(default_factory=dict)
    last_assistant_full_message: dict[str, object] | None = None
    last_assistant_audio_text_marks: list[DuplexAssistantAudioTextMark] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = time.monotonic()

    def append_text(self, text: str) -> None:
        if not text:
            return
        self.pending_text.append(text)
        self.turn_state = DuplexTurnState.USER_SPEAKING
        self.touch()

    def append_audio(self, data: str, *, fmt: str = "wav", sample_rate_hz: int | None = None) -> None:
        if not data:
            return
        self.pending_audio.append(DuplexAudioChunk(data=data, format=fmt, sample_rate_hz=sample_rate_hz))
        self.turn_state = DuplexTurnState.USER_SPEAKING
        self.touch()

    def mark_user_input_activity(self) -> None:
        self.turn_state = DuplexTurnState.USER_SPEAKING
        self.touch()

    def cancel_pending_input(self) -> dict[str, int]:
        cancelled = {
            "text_chunks": len(self.pending_text),
            "audio_chunks": len(self.pending_audio),
        }
        self.pending_text.clear()
        self.pending_audio.clear()
        self.turn_state = DuplexTurnState.IDLE
        self.touch()
        return cancelled

    def commit_user_input(self) -> DuplexCommittedInput | None:
        text = "".join(self.pending_text).strip()
        audio_chunks = list(self.pending_audio)
        if not text and not audio_chunks:
            return None

        content: str | list[dict[str, object]]
        if audio_chunks:
            content_items: list[dict[str, object]] = []
            if text:
                content_items.append({"type": "text", "text": text})
            for chunk in audio_chunks:
                content_items.append(
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/{chunk.format};base64,{chunk.data}",
                        },
                    }
                )
            content = content_items
        else:
            content = text

        self.input_commit_seq += 1
        message = {"role": "user", "content": content}
        self.history.append(message)
        self.pending_text.clear()
        self.pending_audio.clear()
        self.turn_state = DuplexTurnState.USER_COMMITTED
        self.touch()
        return DuplexCommittedInput(
            message=message,
            turn_id=self.turn_id,
            epoch=self.epoch,
            input_commit_seq=self.input_commit_seq,
        )

    def commit_native_audio_input(self, *, transcript: str | None = None) -> DuplexCommittedInput:
        input_audio_part: dict[str, object] = {
            "type": "audio_url",
            "audio_url": {"url": "native-duplex:input-audio"},
        }
        if transcript:
            input_audio_part["transcript"] = transcript
        self.input_commit_seq += 1
        message = {"role": "user", "content": [input_audio_part]}
        if transcript:
            message["transcript"] = transcript
        self.history.append(message)
        self.pending_text.clear()
        self.pending_audio.clear()
        self.turn_state = DuplexTurnState.USER_COMMITTED
        self.touch()
        return DuplexCommittedInput(
            message=message,
            turn_id=self.turn_id,
            epoch=self.epoch,
            input_commit_seq=self.input_commit_seq,
        )

    def complete_model_turn(self, turn_id: int) -> None:
        """Advance the model-owned output identity after its terminal signal."""
        completed_turn_id = int(turn_id)
        if completed_turn_id >= self.turn_id:
            self.turn_id = completed_turn_id + 1
        self.touch()

    def begin_response(self, *, turn_id: int | None = None) -> str:
        self.active_response_id = f"resp-{self.session_id}-{self.epoch}-{uuid4().hex[:8]}"
        self.active_response_turn_id = self.turn_id if turn_id is None else int(turn_id)
        self.assistant_text_buffer.clear()
        self.assistant_audio_text_marks.clear()
        self.last_assistant_full_message = None
        self.last_assistant_audio_text_marks.clear()
        self.playback = DuplexPlaybackCursor()
        self.turn_state = DuplexTurnState.ASSISTANT_GENERATING
        self.touch()
        return self.active_response_id

    def append_assistant_text(self, text: str) -> None:
        if text:
            self.assistant_text_buffer.append(text)
            self.touch()

    def mark_audio_sent(
        self,
        duration_ms: int | None = None,
        *,
        text_chars: int | None = None,
        audio_text_marks: list[dict[str, object]] | None = None,
    ) -> None:
        if duration_ms is not None:
            self.playback.generated_ms = max(self.playback.generated_ms, duration_ms)
            self.playback.sent_ms = max(self.playback.sent_ms, duration_ms)
            if text_chars is not None and text_chars >= 0:
                self.assistant_audio_text_marks.append(
                    DuplexAssistantAudioTextMark(
                        text_chars=int(text_chars),
                        audio_end_ms=max(0, int(duration_ms)),
                    )
                )
        if audio_text_marks:
            for raw_mark in audio_text_marks:
                if not isinstance(raw_mark, dict):
                    continue
                raw_text_chars = raw_mark.get("text_chars")
                raw_audio_end_ms = raw_mark.get("audio_end_ms", raw_mark.get("audio_ms"))
                if not isinstance(raw_text_chars, int | float) or not isinstance(raw_audio_end_ms, int | float):
                    continue
                self.assistant_audio_text_marks.append(
                    DuplexAssistantAudioTextMark(
                        text_chars=max(0, int(raw_text_chars)),
                        audio_end_ms=max(0, int(raw_audio_end_ms)),
                    )
                )
        self.turn_state = DuplexTurnState.ASSISTANT_PLAYING
        self.touch()

    def acknowledge_playback(self, played_ms: int, committed_ms: int | None = None) -> None:
        self.playback.acknowledge(played_ms, committed_ms)
        self.touch()

    def truncate_playback_commit(self, committed_ms: int) -> None:
        self.playback.truncate_committed(committed_ms)
        self.touch()

    def clear_playback_cursor(self) -> None:
        self.playback = DuplexPlaybackCursor()
        self.touch()

    def end_response(
        self,
        *,
        commit_text: bool = True,
        playback_commit_policy: str | None = None,
        preserve_request: bool = False,
    ) -> dict[str, object] | None:
        assistant_text = "".join(self.assistant_text_buffer).strip()
        message = None
        if assistant_text:
            self.last_assistant_full_message = {"role": "assistant", "content": assistant_text}
            self.last_assistant_audio_text_marks = list(self.assistant_audio_text_marks)
        if commit_text and assistant_text:
            committed_text = self._playback_committed_text(
                assistant_text,
                playback_commit_policy=playback_commit_policy,
            )
        else:
            committed_text = ""
        if commit_text and committed_text:
            message = {"role": "assistant", "content": committed_text}
            self.history.append(message)
        self.assistant_text_buffer.clear()
        if not preserve_request:
            self.active_request_id = None
        self.active_response_id = None
        self.active_response_turn_id = None
        self.turn_state = DuplexTurnState.IDLE
        self.touch()
        return message

    def register_history_item(self, item_id: str | None, message: dict[str, object] | None) -> None:
        if not item_id:
            return
        if message is None:
            if self.last_assistant_full_message is None:
                return
            self.pending_history_item_ids[item_id] = dict(self.last_assistant_full_message)
            if self.last_assistant_audio_text_marks:
                self.pending_history_item_audio_text_marks[item_id] = list(self.last_assistant_audio_text_marks)
            pending_audio_ms = self.pending_history_truncations_ms.pop(item_id, None)
            if pending_audio_ms is not None:
                self.truncate_history_item(item_id, audio_end_ms=pending_audio_ms)
            self.touch()
            return
        pending_audio_ms = self.pending_history_truncations_ms.pop(item_id, None)
        if pending_audio_ms is not None:
            self._truncate_message_to_audio_ms(
                message,
                audio_end_ms=pending_audio_ms,
                marks=self.assistant_audio_text_marks or self.last_assistant_audio_text_marks,
            )
            if self._message_text_len(message) <= 0:
                try:
                    self.history.remove(message)
                except ValueError:
                    pass
                return
        self.history_item_ids[item_id] = message
        self.pending_history_item_ids.pop(item_id, None)
        self.pending_history_item_audio_text_marks.pop(item_id, None)
        if message.get("role") == "assistant":
            marks = self.assistant_audio_text_marks or self.last_assistant_audio_text_marks
            if marks:
                self.history_item_audio_text_marks[item_id] = list(marks)
        self.touch()

    def delete_history_item(self, item_id: str) -> bool:
        message = self.history_item_ids.pop(item_id, None)
        self.history_item_audio_text_marks.pop(item_id, None)
        pending = self.pending_history_item_ids.pop(item_id, None)
        self.pending_history_item_audio_text_marks.pop(item_id, None)
        self.pending_history_truncations_ms.pop(item_id, None)
        if message is None:
            if pending is not None:
                self.touch()
            return pending is not None
        try:
            self.history.remove(message)
        except ValueError:
            pass
        self.touch()
        return True

    def truncate_history_item(self, item_id: str, *, audio_end_ms: int) -> bool:
        message = self.history_item_ids.get(item_id)
        if message is None:
            pending = self.pending_history_item_ids.get(item_id)
            if pending is None:
                self.pending_history_truncations_ms[item_id] = max(0, int(audio_end_ms))
                self.touch()
                return False
            message = dict(pending)
            changed = self._truncate_message_to_audio_ms(
                message,
                audio_end_ms=audio_end_ms,
                marks=self.pending_history_item_audio_text_marks.get(item_id),
            )
            if not changed or self._message_text_len(message) <= 0:
                if changed:
                    self.pending_history_item_ids.pop(item_id, None)
                    self.pending_history_item_audio_text_marks.pop(item_id, None)
                    self.pending_history_truncations_ms.pop(item_id, None)
                    self.touch()
                return changed
            self.history.append(message)
            self.history_item_ids[item_id] = message
            if item_id in self.pending_history_item_audio_text_marks:
                self.history_item_audio_text_marks[item_id] = list(self.pending_history_item_audio_text_marks[item_id])
            self.pending_history_item_ids.pop(item_id, None)
            self.pending_history_item_audio_text_marks.pop(item_id, None)
            self.pending_history_truncations_ms.pop(item_id, None)
            self.touch()
            return True
        changed = self._truncate_message_to_audio_ms(
            message,
            audio_end_ms=audio_end_ms,
            marks=self.history_item_audio_text_marks.get(item_id),
        )
        if changed and self._message_text_len(message) <= 0:
            self.history_item_ids.pop(item_id, None)
            self.history_item_audio_text_marks.pop(item_id, None)
            try:
                self.history.remove(message)
            except ValueError:
                pass
            self.touch()
        return changed

    def _truncate_message_to_audio_ms(
        self,
        message: dict[str, object],
        *,
        audio_end_ms: int,
        marks: list[DuplexAssistantAudioTextMark] | None = None,
    ) -> bool:
        content = message.get("content")
        if isinstance(content, str):
            keep_chars = self._text_chars_for_audio_ms(
                audio_end_ms,
                len(content),
                marks=marks,
            )
            message["content"] = content[:keep_chars].rstrip()
            self.touch()
            return True
        if not isinstance(content, list):
            return False
        changed = False
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"output_audio", "audio", "audio_transcript"}:
                transcript = part.get("transcript")
                if isinstance(transcript, str):
                    keep_chars = self._text_chars_for_audio_ms(
                        audio_end_ms,
                        len(transcript),
                        marks=marks,
                    )
                    part["transcript"] = transcript[:keep_chars].rstrip()
                    changed = True
            if part_type in {"output_text", "text"}:
                text = part.get("text")
                if isinstance(text, str):
                    keep_chars = self._text_chars_for_audio_ms(
                        audio_end_ms,
                        len(text),
                        marks=marks,
                    )
                    part["text"] = text[:keep_chars].rstrip()
                    changed = True
        if not changed:
            return False
        self.touch()
        return True

    @staticmethod
    def _message_text_len(message: dict[str, object]) -> int:
        content = message.get("content")
        if isinstance(content, str):
            return len(content)
        if not isinstance(content, list):
            return 0
        total = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in ("text", "transcript"):
                value = part.get(key)
                if isinstance(value, str):
                    total += len(value)
        return total

    def _playback_committed_text(
        self,
        assistant_text: str,
        *,
        playback_commit_policy: str | None = None,
    ) -> str:
        sent_ms = max(self.playback.sent_ms, self.playback.generated_ms)
        committed_ms = self.playback.committed_ms
        policy = playback_commit_policy or self.config.playback_commit_policy
        if sent_ms <= 0 or committed_ms >= sent_ms:
            return assistant_text
        if committed_ms <= 0:
            if policy == DuplexPlaybackCommitPolicy.COMMIT_ALL_ON_DONE.value:
                return assistant_text
            return ""
        keep_chars = self._text_chars_for_audio_ms(committed_ms, len(assistant_text))
        if keep_chars <= 0:
            return ""
        return assistant_text[:keep_chars].rstrip()

    def _text_chars_for_audio_ms(
        self,
        audio_end_ms: int,
        text_len: int,
        *,
        marks: list[DuplexAssistantAudioTextMark] | None = None,
    ) -> int:
        if text_len <= 0:
            return 0
        audio_end_ms = max(0, int(audio_end_ms))
        marks = marks if marks is not None else self.assistant_audio_text_marks
        if not marks:
            sent_ms = max(1, self.playback.sent_ms, self.playback.generated_ms)
            return int(text_len * max(0.0, min(1.0, audio_end_ms / sent_ms)))
        marks = sorted(
            (mark for mark in marks if mark.audio_end_ms >= 0 and mark.text_chars >= 0),
            key=lambda mark: mark.audio_end_ms,
        )
        if not marks:
            return 0
        if audio_end_ms <= 0:
            return 0
        previous_ms = 0
        previous_chars = 0
        for mark in marks:
            mark_ms = max(previous_ms, mark.audio_end_ms)
            mark_chars = min(text_len, max(previous_chars, mark.text_chars))
            if audio_end_ms <= mark_ms:
                if mark_ms <= previous_ms:
                    return mark_chars
                ratio = (audio_end_ms - previous_ms) / max(1, mark_ms - previous_ms)
                return int(previous_chars + (mark_chars - previous_chars) * max(0.0, min(1.0, ratio)))
            previous_ms = mark_ms
            previous_chars = mark_chars
        final_ms = max(self.playback.sent_ms, self.playback.generated_ms, previous_ms)
        if audio_end_ms >= final_ms:
            return text_len
        ratio = (audio_end_ms - previous_ms) / max(1, final_ms - previous_ms)
        return int(previous_chars + (text_len - previous_chars) * max(0.0, min(1.0, ratio)))

    def barge_in(self) -> int:
        self.epoch += 1
        self.pending_text.clear()
        self.pending_audio.clear()
        self.assistant_text_buffer.clear()
        self.assistant_audio_text_marks.clear()
        self.active_request_id = None
        self.active_response_id = None
        self.active_response_turn_id = None
        self.turn_state = DuplexTurnState.BARGE_IN
        self.touch()
        return self.epoch

    def mark_closing(self) -> None:
        if self.state != DuplexSessionState.CLOSED:
            self.state = DuplexSessionState.CLOSING
        self.touch()

    def close(self) -> None:
        self.state = DuplexSessionState.CLOSED
        self.turn_state = DuplexTurnState.IDLE
        self.active_response_turn_id = None
        self.touch()

    def is_idle_expired(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return now - self.updated_at > self.config.idle_timeout_s

    def as_public_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.session_id,
            "state": self.state.value,
            "turn_state": self.turn_state.value,
            "epoch": self.epoch,
            "turn_id": self.turn_id,
            "active_request_id": self.active_request_id,
            "active_response_id": self.active_response_id,
            "active_response_turn_id": self.active_response_turn_id,
            "model": self.config.model,
            "modalities": list(self.config.modalities),
            "instructions": self.config.instructions,
            "voice": self.config.voice,
            "response_format": self.config.response_format,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "speed": self.config.speed,
            "idle_timeout_s": self.config.idle_timeout_s,
            "overlap_policy": self.config.overlap_policy,
            "overlap_short_ack_ms": self.config.overlap_short_ack_ms,
            "overlap_barge_in_ms": self.config.overlap_barge_in_ms,
            "overlap_silence_rms": self.config.overlap_silence_rms,
            "playback_commit_policy": self.config.playback_commit_policy,
            "playback": self.playback.as_dict(),
            "capabilities": self.capabilities.as_dict(),
        }
        if isinstance(self.config.extra_body.get("realtime_tools"), list):
            payload["tools"] = self.config.extra_body["realtime_tools"]
        if isinstance(self.config.extra_body.get("realtime_tool_choice"), str | dict):
            payload["tool_choice"] = self.config.extra_body["realtime_tool_choice"]
        if isinstance(self.config.extra_body.get("realtime_metadata"), dict):
            payload["metadata"] = dict(self.config.extra_body["realtime_metadata"])
        if isinstance(self.config.extra_body.get("realtime_include"), list):
            payload["include"] = list(self.config.extra_body["realtime_include"])
        if isinstance(self.config.extra_body.get("realtime_prompt"), dict):
            payload["prompt"] = dict(self.config.extra_body["realtime_prompt"])
        if isinstance(self.config.extra_body.get("realtime_turn_detection"), dict):
            payload["turn_detection"] = dict(self.config.extra_body["realtime_turn_detection"])
        if isinstance(self.config.extra_body.get("realtime_input_audio_transcription"), dict):
            payload["input_audio_transcription"] = dict(self.config.extra_body["realtime_input_audio_transcription"])
        if isinstance(self.config.extra_body.get("realtime_input_audio_noise_reduction"), dict):
            payload["input_audio_noise_reduction"] = dict(
                self.config.extra_body["realtime_input_audio_noise_reduction"]
            )
        if isinstance(self.config.extra_body.get("realtime_audio"), dict):
            payload["audio"] = dict(self.config.extra_body["realtime_audio"])
        if isinstance(self.config.extra_body.get("realtime_tracing"), str | dict):
            payload["tracing"] = self.config.extra_body["realtime_tracing"]
        raw_realtime_session = self.config.extra_body.get("realtime_session_payload")
        if isinstance(raw_realtime_session, dict):
            for key, value in raw_realtime_session.items():
                if key not in payload and key != "extra_body":
                    payload[key] = value
        return payload


class DuplexTurnController:
    """Interaction controller that accepts signals from multiple sources."""

    def signal(
        self,
        session: DuplexSession,
        event_type: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = payload or {}
        if event_type == DuplexTurnEventType.USER_STARTED.value:
            session.turn_state = DuplexTurnState.USER_SPEAKING
        elif event_type == DuplexTurnEventType.USER_COMMITTED.value:
            session.turn_state = DuplexTurnState.USER_COMMITTED
        elif event_type == DuplexTurnEventType.ASSISTANT_STARTED.value:
            session.turn_state = DuplexTurnState.ASSISTANT_GENERATING
        elif event_type == DuplexTurnEventType.ASSISTANT_DONE.value:
            session.turn_state = DuplexTurnState.IDLE
        elif event_type == DuplexTurnEventType.PLAYBACK_ACK.value:
            played_ms = int(payload.get("played_ms", 0) or 0)
            committed_ms = payload.get("committed_ms")
            session.acknowledge_playback(
                played_ms,
                int(committed_ms) if isinstance(committed_ms, int | float) else None,
            )
        elif event_type == DuplexTurnEventType.BARGE_IN.value:
            session.turn_state = DuplexTurnState.BARGE_IN
        elif event_type in {DuplexTurnEventType.CLOSE.value, DuplexTurnEventType.TIMEOUT.value}:
            session.state = DuplexSessionState.CLOSING
        session.touch()
        return {
            "type": "turn.event",
            "session_id": session.session_id,
            "event": event_type,
            "turn_state": session.turn_state.value,
            "epoch": session.epoch,
        }


class DuplexSessionRegistry:
    def __init__(self, capabilities: DuplexCapabilities | None = None) -> None:
        self._capabilities = capabilities or DuplexCapabilities()
        self._sessions: dict[str, DuplexSession] = {}

    def create(self, config: DuplexSessionConfig | None = None, session_id: str | None = None) -> DuplexSession:
        sid = session_id or f"duplex-{uuid4().hex}"
        if sid in self._sessions:
            raise ValueError(f"Duplex session already exists: {sid}")
        session = DuplexSession(
            session_id=sid,
            config=config or DuplexSessionConfig(),
            capabilities=self._capabilities,
        )
        self._sessions[sid] = session
        return session

    def active_count(self) -> int:
        return len(self._sessions)

    def get(self, session_id: str) -> DuplexSession | None:
        return self._sessions.get(session_id)

    def close(self, session_id: str) -> DuplexSession | None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()
        return session

    def reap_expired(self) -> list[DuplexSession]:
        expired: list[DuplexSession] = []
        for session_id, session in list(self._sessions.items()):
            if session.is_idle_expired():
                expired.append(session)
                self.close(session_id)
        return expired
