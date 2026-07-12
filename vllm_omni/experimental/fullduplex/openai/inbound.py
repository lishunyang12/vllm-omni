# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import array
import base64
import binascii
import sys
from collections.abc import AsyncIterator

from vllm_omni.experimental.fullduplex.core.events import (
    DomainEvent,
    InputChunk,
    InputCommitted,
    InputStarted,
    InterruptRequested,
    PlaybackAcknowledged,
    SessionCloseRequested,
)
from vllm_omni.experimental.fullduplex.minicpmo45.input import MiniCPMO45PcmAppendBuffer

# One model chunk per second of input audio, matching the reference demo's
# continuous streaming cadence (chunk_ms=1000, input sample_rate=16000).
DEFAULT_CHUNK_PERIOD_MS = 1000
# Output audio runs at 24 kHz; playback acks expressed in milliseconds convert
# to the sample cursor unit the reducer/adapter use.
OUTPUT_SAMPLE_RATE_HZ = 24000

# Inbound event names (OpenAI-Realtime-style, with the demo's aliases).
# Client audio formats that mean signed 16-bit little-endian PCM. The demo's
# app.js uploads mic audio as pcm16; the engine/buffer work in pcm_f32le, so
# these are converted on the way in.
_PCM16_FORMATS = frozenset({"pcm16", "pcm_s16le", "s16le", "pcm"})

_APPEND_EVENTS = frozenset({"input_audio_buffer.append", "input.audio.append", "input.append", "push_chunk"})
_COMMIT_EVENTS = frozenset({"input_audio_buffer.commit", "input.commit", "commit"})
_INTERRUPT_EVENTS = frozenset({"response.cancel", "barge_in", "input.cancel", "output_audio_buffer.clear"})
_ACK_EVENTS = frozenset({"playback.ack", "audio.playback_ack"})
_CLOSE_EVENTS = frozenset({"session.close", "close_session", "close"})


def _frame_type(frame: dict) -> str:
    value = frame.get("type") or frame.get("event")
    return value if isinstance(value, str) else ""


def _extract_audio_payload(frame: dict, *, default_format: str = "pcm_f32le") -> dict | None:
    """Pull a ``{format, sample_rate_hz, audio}`` payload out of a client frame.

    Accepts the audio base64 under ``audio``/``audio_base64``/``delta`` either at
    the top level or nested under ``input``. When a frame carries no per-frame
    ``format`` the session-level ``input_audio_format`` (``default_format``) is
    used — the OpenAI realtime dialect (and the OpenBMB demo/bridge) declare the
    input format once on ``session.update`` and omit it on every append, so a
    pcm16 stream would otherwise be misread as ``pcm_f32le`` and decode to NaN.
    Sample rate defaults to 16 kHz — the handler normalizes to that before
    frames reach here.
    """
    source = frame.get("input") if isinstance(frame.get("input"), dict) else frame
    audio = None
    for key in ("audio", "audio_base64", "audio_data", "delta"):
        candidate = source.get(key)
        if isinstance(candidate, str) and candidate:
            audio = candidate
            break
    if audio is None:
        return None
    fmt = str(source.get("format", default_format)).lower()
    if fmt in _PCM16_FORMATS:
        audio = _pcm16_b64_to_f32le_b64(audio)
        if audio is None:
            return None
        fmt = "pcm_f32le"
    return {
        "type": "audio",
        "audio": audio,
        "format": fmt,
        "sample_rate_hz": int(source.get("sample_rate_hz", 16000)),
        "force_listen": bool(source.get("force_listen", False)),
        "is_speech": bool(source.get("is_speech", False)),
    }


def _pcm16_b64_to_f32le_b64(encoded: str) -> str | None:
    """Convert base64 signed-16-bit LE PCM to base64 little-endian float32 PCM."""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not raw or len(raw) % 2 != 0:
        return None
    samples = array.array("h")  # signed 16-bit, native byte order
    samples.frombytes(raw)
    if sys.byteorder == "big":
        samples.byteswap()  # the wire bytes are little-endian
    floats = array.array("f", (s / 32768.0 for s in samples))
    if sys.byteorder == "big":
        floats.byteswap()  # emit little-endian float32 for pcm_f32le
    return base64.b64encode(floats.tobytes()).decode("ascii")


def _ack_cursor(frame: dict, key_samples: str, key_ms: str) -> int | None:
    source = frame.get("input") if isinstance(frame.get("input"), dict) else frame
    samples = source.get(key_samples)
    if isinstance(samples, int | float):
        return max(0, int(samples))
    millis = source.get(key_ms)
    if isinstance(millis, int | float):
        return max(0, int(round(float(millis) * OUTPUT_SAMPLE_RATE_HZ / 1000.0)))
    return None


class RealtimeInboundAdapter:
    """Translate parsed client WebSocket frames into duplex :class:`DomainEvent`s.

    Audio appends are accumulated into whole model chunks by the shared
    :class:`MiniCPMO45PcmAppendBuffer`; each whole chunk becomes one
    :class:`InputChunk` (raw ``float32`` PCM bytes). ``InputStarted`` is emitted
    once per streaming segment. A commit flushes the residual (zero-padded to a
    chunk boundary) then emits :class:`InputCommitted`.

    Per-chunk ``force_listen``/``is_speech`` hints from the demo are not carried
    on the FSM ``InputChunk`` (which is modality+data only); explicit user
    interruption is modeled as :class:`InterruptRequested` (a barge-in epoch
    bump) instead, which is the FSM-native equivalent.
    """

    def __init__(
        self,
        *,
        chunk_period_ms: int = DEFAULT_CHUNK_PERIOD_MS,
        input_audio_format: str = "pcm_f32le",
    ) -> None:
        self._buffer = MiniCPMO45PcmAppendBuffer()
        self._chunk_period_ms = chunk_period_ms
        # Session-level input format (OpenAI realtime `session.update`): appends
        # normally omit a per-frame format, so this is what pcm16 streams are
        # converted from. Without it pcm16 bytes decode as float32 -> NaN.
        self._input_audio_format = str(input_audio_format or "pcm_f32le").lower()
        self._streaming = False

    async def events(self, frames: AsyncIterator[dict]) -> AsyncIterator[DomainEvent]:
        async for frame in frames:
            if not isinstance(frame, dict):
                continue
            kind = _frame_type(frame)
            if kind in _APPEND_EVENTS:
                for event in self._on_append(frame):
                    yield event
            elif kind in _COMMIT_EVENTS:
                for event in self._on_commit():
                    yield event
            elif kind in _INTERRUPT_EVENTS:
                yield InterruptRequested(reason=kind)
            elif kind in _ACK_EVENTS:
                event = self._on_ack(frame)
                if event is not None:
                    yield event
            elif kind in _CLOSE_EVENTS:
                reason = frame.get("reason")
                yield SessionCloseRequested(reason=reason if isinstance(reason, str) and reason else "client_close")
                return

    def _on_append(self, frame: dict) -> list[DomainEvent]:
        payload = _extract_audio_payload(frame, default_format=self._input_audio_format)
        if payload is None:
            return []
        chunk = self._buffer.append(payload, chunk_period_ms=self._chunk_period_ms)
        return self._emit_chunk(chunk)

    def _on_commit(self) -> list[DomainEvent]:
        events: list[DomainEvent] = []
        committed = self._buffer.commit(chunk_period_ms=self._chunk_period_ms)
        if committed.payload is not None:
            events.extend(self._emit_chunk(committed.payload))
        if self._streaming:
            events.append(InputCommitted())
            self._streaming = False
        return events

    def _emit_chunk(self, payload: dict | None) -> list[DomainEvent]:
        pcm = _decode_chunk_pcm(payload)
        if pcm is None:
            return []
        events: list[DomainEvent] = []
        if not self._streaming:
            events.append(InputStarted(modality="audio"))
            self._streaming = True
        events.append(InputChunk(data=pcm, modality="audio"))
        return events

    def _on_ack(self, frame: dict) -> PlaybackAcknowledged | None:
        played = _ack_cursor(frame, "played_samples", "played_ms")
        if played is None:
            return None
        committed = _ack_cursor(frame, "committed_samples", "committed_ms")
        return PlaybackAcknowledged(cursor=played, committed_cursor=committed)


def _decode_chunk_pcm(payload: dict | None) -> bytes | None:
    if not isinstance(payload, dict):
        return None
    audio = payload.get("audio")
    if not isinstance(audio, str) or not audio:
        return None
    try:
        pcm = base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not pcm or len(pcm) % 4 != 0:
        return None
    return pcm


__all__ = [
    "DEFAULT_CHUNK_PERIOD_MS",
    "OUTPUT_SAMPLE_RATE_HZ",
    "RealtimeInboundAdapter",
]
