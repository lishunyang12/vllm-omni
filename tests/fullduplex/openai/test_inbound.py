# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64

import pytest

from vllm_omni.experimental.fullduplex.core.events import (
    InputChunk,
    InputCommitted,
    InputStarted,
    InterruptRequested,
    PlaybackAcknowledged,
    SessionCloseRequested,
)
from vllm_omni.experimental.fullduplex.openai.inbound import RealtimeInboundAdapter

# 100 Hz keeps a "1 s" model chunk at 100 samples (400 bytes) so tests stay tiny.
_SR = 100


def _pcm(num_samples: int) -> str:
    return base64.b64encode(b"\x00\x00\x00\x00" * num_samples).decode("ascii")


def _append(num_samples: int) -> dict:
    return {
        "type": "input_audio_buffer.append",
        "audio": _pcm(num_samples),
        "format": "pcm_f32le",
        "sample_rate_hz": _SR,
        "is_speech": True,
    }


async def _aiter(items):
    for item in items:
        yield item


async def _collect(frames):
    adapter = RealtimeInboundAdapter()
    return [event async for event in adapter.events(_aiter(frames))]


@pytest.mark.asyncio
async def test_session_input_audio_format_pcm16_converts_appends_without_per_frame_format():
    # The OpenBMB demo/bridge declare pcm16 once on session.update and send
    # appends with NO per-frame format. The session-level input_audio_format
    # must drive the pcm16->f32le conversion; otherwise pcm16 bytes reach the
    # model reinterpreted as float32 and the audio encoder emits NaN.
    adapter = RealtimeInboundAdapter(input_audio_format="pcm16")
    # 100 pcm16 samples (200 bytes) -> 100 f32le samples (400 bytes) = one chunk.
    pcm16 = base64.b64encode(b"\x01\x00" * 100).decode("ascii")
    frame = {"type": "input_audio_buffer.append", "audio": pcm16, "sample_rate_hz": _SR}
    events = [event async for event in adapter.events(_aiter([frame]))]

    assert len(events) == 2
    assert events[0] == InputStarted(modality="audio")
    chunk = events[1]
    assert isinstance(chunk, InputChunk)
    # 100 float32 samples of 1/32768 -> 400 bytes, NOT the raw pcm16 bytes.
    assert len(chunk.data) == 400
    import struct

    values = struct.unpack("<100f", chunk.data)
    assert all(abs(v - (1.0 / 32768.0)) < 1e-6 for v in values)


@pytest.mark.asyncio
async def test_per_frame_format_overrides_session_default():
    # An explicit per-frame format still wins over the session default.
    adapter = RealtimeInboundAdapter(input_audio_format="pcm16")
    f32 = base64.b64encode(b"\x00\x00\x00\x00" * 100).decode("ascii")
    frame = {
        "type": "input_audio_buffer.append",
        "audio": f32,
        "format": "pcm_f32le",
        "sample_rate_hz": _SR,
    }
    events = [event async for event in adapter.events(_aiter([frame]))]
    assert events[-1] == InputChunk(data=b"\x00\x00\x00\x00" * 100, modality="audio")


@pytest.mark.asyncio
async def test_partial_then_whole_chunk_emits_started_and_one_input_chunk():
    # 60 + 60 samples -> a single 100-sample model chunk (400 bytes), 20 buffered.
    events = await _collect([_append(60), _append(60)])

    assert events == [
        InputStarted(modality="audio"),
        InputChunk(data=b"\x00\x00\x00\x00" * 100, modality="audio"),
    ]


@pytest.mark.asyncio
async def test_commit_flushes_residual_and_emits_input_committed():
    # 60 samples buffered (< one chunk); commit zero-pads to 100 and commits.
    events = await _collect(
        [
            _append(60),
            {"type": "input_audio_buffer.commit"},
        ]
    )

    assert events == [
        InputStarted(modality="audio"),
        InputChunk(data=b"\x00\x00\x00\x00" * 100, modality="audio"),
        InputCommitted(),
    ]


@pytest.mark.asyncio
async def test_response_cancel_maps_to_interrupt_requested():
    events = await _collect([{"type": "response.cancel"}])
    assert events == [InterruptRequested(reason="response.cancel")]


@pytest.mark.asyncio
async def test_playback_ack_ms_converts_to_sample_cursor_at_24k():
    events = await _collect([{"type": "playback.ack", "played_ms": 10, "committed_ms": 5}])
    # 10 ms * 24000 / 1000 = 240 samples; 5 ms -> 120 samples.
    assert events == [PlaybackAcknowledged(cursor=240, committed_cursor=120)]


@pytest.mark.asyncio
async def test_playback_ack_samples_are_used_directly():
    events = await _collect([{"type": "audio.playback_ack", "played_samples": 480}])
    assert events == [PlaybackAcknowledged(cursor=480, committed_cursor=None)]


@pytest.mark.asyncio
async def test_session_close_emits_close_and_stops_iteration():
    events = await _collect(
        [
            {"type": "session.close", "reason": "bye"},
            {"type": "input_audio_buffer.append", "audio": _pcm(200), "sample_rate_hz": _SR},
        ]
    )
    # Everything after the close frame is ignored.
    assert events == [SessionCloseRequested(reason="bye")]


def _pcm16(samples) -> str:
    import array

    a = array.array("h", samples)
    return base64.b64encode(a.tobytes()).decode("ascii")


@pytest.mark.asyncio
async def test_pcm16_append_is_converted_to_f32le_input_chunk():
    # 100 int16 samples at 100 Hz -> exactly one model chunk; 16384 -> 0.5 float.
    frame = {
        "type": "input_audio_buffer.append",
        "audio": _pcm16([16384] * 100),
        "format": "pcm16",
        "sample_rate_hz": _SR,
        "is_speech": True,
    }
    events = await _collect([frame])

    assert events[0] == InputStarted(modality="audio")
    chunk = events[1]
    assert isinstance(chunk, InputChunk)
    # 100 float32 samples = 400 bytes; first sample decodes to 0.5.
    assert len(chunk.data) == 400
    import struct

    assert struct.unpack_from("<f", chunk.data, 0)[0] == 0.5
