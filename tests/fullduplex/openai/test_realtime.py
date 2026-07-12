# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64

from vllm_omni.experimental.fullduplex.core.events import (
    EmitProtocolEvent,
    ModelAudioDelta,
    ModelTextDelta,
    ProtocolEventKind,
)
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.openai.realtime import (
    DemoRealtimeProjector,
    RealtimeEventProjector,
)


def test_realtime_projection_uses_effect_fence_without_mutating_identity():
    fence = DuplexFence("s", epoch=2, turn_id=3, response_seq=4)
    projector = RealtimeEventProjector()

    events = projector.project(
        EmitProtocolEvent(
            fence=fence,
            kind=ProtocolEventKind.RESPONSE_STARTED,
        )
    )

    assert events == (
        {
            "type": "response.created",
            "response": {
                "id": "resp-s-e2-t3-r4",
                "status": "in_progress",
            },
        },
    )
    assert fence == DuplexFence("s", epoch=2, turn_id=3, response_seq=4)


def test_realtime_projection_rejects_late_audio_via_shared_lifecycle():
    fence = DuplexFence("s", turn_id=1, response_seq=1)
    projector = RealtimeEventProjector()
    projector.project(EmitProtocolEvent(fence, ProtocolEventKind.RESPONSE_STARTED))
    projector.project(EmitProtocolEvent(fence, ProtocolEventKind.RESPONSE_COMPLETED))

    try:
        projector.project(EmitProtocolEvent(fence, ProtocolEventKind.AUDIO_DELTA, payload=b"late"))
    except Exception as exc:
        assert type(exc).__name__ == "LateResponseOutputError"
    else:
        raise AssertionError("late audio must fail")


def test_text_delta_reads_text_from_event_payload():
    fence = DuplexFence("s", turn_id=1, response_seq=1)
    projector = RealtimeEventProjector()
    projector.project(EmitProtocolEvent(fence, ProtocolEventKind.RESPONSE_STARTED))

    # The reducer passes the ModelTextDelta event object as the payload.
    events = projector.project(
        EmitProtocolEvent(fence, ProtocolEventKind.TEXT_DELTA, payload=ModelTextDelta(text="hello", fence=fence))
    )

    assert events == ({"type": "response.output_text.delta", "response_id": "resp-s-e0-t1-r1", "delta": "hello"},)


def test_audio_delta_reads_bytes_from_event_payload_and_base64_encodes():
    fence = DuplexFence("s", turn_id=1, response_seq=1)
    projector = RealtimeEventProjector()
    projector.project(EmitProtocolEvent(fence, ProtocolEventKind.RESPONSE_STARTED))

    pcm = b"\x01\x02\x03\x04"
    events = projector.project(
        EmitProtocolEvent(
            fence,
            ProtocolEventKind.AUDIO_DELTA,
            payload=ModelAudioDelta(data=pcm, generated_cursor=1, sent_cursor=1, fence=fence),
        )
    )

    assert events == (
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-s-e0-t1-r1",
            "delta": base64.b64encode(pcm).decode("ascii"),
        },
    )


def test_demo_projector_uses_compact_dialect_for_appjs():
    fence = DuplexFence("s", turn_id=1, response_seq=1)
    p = DemoRealtimeProjector()

    started = p.project(EmitProtocolEvent(fence, ProtocolEventKind.RESPONSE_STARTED))
    text = p.project(
        EmitProtocolEvent(fence, ProtocolEventKind.TEXT_DELTA, payload=ModelTextDelta(text="hi", fence=fence))
    )
    pcm = b"\x01\x02\x03\x04"
    audio = p.project(
        EmitProtocolEvent(
            fence,
            ProtocolEventKind.AUDIO_DELTA,
            payload=ModelAudioDelta(data=pcm, generated_cursor=1, sent_cursor=1, fence=fence),
        )
    )
    done = p.project(EmitProtocolEvent(fence, ProtocolEventKind.RESPONSE_COMPLETED))

    assert started == ({"type": "response.created", "response": {"id": "resp-s-e0-t1-r1", "status": "in_progress"}},)
    assert text == ({"type": "response.audio_transcript.delta", "response_id": "resp-s-e0-t1-r1", "delta": "hi"},)
    assert audio == (
        {
            "type": "response.audio.delta",
            "response_id": "resp-s-e0-t1-r1",
            "delta": base64.b64encode(pcm).decode("ascii"),
            "format": "pcm_f32le",
            "sample_rate_hz": 24000,
        },
    )
    assert done == (
        {"type": "response.audio_transcript.done", "response_id": "resp-s-e0-t1-r1", "transcript": "hi"},
        {"type": "response.done", "response": {"id": "resp-s-e0-t1-r1", "status": "completed"}},
    )


def test_demo_projector_listen_and_segment_names():
    fence = DuplexFence("s", turn_id=1, response_seq=1)
    p = DemoRealtimeProjector()
    p.project(EmitProtocolEvent(fence, ProtocolEventKind.RESPONSE_STARTED))
    assert p.project(EmitProtocolEvent(fence, ProtocolEventKind.SEGMENT_ENDED)) == (
        {"type": "response.audio.done", "response_id": "resp-s-e0-t1-r1"},
    )
    assert p.project(EmitProtocolEvent(fence, ProtocolEventKind.MODEL_LISTENING)) == (
        {"type": "response.listen", "response_id": "resp-s-e0-t1-r1"},
    )
