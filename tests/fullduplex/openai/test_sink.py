# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64

import pytest

from vllm_omni.experimental.fullduplex.core.events import (
    EmitProtocolEvent,
    ModelAudioDelta,
    ModelTextDelta,
    ProtocolEventKind,
)
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.openai.sink import WebSocketRealtimeSink


@pytest.mark.asyncio
async def test_sink_projects_and_sends_frames_in_order():
    sent: list[dict] = []

    async def send_json(frame):
        sent.append(frame)

    sink = WebSocketRealtimeSink(send_json)
    fence = DuplexFence("s", turn_id=1, response_seq=1)

    await sink.emit(EmitProtocolEvent(fence, ProtocolEventKind.RESPONSE_STARTED, payload=None))
    await sink.emit(
        EmitProtocolEvent(fence, ProtocolEventKind.TEXT_DELTA, payload=ModelTextDelta(text="hi", fence=fence))
    )
    pcm = b"\x00\x00\x00\x00"
    await sink.emit(
        EmitProtocolEvent(
            fence,
            ProtocolEventKind.AUDIO_DELTA,
            payload=ModelAudioDelta(data=pcm, generated_cursor=1, sent_cursor=1, fence=fence),
        )
    )

    assert sent == [
        {"type": "response.created", "response": {"id": "resp-s-e0-t1-r1", "status": "in_progress"}},
        {"type": "response.output_text.delta", "response_id": "resp-s-e0-t1-r1", "delta": "hi"},
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-s-e0-t1-r1",
            "delta": base64.b64encode(pcm).decode("ascii"),
        },
    ]
