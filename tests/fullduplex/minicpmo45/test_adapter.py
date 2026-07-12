# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64

from vllm_omni.experimental.fullduplex.core.events import (
    ModelAudioDelta,
    ModelListening,
    ModelSegmentEnded,
    ModelSpeaking,
    ModelTextDelta,
    ModelTurnEnded,
)
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.minicpmo45.adapter import MiniCPMO45ModelEventAdapter


def _pcm(num_samples: int) -> str:
    """Base64 of ``num_samples`` little-endian float32 samples, as the worker emits."""
    return base64.b64encode(b"\x00\x00\x00\x00" * num_samples).decode("ascii")


def test_listen_output_maps_to_model_listening_with_same_fence():
    fence = DuplexFence("s", epoch=2, turn_id=3, response_seq=4)

    events = MiniCPMO45ModelEventAdapter().map_output({"is_listen": True}, fence)

    assert events == (ModelListening(fence=fence),)


def test_speaking_segment_emits_speaking_text_and_segment_end_only():
    fence = DuplexFence("s", turn_id=1, response_seq=1)

    events = MiniCPMO45ModelEventAdapter().map_output(
        {
            "is_listen": False,
            "text": "first sentence",
            "tts_is_last_chunk": True,
            "turn_end": False,
        },
        fence,
    )

    assert events == (
        ModelSpeaking(fence=fence),
        ModelTextDelta(text="first sentence", fence=fence),
        ModelSegmentEnded(fence=fence),
    )
    assert not any(isinstance(event, ModelTurnEnded) for event in events)


def test_only_model_turn_end_metadata_finishes_response():
    fence = DuplexFence("s", turn_id=1, response_seq=1)

    events = MiniCPMO45ModelEventAdapter().map_output(
        {
            "is_listen": False,
            "text": "",
            "tts_is_last_chunk": True,
            "turn_end": True,
        },
        fence,
    )

    assert events[-2:] == (
        ModelSegmentEnded(fence=fence),
        ModelTurnEnded(reason="model_turn_eos", fence=fence),
    )


def test_audio_data_emits_audio_delta_after_text_before_segment_end():
    fence = DuplexFence("s", turn_id=1, response_seq=1)

    events = MiniCPMO45ModelEventAdapter().map_output(
        {
            "is_listen": False,
            "text": "hi",
            "audio_data": _pcm(240),
            "tts_is_last_chunk": True,
        },
        fence,
    )

    assert events == (
        ModelSpeaking(fence=fence),
        ModelTextDelta(text="hi", fence=fence),
        ModelAudioDelta(data=b"\x00\x00\x00\x00" * 240, generated_cursor=240, sent_cursor=240, fence=fence),
        ModelSegmentEnded(fence=fence),
    )


def test_audio_cursor_accumulates_across_chunks_and_resets_after_turn_end():
    fence = DuplexFence("s", turn_id=1, response_seq=1)
    adapter = MiniCPMO45ModelEventAdapter()

    first = adapter.map_output({"is_listen": False, "audio_data": _pcm(100)}, fence)
    second = adapter.map_output({"is_listen": False, "audio_data": _pcm(50)}, fence)

    assert first == (
        ModelSpeaking(fence=fence),
        ModelAudioDelta(data=b"\x00\x00\x00\x00" * 100, generated_cursor=100, sent_cursor=100, fence=fence),
    )
    # Cumulative: the second delta advances the cursor to 150, not back to 50.
    assert second == (
        ModelAudioDelta(data=b"\x00\x00\x00\x00" * 50, generated_cursor=150, sent_cursor=150, fence=fence),
    )

    adapter.map_output({"is_listen": False, "turn_end": True}, fence)
    # A new turn on the same fence restarts the cursor from zero.
    third = adapter.map_output({"is_listen": False, "audio_data": _pcm(30)}, fence)
    assert third == (
        ModelSpeaking(fence=fence),
        ModelAudioDelta(data=b"\x00\x00\x00\x00" * 30, generated_cursor=30, sent_cursor=30, fence=fence),
    )


def test_empty_or_absent_audio_data_yields_no_audio_delta():
    fence = DuplexFence("s", turn_id=1, response_seq=1)
    adapter = MiniCPMO45ModelEventAdapter()

    absent = adapter.map_output({"is_listen": False, "text": "x"}, fence)
    empty = adapter.map_output({"is_listen": False, "audio_data": ""}, fence)

    assert not any(isinstance(event, ModelAudioDelta) for event in absent)
    assert not any(isinstance(event, ModelAudioDelta) for event in empty)
