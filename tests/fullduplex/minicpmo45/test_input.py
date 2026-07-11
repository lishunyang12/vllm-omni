# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64

import numpy as np

from vllm_omni.experimental.fullduplex.minicpmo45.input import MiniCPMO45PcmAppendBuffer


def pcm_payload(samples: int, *, speech: bool = True) -> dict[str, object]:
    audio = np.ones(samples, dtype=np.float32).tobytes()
    return {
        "type": "audio",
        "audio": base64.b64encode(audio).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate_hz": 16_000,
        "is_speech": speech,
    }


def test_commit_does_not_add_silence_after_incremental_audio_was_drained():
    buffer = MiniCPMO45PcmAppendBuffer()

    emitted = buffer.append(pcm_payload(16_000), chunk_period_ms=1_000)
    committed = buffer.commit(chunk_period_ms=1_000)

    assert emitted is not None
    assert not buffer.has_pending()
    assert committed.had_input is True
    assert committed.had_speech is True
    assert committed.payload is None


def test_commit_without_speech_does_not_synthesize_terminal_audio():
    buffer = MiniCPMO45PcmAppendBuffer()
    buffer.append(pcm_payload(8_000, speech=False), chunk_period_ms=1_000)

    committed = buffer.commit(chunk_period_ms=1_000)

    assert committed.had_input is True
    assert committed.had_speech is False
    assert committed.payload is None


def test_commit_resets_cumulative_turn_accounting():
    buffer = MiniCPMO45PcmAppendBuffer()
    buffer.append(pcm_payload(16_000), chunk_period_ms=1_000)
    buffer.commit(chunk_period_ms=1_000)

    empty = buffer.commit(chunk_period_ms=1_000)

    assert empty.had_input is False
    assert empty.had_speech is False
    assert empty.payload is None
