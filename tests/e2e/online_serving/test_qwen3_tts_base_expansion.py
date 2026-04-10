# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-TTS model with text input and audio output.

These tests verify the /v1/audio/speech endpoint works correctly with
actual model inference, not mocks.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.conftest import OmniServerParams
from tests.utils import get_deploy_config_path, hardware_test

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

REF_AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


def get_stage_config(name: str = "qwen3_tts.yaml") -> str:
    """Resolve a deploy config path under vllm_omni/deploy/."""
    return get_deploy_config_path(name)


# JSON for --stage-overrides that mirrors the deleted qwen3_tts_no_async_chunk.yaml
# stage settings. Paired with --no-async-chunk + --pipeline qwen3_tts_no_async_chunk.
_NO_ASYNC_CHUNK_STAGE_OVERRIDES = (
    '{"0":{"max_num_seqs":1,"gpu_memory_utilization":0.2,"enforce_eager":true,'
    '"async_scheduling":false},'
    '"1":{"gpu_memory_utilization":0.2,"async_scheduling":false}}'
)


def get_prompt(prompt_type="text"):
    """Text prompt for text-to-audio tests (same as test_qwen3_omni - beijing test case)."""
    prompts = {
        "text": "The weather is nice today, perfect for a walk in the park.",
    }
    return prompts.get(prompt_type, prompts["text"])


def get_max_batch_size(size_type="few"):
    """Batch size for concurrent requests (same as test_qwen3_omni)."""
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_stage_config("qwen3_tts.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="async_chunk",
    ),
    # Synchronous (no async-chunk) variant — composed entirely from CLI flags
    # against the bundled qwen3_tts.yaml prod default. Replaces the deleted
    # qwen3_tts_no_async_chunk.yaml file; same effective config.
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_stage_config("qwen3_tts.yaml"),
            server_args=[
                "--trust-remote-code",
                "--disable-log-stats",
                "--no-async-chunk",
                "--pipeline",
                "qwen3_tts_no_async_chunk",
                "--stage-overrides",
                _NO_ASYNC_CHUNK_STAGE_OVERRIDES,
            ],
        ),
        id="no_async_chunk",
    ),
]


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_streaming_001(omni_server, openai_client) -> None:
    """
    Test text input processing and audio output via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=True
    Datasets: few requests
    """

    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "response_format": "wav",
        "task_type": "Base",
        "voice": "clone",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config, request_num=get_max_batch_size("few"))


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_response_format_001(omni_server, openai_client) -> None:
    """
    Test text input processing and audio output via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: non-stream PCM
    Datasets: few requests
    """

    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "response_format": "pcm",
        "task_type": "Base",
        "voice": "clone",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)
