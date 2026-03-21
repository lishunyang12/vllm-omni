# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following models:
- HunyuanVideo-1.5-T2V (480p)

Coverage (2x H100, since model cannot fit 4x L4):
- TeaCache + Layerwise CPU offloading (1 GPU)
- CacheDiT + Ulysses SP=2 (2 GPUs)
- CacheDiT + Ring SP=2 (2 GPUs)
- TeaCache + CFG-Parallel=2 (2 GPUs)
- CacheDiT + TP=2 + VAE patch parallel=2 (2 GPUs)
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)
from tests.utils import hardware_marks

PROMPT = "A cat walking across a sunlit garden, cinematic lighting, slow motion."
NEGATIVE_PROMPT = "low quality, blurry, distorted"

MODEL = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_diffusion_feature_cases(model: str):
    """Return L4 diffusion feature cases for HunyuanVideo-1.5.

    Designed for 2x H100 environment per issue #1832.
    """
    return [
        # (1 GPU) TeaCache + Layerwise CPU offloading
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "tea_cache",
                    "--enable-layerwise-offload",
                ],
            ),
            id="single_card_teacache_layerwise",
            marks=SINGLE_CARD_MARKS,
        ),
        # (2 GPUs) CacheDiT + Ulysses SP=2
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ulysses-degree",
                    "2",
                ],
            ),
            id="parallel_cachedit_ulysses_2",
            marks=PARALLEL_MARKS,
        ),
        # (2 GPUs) CacheDiT + Ring SP=2
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ring",
                    "2",
                ],
            ),
            id="parallel_cachedit_ring_2",
            marks=PARALLEL_MARKS,
        ),
        # (2 GPUs) TeaCache + CFG-Parallel=2
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "tea_cache",
                    "--cfg-parallel-size",
                    "2",
                ],
            ),
            id="parallel_teacache_cfg_2",
            marks=PARALLEL_MARKS,
        ),
        # (2 GPUs) CacheDiT + TP=2 + VAE patch parallel=2
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--tensor-parallel-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_cachedit_tp2_vae2",
            marks=PARALLEL_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_hunyuan_video_15_t2v(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """L4 diffusion feature coverage for HunyuanVideo-1.5-T2V on H100.

    Exercises TeaCache, CacheDiT, layerwise offload, Ulysses SP,
    Ring SP, CFG-Parallel, TP, and VAE patch parallel.
    """
    messages = dummy_messages_from_mix_data(content_text=PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 480,
            "width": 640,
            "num_frames": 5,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
