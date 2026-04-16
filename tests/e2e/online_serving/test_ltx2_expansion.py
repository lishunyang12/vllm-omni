"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the LTX2 model.
- Lightricks/LTX-2
- rootonchair/LTX-2-19b-distilled

Coverage:
- Cache-DiT
- CFG-Parallel
- Ulysses-SP
- Ring-Attn

assert_diffusion_response validates successful generation
"""

import os

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    generate_synthetic_image,
)
from tests.utils import hardware_marks

PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
NEGATIVE_PROMPT = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

LTX2_MODELS = [
    ("Lightricks/LTX-2", "LTX2Pipeline"),
    ("Lightricks/LTX-2", "LTX2ImageToVideoPipeline"),
    ("rootonchair/LTX-2-19b-distilled", "LTX2TwoStagesPipeline"),
    ("rootonchair/LTX-2-19b-distilled", "LTX2ImageToVideoTwoStagesPipeline"),
]

PARALLEL_CONFIGS = [
    ("cfg_parallel", ["--cfg-parallel-size", "2"]),
    ("ulysses_sp", ["--usp", "2"]),
    ("ring_atten", ["--ring", "2"]),
]


def _get_ltx2_feature_cases():
    cases = []

    # Single-card: Cache-DiT (applies to all models)
    for model_path, model_cls_name in LTX2_MODELS:
        cases.append(
            pytest.param(
                OmniServerParams(
                    model=model_path,
                    model_class_name=model_cls_name,
                    server_args=["--cache-backend", "cache_dit"],
                ),
                id=f"{model_cls_name}_cache_dit",
                marks=SINGLE_CARD_FEATURE_MARKS,
            )
        )

    # Multi-card features
    for model_path, model_cls_name in LTX2_MODELS:
        for feat_id, server_args in PARALLEL_CONFIGS:
            cases.append(
                pytest.param(
                    OmniServerParams(
                        model=model_path,
                        model_class_name=model_cls_name,
                        server_args=server_args.extend(["--model-class-name", model_cls_name]),
                    ),
                    id=f"{model_cls_name}_{feat_id}",
                    marks=PARALLEL_FEATURE_MARKS,
                )
            )

    return cases


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_ltx2_feature_cases(),
    indirect=True,
)
def test_ltx2_diffusion_features(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    model_path = omni_server.model
    model_name = os.path.basename(os.path.normpath(model_path))
    model_class_name = omni_server.model_class_name
    is_distilled_model = model_name == "LTX-2-19b-distilled"
    is_i2v = "ImageToVideo" in model_class_name

    form_data = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": 768,
        "width": 512,
        "num_inference_steps": 2,
        "guidance_scale": 1.0 if is_distilled_model else 4.0,
        "seed": 42,
    }

    request_config = {
        "model": model_path,
        "form_data": form_data,
        "model_class_name": model_class_name,  # use for validate diffusion response for two-stages pipeline
    }

    if is_i2v:
        request_config["image_reference"] = f"data:image/jpeg;base64,{generate_synthetic_image(758, 512)['base64']}"

    openai_client.send_video_diffusion_request(request_config)
