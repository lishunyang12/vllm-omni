# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from PIL import Image

from examples.offline_inference.image_to_video.image_to_video import (
    _detect_ltx_checkpoint_version,
    _is_ltx2_two_stage_pipeline,
    _ltx25_i2v_defaults,
    prepare_primary_image,
)
from examples.offline_inference.text_to_video.text_to_video import _detect_preset

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_prepare_primary_image_can_defer_resize_to_pipeline():
    image = Image.new("RGB", (17, 9))

    prepared = prepare_primary_image(
        image,
        width=8,
        height=6,
        defer_resize_to_pipeline=True,
    )

    assert prepared is image
    assert prepared.size == (17, 9)


def test_prepare_primary_image_preserves_legacy_lanczos_resize():
    image = Image.new("RGB", (17, 9))

    prepared = prepare_primary_image(image, width=8, height=6)

    assert prepared is not image
    assert prepared.size == (8, 6)


def test_ltx25_local_checkpoint_uses_metadata_driven_example_defaults(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "LTX2Pipeline",
                "text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"],
            }
        )
    )

    assert _detect_ltx_checkpoint_version(str(tmp_path)) == "2.5"
    assert _detect_preset(str(tmp_path)) == {
        "height": 544,
        "width": 960,
        "num_frames": 121,
        "num_inference_steps": 30,
        "fps": 24,
        "output": "ltx25_full_output.mp4",
    }


@pytest.mark.parametrize("model_class_name", ["LTX2TwoStagePipeline", "LTX2DistilledPipeline"])
def test_ltx25_i2v_two_stage_defaults_accept_canonical_and_compatibility_names(model_class_name):
    assert _is_ltx2_two_stage_pipeline(model_class_name)
    assert _ltx25_i2v_defaults("2.5", model_class_name) == (24, None, 121, 8, None, 1088 * 1920, 64)


def test_ltx25_i2v_one_stage_defaults_remain_low_resolution():
    assert not _is_ltx2_two_stage_pipeline("LTX2Pipeline")
    assert _ltx25_i2v_defaults("2.5", "LTX2Pipeline") == (24, None, 121, 30, None, 544 * 960, 32)
    assert _ltx25_i2v_defaults("2.3", "LTX2Pipeline") is None


def test_ltx25_i2v_two_stage_full_uses_full_schedule():
    assert _ltx25_i2v_defaults("2.5", "LTX2TwoStagePipeline", "full") == (
        24,
        None,
        121,
        30,
        None,
        1088 * 1920,
        64,
    )
