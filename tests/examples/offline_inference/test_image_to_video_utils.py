# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from PIL import Image

from examples.offline_inference.image_to_video.image_to_video import (
    _detect_ltx_checkpoint_version,
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
        "num_inference_steps": 8,
        "fps": 24,
        "output": "ltx25_output.mp4",
    }
