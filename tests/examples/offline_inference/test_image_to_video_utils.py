# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from PIL import Image

from examples.offline_inference.image_to_video.image_to_video import prepare_primary_image

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
