# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online serving smoke for the LTX-2.5 Diffusers checkpoint."""

import os

import pytest
import requests

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

DEFAULT_MODEL = "Lightricks/LTX-2.5-Diffusers"
DEFAULT_REVISION = "a6de4b5354f078db24d9cf4778c14846788aea3d"
MODEL = os.getenv("VLLM_TEST_LTX25_MODEL", DEFAULT_MODEL)
MODEL_REVISION = os.getenv("VLLM_TEST_LTX25_MODEL_REVISION", DEFAULT_REVISION if MODEL == DEFAULT_MODEL else "")
PROMPT = "A red fox walks through a snowy forest while the camera remains fixed."

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]
SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})


def _cases():
    return [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    *(["--revision", MODEL_REVISION] if MODEL_REVISION else []),
                    "--model-class-name",
                    "LTX2TwoStagePipeline",
                    "--enforce-eager",
                    "--enable-layerwise-offload",
                    "--diffusion-attention-backend",
                    "CUDNN_ATTN",
                ],
            ),
            id="default_distilled_two_stage_pinned_sync",
            marks=SINGLE_CARD_MARKS,
        )
    ]


@pytest.mark.parametrize("omni_server", _cases(), indirect=True)
def test_ltx25_sync_video(omni_server: OmniServer) -> None:
    """Serve the default distilled two-stage pipeline and return an MP4."""
    response = requests.post(
        f"http://{omni_server.host}:{omni_server.port}/v1/videos/sync",
        data={
            "model": omni_server.model,
            "prompt": PROMPT,
            "height": "128",
            "width": "128",
            "num_frames": "9",
            "fps": "24",
            # LTX-2.5 defaults to the official eight-step schedule.
            "num_inference_steps": "8",
            "seed": "42",
        },
        timeout=1800,
    )

    response.raise_for_status()
    assert response.headers["content-type"].startswith("video/mp4")
    assert b"ftyp" in response.content[:32]
