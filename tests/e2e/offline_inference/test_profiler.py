# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline E2E test for Z-Image diffusion profiling.

Verifies that:
- Profiling can be started and stopped via Omni interface
- Trace files are generated on disk
- Generation works correctly under profiling
"""

import os
import sys
from pathlib import Path

import pytest
import torch

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.utils.platform_utils import is_npu, is_rocm

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Use only fast, publicly available Z-Image models that support profiling
# These are known to work well on CUDA and are suitable for CI
models = ["Tongyi-MAI/Z-Image-Turbo"]

if is_npu():
    pytest.skip("Profiling test not yet supported on NPU", allow_module_level=True)
elif is_rocm():
    models = ["Tongyi-MAI/Z-Image-Turbo"]


@pytest.mark.parametrize("model_name", models)
def test_z_image_profiler(model_name: str, tmp_path: Path) -> None:
    """
    E2E test: Run Z-Image generation with profiling enabled and verify trace output.

    Args:
        model_name: Name of the Z-Image model to test
        tmp_path: Pytest temporary directory for trace files
    """
    # Use a temporary profile directory to avoid clutter
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    os.environ["VLLM_OMNI_DIFFUSION_PROFILE_DIR"] = str(profile_dir)

    m = Omni(model=model_name)

    # Start profiling with a deterministic name
    trace_name = "z_image_profiler_test"
    m.engine.start_profile(trace_name)

    try:
        # Run generation under profiling — fast settings to avoid OOM/timeout
        height = 256
        width = 256
        outputs = m.generate(
            prompt="a cute robot waving hello",
            height=height,
            width=width,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(42),
            num_outputs_per_prompt=1,
        )

        # Basic output validation (same style as official test)
        first_output = outputs[0]
        assert first_output.final_output_type == "image"

        if not hasattr(first_output, "request_output") or not first_output.request_output:
            raise ValueError("No request_output found in OmniRequestOutput")

        req_out = first_output.request_output[0]
        images = req_out.images

        assert len(images) == 1
        assert images[0].width == width
        assert images[0].height == height

        # Optional: save for manual inspection
        images[0].save(tmp_path / "image_output.png")

        # Stop profiling and get trace file paths
        trace_files = m.engine.stop_profile()

        # Verify trace files were created
        assert len(trace_files) == 1  # Single GPU in test
        trace_path = Path(trace_files[0])

        assert trace_path.exists()
        assert trace_path.parent == profile_dir
        assert trace_path.name == f"{trace_name}_rank0.json"
        assert trace_path.stat().st_size > 5000  # Contains real kernel data

        print(f"Profiling successful! Trace saved: {trace_path}")

    finally:
        # Always close the engine to clean up workers
        m.engine.close()
