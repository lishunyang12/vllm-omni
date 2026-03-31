# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for DiffusionWorker CUDA (nsys) profiler integration.

Verifies that DiffusionWorker correctly creates a CudaProfilerWrapper
when profiler_config.profiler == "cuda", enabling Nsight Systems profiling.
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def mock_od_config():
    """Create a mock OmniDiffusionConfig with cuda profiler."""
    config = MagicMock()
    config.num_gpus = 1
    config.master_port = 12345
    config.enable_sleep_mode = False
    config.cache_backend = None
    config.cache_config = None
    config.model = "test-model"
    config.diffusion_load_format = "default"
    config.max_cpu_loras = 0
    config.lora_path = None
    config.lora_scale = 1.0

    # Profiler config for cuda (nsys)
    profiler_config = MagicMock()
    profiler_config.profiler = "cuda"
    profiler_config.delay_iterations = 0
    profiler_config.max_iterations = 0
    config.profiler_config = profiler_config

    # Parallel config
    parallel_config = MagicMock()
    parallel_config.tensor_parallel_size = 1
    parallel_config.data_parallel_size = 1
    parallel_config.enable_expert_parallel = False
    config.parallel_config = parallel_config

    return config


class TestDiffusionWorkerCudaProfiler:
    """Test that DiffusionWorker creates CudaProfilerWrapper for nsys."""

    def test_cuda_profiler_created(self, mock_od_config):
        """DiffusionWorker should create CudaProfilerWrapper when profiler == 'cuda'."""
        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

        with (
            patch.object(DiffusionWorker, "init_device"),
            patch.object(DiffusionWorker, "load_model"),
            patch.object(DiffusionWorker, "init_lora_manager"),
        ):
            worker = DiffusionWorker(local_rank=0, rank=0, od_config=mock_od_config, skip_load_model=True)

        from vllm.profiler.wrapper import CudaProfilerWrapper

        assert worker.profiler is not None
        assert isinstance(worker.profiler, CudaProfilerWrapper)

    def test_torch_profiler_not_created_for_cuda(self, mock_od_config):
        """CudaProfilerWrapper should not be an OmniTorchProfilerWrapper."""
        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

        with (
            patch.object(DiffusionWorker, "init_device"),
            patch.object(DiffusionWorker, "load_model"),
            patch.object(DiffusionWorker, "init_lora_manager"),
        ):
            worker = DiffusionWorker(local_rank=0, rank=0, od_config=mock_od_config, skip_load_model=True)

        from vllm_omni.profiler import OmniTorchProfilerWrapper

        assert not isinstance(worker.profiler, OmniTorchProfilerWrapper)

    def test_no_profiler_when_none(self, mock_od_config):
        """DiffusionWorker should have no profiler when profiler_config is None."""
        mock_od_config.profiler_config = None

        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

        with (
            patch.object(DiffusionWorker, "init_device"),
            patch.object(DiffusionWorker, "load_model"),
            patch.object(DiffusionWorker, "init_lora_manager"),
        ):
            worker = DiffusionWorker(local_rank=0, rank=0, od_config=mock_od_config, skip_load_model=True)

        assert worker.profiler is None

    def test_profile_start_stop_with_cuda_profiler(self, mock_od_config):
        """profile() should call start/stop on CudaProfilerWrapper without errors."""
        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

        with (
            patch.object(DiffusionWorker, "init_device"),
            patch.object(DiffusionWorker, "load_model"),
            patch.object(DiffusionWorker, "init_lora_manager"),
        ):
            worker = DiffusionWorker(local_rank=0, rank=0, od_config=mock_od_config, skip_load_model=True)

        # Mock the underlying cuda profiler methods to avoid actual CUDA calls
        worker.profiler._cuda_profiler = MagicMock()

        # Should not raise
        worker.profile(is_start=True)
        worker.profile(is_start=False)

        # Verify start and stop were called
        worker.profiler._cuda_profiler.start.assert_called_once()
        worker.profiler._cuda_profiler.stop.assert_called_once()
