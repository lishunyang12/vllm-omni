# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import os
import tempfile

import pytest
import torch

from vllm_omni.profiler import ProfilerConfig, TorchProfiler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTorchProfiler:
    """Tests for TorchProfiler (require CUDA)."""

    def _make_config(self, tmpdir: str) -> ProfilerConfig:
        return ProfilerConfig(profiler="torch", torch_profiler_dir=tmpdir)

    def test_start_stop_lifecycle(self):
        """Test basic start/stop lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            profiler = TorchProfiler(config, worker_name="test")

            assert not profiler.is_running
            profiler.start()
            assert profiler.is_running

            # Do some work
            x = torch.randn(100, 100, device="cuda")
            y = x @ x.T
            del x, y

            profiler.stop()
            assert not profiler.is_running

    def test_stop_without_start_is_noop(self):
        """Test stop without start is safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            profiler = TorchProfiler(config, worker_name="test")
            profiler.stop()  # Should not raise
            assert not profiler.is_running

    def test_double_start_is_noop(self):
        """Test that calling start twice does not error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            profiler = TorchProfiler(config, worker_name="test")
            profiler.start()
            profiler.start()  # Second call should be no-op
            assert profiler.is_running
            profiler.stop()
            assert not profiler.is_running

    def test_shutdown(self):
        """Test shutdown stops a running profiler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            profiler = TorchProfiler(config, worker_name="test")
            profiler.start()
            assert profiler.is_running
            profiler.shutdown()
            assert not profiler.is_running

    def test_step_basic(self):
        """Test step method doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            profiler = TorchProfiler(config, worker_name="test")
            profiler.start()
            for _ in range(5):
                profiler.step()
            profiler.stop()

    def test_delay_iterations(self):
        """Test that profiler doesn't start until delay iterations pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=tmpdir,
                delay_iterations=3,
            )
            profiler = TorchProfiler(config, worker_name="test")
            profiler.start()
            # Active but not yet running due to delay
            assert not profiler.is_running
            for _ in range(2):
                profiler.step()
            assert not profiler.is_running
            # Third step triggers actual start
            profiler.step()
            assert profiler.is_running
            profiler.stop()

    def test_max_iterations(self):
        """Test that profiler stops after max iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=tmpdir,
                max_iterations=2,
            )
            profiler = TorchProfiler(config, worker_name="test")
            profiler.start()
            assert profiler.is_running
            profiler.step()  # iter 1
            assert profiler.is_running
            profiler.step()  # iter 2
            assert profiler.is_running
            profiler.step()  # iter 3 -> exceeds max, stops
            assert not profiler.is_running

    def test_config_driven_settings(self):
        """Test that config fields are passed to the profiler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=tmpdir,
                torch_profiler_record_shapes=True,
                torch_profiler_with_memory=True,
                torch_profiler_with_flops=True,
            )
            profiler = TorchProfiler(config, worker_name="test")
            profiler.start()
            x = torch.randn(10, 10, device="cuda")
            _ = x @ x.T
            profiler.stop()

    def test_trace_files_written(self):
        """Test that stop() produces trace files in torch_profiler_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            profiler = TorchProfiler(config, worker_name="test-trace")
            profiler.start()
            x = torch.randn(64, 64, device="cuda")
            _ = x @ x.T
            del x
            profiler.stop()

            # Verify trace file exists and is non-empty
            trace_files = glob.glob(os.path.join(tmpdir, "*.trace.json.gz"))
            assert len(trace_files) >= 1, f"Expected trace files in {tmpdir}, found: {os.listdir(tmpdir)}"
            for tf in trace_files:
                assert os.path.getsize(tf) > 0, f"Trace file {tf} is empty"

    def test_cuda_time_stats_written(self):
        """Test that stop() writes profiler_out_*.txt when dump_cuda_time_total=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=tmpdir,
                torch_profiler_dump_cuda_time_total=True,
            )
            profiler = TorchProfiler(config, worker_name="test-stats", local_rank=0)
            profiler.start()
            x = torch.randn(64, 64, device="cuda")
            _ = x @ x.T
            del x
            profiler.stop()

            stats_file = os.path.join(tmpdir, "profiler_out_0.txt")
            assert os.path.exists(stats_file), f"Expected {stats_file}"
            assert os.path.getsize(stats_file) > 0

    def test_worker_name_in_trace_filename(self):
        """Test that worker_name appears in the trace filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            profiler = TorchProfiler(config, worker_name="stage-0")
            profiler.start()
            x = torch.randn(32, 32, device="cuda")
            _ = x @ x.T
            del x
            profiler.stop()

            trace_files = glob.glob(os.path.join(tmpdir, "*stage-0*"))
            assert len(trace_files) >= 1, f"Expected trace with 'stage-0' in name, found: {os.listdir(tmpdir)}"


class TestTorchProfilerCPU:
    """Tests that don't require CUDA."""

    def test_import(self):
        """Test that TorchProfiler class can be imported."""
        from vllm_omni.profiler import TorchProfiler

        assert TorchProfiler is not None

    def test_profiler_config_import(self):
        """Test that ProfilerConfig can be imported."""
        from vllm_omni.profiler import ProfilerConfig

        assert ProfilerConfig is not None
