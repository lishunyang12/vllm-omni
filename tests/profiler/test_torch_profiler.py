# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile

import pytest
import torch

from vllm_omni.profiler import ProfilerConfig, TorchProfiler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTorchProfiler:
    """Tests for TorchProfiler."""

    def test_start_stop_lifecycle(self):
        """Test basic start/stop lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(output_dir=tmpdir)
            profiler = TorchProfiler()

            assert not profiler.is_active()

            output_path = profiler.start(f"{tmpdir}/test", config)
            assert profiler.is_active()
            assert "test" in output_path

            # Do some work
            x = torch.randn(100, 100, device="cuda")
            y = x @ x.T
            del x, y

            result = profiler.stop()
            assert not profiler.is_active()
            assert result is not None
            assert "trace" in result
            assert "stats" in result

    def test_trace_file_created(self):
        """Test that trace file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(output_dir=tmpdir)
            profiler = TorchProfiler()

            profiler.start(f"{tmpdir}/trace_test", config)
            x = torch.randn(100, 100, device="cuda")
            _ = x @ x.T
            result = profiler.stop()

            trace_path = result.get("trace")
            assert trace_path is not None
            # File may still be compressing, check base exists
            base_path = trace_path.replace(".gz", "")
            assert os.path.exists(trace_path) or os.path.exists(base_path)

    def test_timeline_file_created(self):
        """Test that .html timeline file is created (requires matplotlib)."""
        pytest.importorskip("matplotlib", reason="matplotlib required for timeline export")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(output_dir=tmpdir)
            profiler = TorchProfiler()

            profiler.start(f"{tmpdir}/timeline_test", config)
            x = torch.randn(1000, 1000, device="cuda")
            del x
            result = profiler.stop()

            timeline_path = result.get("timeline_html")
            assert timeline_path is not None
            assert os.path.exists(timeline_path)
            assert timeline_path.endswith(".html")

    def test_memory_stats_collected(self):
        """Test that memory statistics are collected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(output_dir=tmpdir)
            profiler = TorchProfiler()

            profiler.start(f"{tmpdir}/stats_test", config)
            x = torch.randn(1000, 1000, device="cuda")
            del x
            result = profiler.stop()

            stats = result.get("stats")
            assert stats is not None
            assert "peak_allocated_mb" in stats
            assert "current_allocated_mb" in stats
            assert stats["peak_allocated_mb"] >= 0

    def test_stop_without_start(self):
        """Test stop without start returns None."""
        profiler = TorchProfiler()
        result = profiler.stop()
        assert result is None

    def test_double_start(self):
        """Test double start stops previous profiler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(output_dir=tmpdir)
            profiler = TorchProfiler()

            profiler.start(f"{tmpdir}/first", config)
            profiler.start(f"{tmpdir}/second", config)  # Should stop first

            assert profiler.is_active()
            profiler.stop()
            assert not profiler.is_active()

    def test_output_path_includes_rank(self):
        """Test that output path includes rank."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(output_dir=tmpdir)
            profiler = TorchProfiler()

            output_path = profiler.start(f"{tmpdir}/rank_test", config)
            assert "_rank" in output_path
            profiler.stop()

    def test_output_files_exist(self):
        """Test all expected output files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(output_dir=tmpdir)
            profiler = TorchProfiler()

            profiler.start(f"{tmpdir}/files_test", config)
            x = torch.randn(100, 100, device="cuda")
            _ = x @ x.T
            results = profiler.stop()

            # Check performance trace
            trace = results.get("trace")
            if trace:
                base = trace.replace(".gz", "")
                assert os.path.exists(trace) or os.path.exists(base)

            # Timeline requires matplotlib - only check if it was created
            timeline = results.get("timeline_html")
            if timeline:
                assert os.path.exists(timeline)


class TestTorchProfilerCPU:
    """Tests that don't require CUDA."""

    def test_profiler_class_exists(self):
        """Test that TorchProfiler class can be imported."""
        from vllm_omni.profiler import TorchProfiler

        assert TorchProfiler is not None

    def test_is_active_default(self):
        """Test is_active returns False by default."""
        profiler = TorchProfiler()
        assert not profiler.is_active()

    def test_stop_inactive_returns_none(self):
        """Test stopping an inactive profiler returns None."""
        profiler = TorchProfiler()
        assert profiler.stop() is None

    def test_get_step_context(self):
        """Test get_step_context returns nullcontext."""
        profiler = TorchProfiler()
        ctx = profiler.get_step_context()
        # Should be usable as context manager
        with ctx:
            pass


class TestProfilerConfig:
    """Tests for ProfilerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProfilerConfig()
        assert config.output_dir == "./profiles"

    def test_custom_output_dir(self):
        """Test custom output directory."""
        config = ProfilerConfig(output_dir="/tmp/my_profiles")
        assert config.output_dir == "/tmp/my_profiles"
