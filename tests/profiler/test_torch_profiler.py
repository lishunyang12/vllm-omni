# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile

import pytest
import torch

from vllm_omni.profiler import ProfilerConfig, TorchProfiler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTorchProfilerPerformance:
    """Tests for performance profiling (Chrome trace)."""

    def test_start_stop_lifecycle(self):
        """Test basic start/stop lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(output_dir=tmpdir, performance=True, memory=False)
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

    def test_trace_file_created(self):
        """Test that trace file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(output_dir=tmpdir, performance=True, memory=False)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTorchProfilerMemory:
    """Tests for memory profiling (snapshot + timeline)."""

    def test_memory_profiling_lifecycle(self):
        """Test memory profiling start/stop lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(output_dir=tmpdir, performance=False, memory=True)
            profiler = TorchProfiler()

            assert not profiler.is_active()

            profiler.start(f"{tmpdir}/mem_test", config)
            assert profiler.is_active()

            # Allocate some GPU memory
            tensors = [torch.randn(1000, 1000, device="cuda") for _ in range(5)]
            del tensors

            result = profiler.stop()
            assert not profiler.is_active()
            assert result is not None

    def test_snapshot_file_created(self):
        """Test that .pickle snapshot file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(output_dir=tmpdir, performance=False, memory=True)
            profiler = TorchProfiler()

            profiler.start(f"{tmpdir}/snapshot_test", config)
            x = torch.randn(1000, 1000, device="cuda")
            del x
            result = profiler.stop()

            snapshot_path = result.get("snapshot")
            assert snapshot_path is not None
            assert os.path.exists(snapshot_path)
            assert snapshot_path.endswith(".pickle")

    def test_timeline_file_created(self):
        """Test that .html timeline file is created (requires matplotlib)."""
        pytest.importorskip("matplotlib", reason="matplotlib required for timeline export")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(output_dir=tmpdir, performance=False, memory=True)
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
            config = ProfilerConfig(output_dir=tmpdir, performance=False, memory=True)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTorchProfilerBoth:
    """Tests for combined performance and memory profiling."""

    def test_both_enabled(self):
        """Test with both performance and memory enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                output_dir=tmpdir,
                performance=True,
                memory=True,
            )
            profiler = TorchProfiler()

            profiler.start(f"{tmpdir}/both", config)
            x = torch.randn(100, 100, device="cuda")
            _ = x @ x.T
            results = profiler.stop()

            # Should have trace
            assert "trace" in results
            # Should have memory outputs
            assert "snapshot" in results
            assert "stats" in results
            # timeline_html requires matplotlib - only present if installed
            # assert "timeline_html" in results  # Optional, requires matplotlib

    def test_nothing_enabled(self):
        """Test with no profiling enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                output_dir=tmpdir,
                performance=False,
                memory=False,
            )
            profiler = TorchProfiler()

            profiler.start(f"{tmpdir}/none", config)
            results = profiler.stop()

            # Should return None or empty when nothing is enabled
            assert results is None or results == {}

    def test_output_files_exist(self):
        """Test all expected output files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                output_dir=tmpdir,
                performance=True,
                memory=True,
            )
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

            # Check memory files
            snapshot = results.get("snapshot")
            if snapshot:
                assert os.path.exists(snapshot)

            # Timeline requires matplotlib - only check if it was created
            timeline = results.get("timeline_html")
            if timeline:
                assert os.path.exists(timeline)
            # Note: timeline_html may be None if matplotlib is not installed


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
