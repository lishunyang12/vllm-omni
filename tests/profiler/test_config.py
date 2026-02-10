# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile

import pytest

from vllm_omni.profiler import ProfilerConfig


class TestProfilerConfig:
    def test_default_config(self):
        """Test default configuration values."""
        config = ProfilerConfig()
        assert config.profiler is None
        assert config.torch_profiler_dir == ""
        assert config.torch_profiler_with_stack is True
        assert config.torch_profiler_with_flops is False
        assert config.torch_profiler_use_gzip is True
        assert config.torch_profiler_dump_cuda_time_total is True
        assert config.torch_profiler_record_shapes is False
        assert config.torch_profiler_with_memory is False
        assert config.delay_iterations == 0
        assert config.max_iterations == 0

    def test_torch_profiler_config(self):
        """Test creating a torch profiler config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(profiler="torch", torch_profiler_dir=tmpdir)
            assert config.profiler == "torch"
            assert config.torch_profiler_dir == os.path.abspath(tmpdir)

    def test_dir_without_profiler_raises(self):
        """Test that setting torch_profiler_dir without profiler='torch' raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="only applicable"):
                ProfilerConfig(torch_profiler_dir=tmpdir)

    def test_torch_without_dir_raises(self):
        """Test that profiler='torch' without dir raises."""
        with pytest.raises(ValueError, match="must be set"):
            ProfilerConfig(profiler="torch")

    def test_to_dict(self):
        """Test to_dict serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(profiler="torch", torch_profiler_dir=tmpdir)
            d = config.to_dict()
            assert isinstance(d, dict)
            assert d["profiler"] == "torch"
            assert d["torch_profiler_dir"] == os.path.abspath(tmpdir)

    def test_from_dict(self):
        """Test from_dict deserialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = {
                "profiler": "torch",
                "torch_profiler_dir": tmpdir,
                "torch_profiler_with_stack": False,
            }
            config = ProfilerConfig.from_dict(d)
            assert config.profiler == "torch"
            assert config.torch_profiler_with_stack is False

    def test_from_dict_ignores_unknown_fields(self):
        """Test from_dict ignores unknown fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = {
                "profiler": "torch",
                "torch_profiler_dir": tmpdir,
                "unknown_field": "value",
            }
            config = ProfilerConfig.from_dict(d)
            assert config.profiler == "torch"
            assert not hasattr(config, "unknown_field")

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=tmpdir,
                delay_iterations=5,
                max_iterations=100,
            )
            restored = ProfilerConfig.from_dict(original.to_dict())
            assert restored.profiler == original.profiler
            assert restored.torch_profiler_dir == original.torch_profiler_dir
            assert restored.delay_iterations == original.delay_iterations
            assert restored.max_iterations == original.max_iterations

    def test_dir_expanded(self):
        """Test that ~ in dir is expanded."""
        config = ProfilerConfig(profiler="torch", torch_profiler_dir="~/profiles")
        assert "~" not in config.torch_profiler_dir
        assert os.path.isabs(config.torch_profiler_dir)


class TestProfilerConfigFromAny:
    """Tests for ProfilerConfig.from_any() — the online serving conversion path."""

    def test_from_any_none(self):
        """None input returns None."""
        assert ProfilerConfig.from_any(None) is None

    def test_from_any_own_instance(self):
        """Our own ProfilerConfig is returned as-is."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(profiler="torch", torch_profiler_dir=tmpdir)
            result = ProfilerConfig.from_any(config)
            assert result is config

    def test_from_any_dict(self):
        """Dict input uses from_dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = {"profiler": "torch", "torch_profiler_dir": tmpdir}
            result = ProfilerConfig.from_any(d)
            assert isinstance(result, ProfilerConfig)
            assert result.profiler == "torch"

    def test_from_any_upstream_like_object(self):
        """Object with .profiler and .torch_profiler_dir attributes (upstream-like)."""

        class UpstreamConfig:
            profiler = "torch"
            torch_profiler_dir = "/tmp/test_profiles"

        with tempfile.TemporaryDirectory() as tmpdir:
            obj = UpstreamConfig()
            obj.torch_profiler_dir = tmpdir
            result = ProfilerConfig.from_any(obj)
            assert isinstance(result, ProfilerConfig)
            assert result.profiler == "torch"
            assert result.torch_profiler_dir == os.path.abspath(tmpdir)

    def test_from_any_object_profiler_none(self):
        """Object with profiler=None returns None."""

        class NullConfig:
            profiler = None

        assert ProfilerConfig.from_any(NullConfig()) is None

    def test_from_any_object_no_profiler_attr(self):
        """Object without .profiler attribute returns None."""

        class RandomObj:
            foo = "bar"

        assert ProfilerConfig.from_any(RandomObj()) is None


class TestProfilerConfigReExport:
    """Test that ProfilerConfig is accessible from vllm_omni.config."""

    def test_import_from_config(self):
        from vllm_omni.config import ProfilerConfig as ReExported

        assert ReExported is ProfilerConfig
