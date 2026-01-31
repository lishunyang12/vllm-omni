# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.profiler import ProfilerConfig


class TestProfilerConfig:
    def test_default_config(self):
        """Test default configuration values."""
        config = ProfilerConfig()
        assert config.output_dir == "./profiles"
        assert config.backend == "torch"
        assert config.performance is True
        assert config.memory is False
        assert config.max_entries == 100000

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProfilerConfig(
            output_dir="/tmp/my_profiles",
            performance=True,
            memory=True,
            max_entries=50000,
        )
        assert config.output_dir == "/tmp/my_profiles"
        assert config.performance is True
        assert config.memory is True
        assert config.max_entries == 50000

    def test_memory_only(self):
        """Test memory-only config."""
        config = ProfilerConfig(performance=False, memory=True)
        assert config.performance is False
        assert config.memory is True

    def test_performance_only(self):
        """Test performance-only config (default)."""
        config = ProfilerConfig(performance=True, memory=False)
        assert config.performance is True
        assert config.memory is False

    def test_invalid_backend(self):
        """Test invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="backend must be"):
            ProfilerConfig(backend="invalid")

    def test_nsight_not_implemented(self):
        """Test that Nsight backend raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Nsight"):
            ProfilerConfig(backend="nsight")

    def test_no_profiling(self):
        """Test config with no profiling enabled."""
        config = ProfilerConfig(performance=False, memory=False)
        assert config.performance is False
        assert config.memory is False

    def test_custom_max_entries(self):
        """Test custom max_entries value."""
        config = ProfilerConfig(max_entries=1000)
        assert config.max_entries == 1000

    def test_config_is_dataclass(self):
        """Test that ProfilerConfig is a proper dataclass."""
        config = ProfilerConfig()
        # Should be able to access all fields
        assert hasattr(config, "output_dir")
        assert hasattr(config, "performance")
        assert hasattr(config, "memory")
        assert hasattr(config, "backend")
        assert hasattr(config, "max_entries")
