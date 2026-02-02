# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.profiler import ProfilerConfig


class TestProfilerConfig:
    def test_default_config(self):
        """Test default configuration values."""
        config = ProfilerConfig()
        assert config.output_dir == "./profiles"

    def test_custom_output_dir(self):
        """Test custom output directory."""
        config = ProfilerConfig(output_dir="/tmp/my_profiles")
        assert config.output_dir == "/tmp/my_profiles"

    def test_config_is_dataclass(self):
        """Test that ProfilerConfig is a proper dataclass."""
        config = ProfilerConfig()
        assert hasattr(config, "output_dir")
