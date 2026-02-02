# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass
class ProfilerConfig:
    """Configuration for profiling.

    When profiling is enabled, both performance trace (.json.gz) and
    memory timeline (.html) are captured.

    Args:
        output_dir: Directory to save profiling outputs.

    Example:
        >>> # Enable profiling with default output dir
        >>> config = ProfilerConfig()
        >>>
        >>> # Custom output dir
        >>> config = ProfilerConfig(output_dir="./my_profiles")
    """

    output_dir: str = "./profiles"
