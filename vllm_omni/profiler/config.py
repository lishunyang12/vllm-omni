# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass
class ProfilerConfig:
    """Configuration for profiling.

    Args:
        output_dir: Directory to save profiling outputs.
        performance: Enable performance profiling (Chrome trace).
        memory: Enable memory profiling (snapshot + timeline).
        backend: Profiler backend ("torch" or "nsight" for future).
        max_entries: Max memory allocation records (memory profiler only).
    """

    output_dir: str = "./profiles"
    performance: bool = True
    memory: bool = False
    backend: str = "torch"  # "torch" or "nsight" (future)
    max_entries: int = 100000

    def __post_init__(self):
        if self.backend not in ("torch", "nsight"):
            raise ValueError(f"backend must be 'torch' or 'nsight', got {self.backend}")
        if self.backend == "nsight":
            raise NotImplementedError("Nsight backend not yet implemented")
