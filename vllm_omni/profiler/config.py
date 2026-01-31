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
        record_from_start: If True and memory=True, start recording memory
            history immediately when the engine initializes (captures model
            loading). If False, only captures allocations after start_profile().
    """

    output_dir: str = "./profiles"
    performance: bool = True
    memory: bool = False
    backend: str = "torch"  # "torch" or "nsight" (future)
    max_entries: int = 100000
    record_from_start: bool = False

    def __post_init__(self):
        if self.backend not in ("torch", "nsight"):
            raise ValueError(f"backend must be 'torch' or 'nsight', got {self.backend}")
        if self.backend == "nsight":
            raise NotImplementedError("Nsight backend not yet implemented")
