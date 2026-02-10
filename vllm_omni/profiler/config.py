# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

ProfilerKind = Literal["torch", "cuda"]


@dataclass
class ProfilerConfig:
    """Profiler configuration aligned with vLLM 0.16.0 semantics.

    Independent implementation with the same fields as upstream vLLM's
    ProfilerConfig. This enables familiar configuration while keeping
    vllm-omni fully decoupled.

    Args:
        profiler: Which profiler to use. Options: 'torch', 'cuda', or None.
        torch_profiler_dir: Directory to save torch profiler traces.
        torch_profiler_with_stack: Enable stack tracing in torch profiler.
        torch_profiler_with_flops: Enable FLOPS counting in torch profiler.
        torch_profiler_use_gzip: Save traces in gzip format.
        torch_profiler_dump_cuda_time_total: Dump total CUDA time stats.
        torch_profiler_record_shapes: Record tensor shapes.
        torch_profiler_with_memory: Enable memory profiling.
        delay_iterations: Engine iterations to skip before starting.
        max_iterations: Maximum engine iterations to profile (0 = no limit).

    Example:
        >>> config = ProfilerConfig(
        ...     profiler="torch",
        ...     torch_profiler_dir="./profiles",
        ... )
    """

    profiler: ProfilerKind | None = None
    torch_profiler_dir: str = ""
    torch_profiler_with_stack: bool = True
    torch_profiler_with_flops: bool = False
    torch_profiler_use_gzip: bool = True
    torch_profiler_dump_cuda_time_total: bool = True
    torch_profiler_record_shapes: bool = False
    torch_profiler_with_memory: bool = False
    delay_iterations: int = 0
    max_iterations: int = 0

    def __post_init__(self):
        if self.torch_profiler_dir and self.profiler != "torch":
            raise ValueError("torch_profiler_dir is only applicable when profiler='torch'")
        if self.profiler == "torch" and not self.torch_profiler_dir:
            raise ValueError("torch_profiler_dir must be set when profiler='torch'")
        if self.torch_profiler_dir:
            self.torch_profiler_dir = os.path.abspath(os.path.expanduser(self.torch_profiler_dir))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProfilerConfig":
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    @classmethod
    def from_any(cls, obj: Any) -> "ProfilerConfig | None":
        """Convert any profiler-config-like object to our ProfilerConfig.

        Accepts our own ProfilerConfig, a compatible dataclass from the CLI
        layer, or a dict.  Returns None when *obj* is None or has no profiler
        set.
        """
        if obj is None:
            return None
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        if hasattr(obj, "profiler") and obj.profiler is not None:
            return cls(
                profiler=obj.profiler,
                torch_profiler_dir=getattr(obj, "torch_profiler_dir", "") or "",
            )
        return None
