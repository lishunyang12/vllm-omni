# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from contextlib import nullcontext
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile
from vllm.logger import init_logger

from .base import ProfilerBase
from .config import ProfilerConfig

logger = init_logger(__name__)


class TorchProfiler(ProfilerBase):
    """PyTorch-based profiler for performance and memory analysis.

    Captures both:
    - Performance: Chrome trace for timing analysis (.json.gz)
    - Memory: Categorized timeline (.html) for memory composition

    Usage:
        TorchProfiler.start("/path/to/output_prefix", ProfilerConfig())
        # ... run your code ...
        results = TorchProfiler.stop()
    """

    _profiler: profile | None = None
    _output_prefix: str = ""

    @classmethod
    def start(cls, output_prefix: str, config: ProfilerConfig | dict | None = None) -> str:
        """Start profiling.

        Args:
            output_prefix: Base path without rank or extension.
            config: ProfilerConfig, dict, or None. Dict is auto-converted.

        Returns:
            Expected path to the trace output file.
        """
        if cls._profiler is not None:
            logger.warning("[Rank %s] Stopping existing profiler", cls._get_rank())
            cls.stop()

        # Handle dict config (from RPC serialization)
        if isinstance(config, dict):
            config = ProfilerConfig(**config)
        elif config is None:
            config = ProfilerConfig(output_dir=os.path.dirname(output_prefix) or ".")

        cls._output_prefix = os.path.abspath(output_prefix)
        rank = cls._get_rank()

        os.makedirs(os.path.dirname(cls._output_prefix), exist_ok=True)
        logger.info("[Rank %s] Starting profiler", rank)

        json_file = f"{cls._output_prefix}_rank{rank}.json"

        def trace_handler(p):
            try:
                p.export_chrome_trace(json_file)
                logger.info("[Rank %s] Trace exported to %s", rank, json_file)
                subprocess.Popen(["gzip", "-f", json_file])
            except Exception as e:
                logger.warning("[Rank %s] Failed to export trace: %s", rank, e)

        cls._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=100000),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        cls._profiler.start()

        return f"{cls._output_prefix}_rank{rank}.json.gz"

    @classmethod
    def stop(cls) -> dict[str, Any] | None:
        """Stop profiling and return results.

        Returns:
            Dict with keys: trace, timeline_html, stats. None if not active.
        """
        if cls._profiler is None:
            return None

        rank = cls._get_rank()
        results: dict[str, Any] = {}

        try:
            cls._profiler.stop()
        except Exception as e:
            logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)
            cls._profiler = None
            return None

        # Performance trace
        results["trace"] = f"{cls._output_prefix}_rank{rank}.json.gz"

        # Memory timeline
        timeline_path = f"{cls._output_prefix}_rank{rank}_timeline.html"
        try:
            cls._profiler.export_memory_timeline(timeline_path, device="cuda:0")
            if os.path.exists(timeline_path):
                results["timeline_html"] = timeline_path
                logger.info("[Rank %s] Memory timeline saved to %s", rank, timeline_path)
        except Exception as e:
            logger.warning("[Rank %s] Failed to export timeline: %s", rank, e)

        # Memory stats
        results["stats"] = cls._collect_memory_stats()

        cls._profiler = None
        return results

    @classmethod
    def _collect_memory_stats(cls) -> dict[str, Any]:
        """Collect current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {}
        try:
            s = torch.cuda.memory_stats()
            return {
                "peak_allocated_mb": s.get("allocated_bytes.all.peak", 0) / (1024**2),
                "current_allocated_mb": s.get("allocated_bytes.all.current", 0) / (1024**2),
                "peak_reserved_mb": s.get("reserved_bytes.all.peak", 0) / (1024**2),
                "current_reserved_mb": s.get("reserved_bytes.all.current", 0) / (1024**2),
            }
        except Exception:
            return {}

    @classmethod
    def step(cls):
        """Advance one profiling step."""
        if cls._profiler is not None:
            cls._profiler.step()

    @classmethod
    def is_active(cls) -> bool:
        """Check if profiling is active."""
        return cls._profiler is not None

    @classmethod
    def get_step_context(cls):
        """Return context manager for profiling step (no-op for torch profiler)."""
        return nullcontext()
