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
    """PyTorch-based profiler for performance traces and memory profiling.

    Supports two profiling modes (can be used together):
    - Performance: Chrome trace for timing analysis (.json.gz)
    - Memory: Snapshot (.pickle) + categorized timeline (.html)
    """

    _profiler: profile | None = None
    _output_prefix: str = ""
    _config: ProfilerConfig | None = None
    _is_memory_recording: bool = False

    @classmethod
    def start(cls, output_prefix: str, config: ProfilerConfig) -> str:
        """Start profiling based on configuration.

        Args:
            output_prefix: Base path without rank or extension.
            config: Profiler configuration.

        Returns:
            Expected path to the primary output file.
        """
        # Cleanup any existing profiler
        if cls._profiler is not None:
            logger.warning("[Rank %s] Stopping existing profiler", cls._get_rank())
            cls.stop()

        cls._config = config
        cls._output_prefix = os.path.abspath(output_prefix)
        rank = cls._get_rank()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(cls._output_prefix), exist_ok=True)

        logger.info(
            "[Rank %s] Starting profiler (performance=%s, memory=%s)",
            rank,
            config.performance,
            config.memory,
        )

        # Start memory history recording if memory profiling enabled
        if config.memory and torch.cuda.is_available():
            try:
                torch.cuda.memory._record_memory_history(
                    enabled="all",
                    context="all",
                    stacks="python",
                    max_entries=config.max_entries,
                )
                cls._is_memory_recording = True
                logger.info("[Rank %s] Memory history recording started", rank)
            except Exception as e:
                logger.warning("[Rank %s] Failed to start memory history: %s", rank, e)

        # Start profiler if either performance or memory is enabled
        if config.performance or config.memory:
            json_file = f"{cls._output_prefix}_rank{rank}.json"

            def trace_handler(p):
                nonlocal json_file
                if config.performance:
                    try:
                        p.export_chrome_trace(json_file)
                        logger.info("[Rank %s] Trace exported to %s", rank, json_file)
                        # Background compression
                        try:
                            subprocess.Popen(["gzip", "-f", json_file])
                        except Exception:
                            pass
                    except Exception as e:
                        logger.warning("[Rank %s] Failed to export trace: %s", rank, e)

            cls._profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=100000),
                on_trace_ready=trace_handler if config.performance else None,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=config.performance,
            )
            cls._profiler.start()

        # Return primary output path
        if config.performance:
            return f"{cls._output_prefix}_rank{rank}.json.gz"
        elif config.memory:
            return f"{cls._output_prefix}_rank{rank}_snapshot.pickle"
        return ""

    @classmethod
    def stop(cls) -> dict[str, Any] | None:
        """Stop profiling and return results."""
        if cls._profiler is None and not cls._is_memory_recording:
            return None

        rank = cls._get_rank()
        config = cls._config
        results: dict[str, Any] = {}

        # Export memory snapshot and timeline
        if cls._is_memory_recording and config and config.memory:
            snapshot_path = f"{cls._output_prefix}_rank{rank}_snapshot.pickle"
            timeline_path = f"{cls._output_prefix}_rank{rank}_timeline.html"

            try:
                torch.cuda.memory._dump_snapshot(snapshot_path)
                results["snapshot"] = snapshot_path
                logger.info("[Rank %s] Memory snapshot saved to %s", rank, snapshot_path)
            except Exception as e:
                logger.warning("[Rank %s] Failed to dump snapshot: %s", rank, e)

            try:
                if cls._profiler:
                    cls._profiler.stop()
                    cls._profiler.export_memory_timeline(timeline_path, device="cuda:0")
                    results["timeline_html"] = timeline_path
                    logger.info("[Rank %s] Memory timeline saved to %s", rank, timeline_path)
                    cls._profiler = None
            except Exception as e:
                logger.warning("[Rank %s] Failed to export timeline: %s", rank, e)

            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception as e:
                logger.warning("[Rank %s] Error disabling memory history: %s", rank, e)

            results["stats"] = cls._collect_memory_stats()
            cls._is_memory_recording = False

        # Stop profiler for performance trace (if not already stopped for memory)
        if cls._profiler is not None:
            gz_path = f"{cls._output_prefix}_rank{rank}.json.gz"
            try:
                cls._profiler.stop()
            except Exception as e:
                logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)
            cls._profiler = None

            if config and config.performance:
                results["trace"] = gz_path

        cls._config = None
        return results if results else None

    @classmethod
    def _collect_memory_stats(cls) -> dict[str, Any]:
        """Collect current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {}

        try:
            stats = torch.cuda.memory_stats()
            return {
                "peak_allocated_mb": stats.get("allocated_bytes.all.peak", 0)
                / (1024**2),
                "current_allocated_mb": stats.get("allocated_bytes.all.current", 0)
                / (1024**2),
                "peak_reserved_mb": stats.get("reserved_bytes.all.peak", 0)
                / (1024**2),
                "current_reserved_mb": stats.get("reserved_bytes.all.current", 0)
                / (1024**2),
                "num_alloc_retries": stats.get("num_alloc_retries", 0),
                "num_ooms": stats.get("num_ooms", 0),
            }
        except Exception as e:
            logger.warning("Failed to collect memory stats: %s", e)
            return {}

    @classmethod
    def step(cls):
        """Advance one profiling step."""
        if cls._profiler is not None:
            cls._profiler.step()

    @classmethod
    def is_active(cls) -> bool:
        """Check if profiling is active."""
        return cls._profiler is not None or cls._is_memory_recording

    @classmethod
    def get_step_context(cls):
        """Return nullcontext (no-op)."""
        return nullcontext()
