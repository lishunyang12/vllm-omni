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

    For full memory granularity (including model loading), call
    start_early_memory_recording() before model initialization, then
    call start() when ready to begin performance profiling.
    """

    _profiler: profile | None = None
    _output_prefix: str = ""
    _config: ProfilerConfig | None = None
    _is_memory_recording: bool = False
    _early_memory_config: ProfilerConfig | None = None

    @classmethod
    def start_early_memory_recording(cls, config: ProfilerConfig) -> None:
        """Start memory history recording early (before model loading).

        Call this before model initialization to capture all GPU allocations
        including model weights. The recording will continue until stop() is
        called, and the snapshot will include all allocations from this point.

        Args:
            config: Profiler configuration (must have memory=True).
        """
        if not config.memory:
            logger.warning("start_early_memory_recording called but config.memory=False")
            return

        if cls._is_memory_recording:
            logger.warning("Memory recording already active, skipping early start")
            return

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, cannot start memory recording")
            return

        rank = cls._get_rank()
        try:
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="python",
                max_entries=config.max_entries,
            )
            cls._is_memory_recording = True
            cls._early_memory_config = config
            logger.info(
                "[Rank %s] Early memory recording started (max_entries=%d)",
                rank,
                config.max_entries,
            )
        except Exception as e:
            logger.warning("[Rank %s] Failed to start early memory recording: %s", rank, e)

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
        # (skip if already started via start_early_memory_recording)
        if config.memory and torch.cuda.is_available() and not cls._is_memory_recording:
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
        elif cls._is_memory_recording:
            logger.info("[Rank %s] Memory recording already active (early start)", rank)

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

        # Add trace path if performance profiling was enabled
        # (must be done before stopping profiler in memory branch)
        if config and config.performance:
            gz_path = f"{cls._output_prefix}_rank{rank}.json.gz"
            results["trace"] = gz_path

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
                    # Only add to results if file was actually created
                    # (export_memory_timeline silently fails if matplotlib is missing)
                    if os.path.exists(timeline_path):
                        results["timeline_html"] = timeline_path
                        logger.info("[Rank %s] Memory timeline saved to %s", rank, timeline_path)
                    else:
                        logger.warning(
                            "[Rank %s] Memory timeline not created (matplotlib may be missing)",
                            rank,
                        )
                    cls._profiler = None
            except Exception as e:
                logger.warning("[Rank %s] Failed to export timeline: %s", rank, e)

            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception as e:
                logger.warning("[Rank %s] Error disabling memory history: %s", rank, e)

            results["stats"] = cls._collect_memory_stats()
            cls._is_memory_recording = False
            cls._early_memory_config = None

        # Stop profiler for performance trace (if not already stopped for memory)
        if cls._profiler is not None:
            try:
                cls._profiler.stop()
            except Exception as e:
                logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)
            cls._profiler = None

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
                "peak_allocated_mb": stats.get("allocated_bytes.all.peak", 0) / (1024**2),
                "current_allocated_mb": stats.get("allocated_bytes.all.current", 0) / (1024**2),
                "peak_reserved_mb": stats.get("reserved_bytes.all.peak", 0) / (1024**2),
                "current_reserved_mb": stats.get("reserved_bytes.all.current", 0) / (1024**2),
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
