# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gzip
import os
import shutil
from contextlib import nullcontext

import torch
from torch.profiler import ProfilerActivity, profile
from vllm.logger import init_logger

from .base import ProfilerBase

logger = init_logger(__name__)


class TorchProfiler(ProfilerBase):
    """
    Torch-based profiler configured for End-to-End continuous recording.
    Uses 'on_trace_ready' to handle both Trace and Table export reliably.
    Compression (.gz) now happens inside the worker for race-free behavior.
    """

    _profiler: profile | None = None
    _trace_template: str = ""

    @classmethod
    def start(cls, trace_path_template: str) -> str:
        """
        Start the profiler with the given trace path template.
        Returns the expected final trace path for this rank (may be .gz).
        """
        # 1. Cleanup any existing profiler
        if cls._profiler is not None:
            logger.warning("[Rank %s] Stopping existing Torch profiler", cls._get_rank())
            cls._profiler.stop()
            cls._profiler = None

        rank = cls._get_rank()

        # 2. Make path absolute (helps in containers / different cwd)
        trace_path_template = os.path.abspath(trace_path_template)
        cls._trace_template = trace_path_template

        # Expected paths (compression happens later)
        json_file = f"{trace_path_template}_rank{rank}.json"
        table_file = f"{trace_path_template}_rank{rank}_table.txt"

        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        logger.info(f"[Rank {rank}] Starting End-to-End Torch profiler")

        # 3. Define the on_trace_ready handler
        def trace_handler(p):
            nonlocal json_file, table_file

            # A. Export JSON Trace
            try:
                p.export_chrome_trace(json_file)
                logger.info(f"[Rank {rank}] Trace exported to {json_file}")

                # ───────────── Compress to .gz immediately ─────────────
                gz_file = f"{json_file}.gz"
                try:
                    with open(json_file, "rb") as f_in:
                        with gzip.open(gz_file, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    logger.info(f"[Rank {rank}] Compressed trace to {gz_file}")

                    # Clean up original large JSON file
                    try:
                        os.remove(json_file)
                        logger.info(f"[Rank {rank}] Removed original trace {json_file}")
                    except OSError as remove_err:
                        logger.warning(f"[Rank {rank}] Failed to delete original {json_file}: {remove_err}")

                    # Use gzipped path as final result
                    json_file = gz_file

                except Exception as compress_err:
                    logger.warning(f"[Rank {rank}] Compression failed, keeping raw .json: {compress_err}")
                    # json_file remains the raw path

            except Exception as e:
                logger.warning(f"[Rank {rank}] Failed to export trace: {e}")

            # B. Export Text Table
            try:
                # Averages calculated here when data is ready
                table_str = p.key_averages().table(sort_by="cuda_time_total", row_limit=50)
                with open(table_file, "w") as f:
                    f.write(table_str)
                logger.info(f"[Rank {rank}] Table exported to {table_file}")
            except Exception as e:
                logger.warning(f"[Rank {rank}] Failed to generate table: {e}")

        # 4. Initialize profiler with long active period + callback
        cls._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=100000,  # long capture window
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )

        # 5. Start profiling
        cls._profiler.start()

        # Return the path we expect to exist at the end (optimistically .gz)
        expected_final = f"{trace_path_template}_rank{rank}.json.gz"
        return expected_final

    @classmethod
    def stop(cls) -> dict | None:
        if cls._profiler is None:
            return None

        rank = cls._get_rank()
        json_path = f"{cls._trace_template}_rank{rank}.json"
        gz_path = f"{json_path}.gz"

        try:
            cls._profiler.stop()  # Triggers trace_handler synchronously
        except Exception as e:
            logger.warning(f"[Rank {rank}] Profiler stop failed: {e}")

        cls._profiler = None

        # After stop(), handler has run → prefer gz if it exists
        final_trace = gz_path if os.path.exists(gz_path) else json_path

        table_path = f"{cls._trace_template}_rank{rank}_table.txt"

        return {"trace": final_trace, "table": table_path}

    @classmethod
    def step(cls):
        """Mark a profiler step (useful if using per-step scheduling)."""
        if cls._profiler is not None:
            cls._profiler.step()

    @classmethod
    def is_active(cls) -> bool:
        return cls._profiler is not None

    @classmethod
    def get_step_context(cls):
        """Context manager for profiling a single step (if needed)."""
        return nullcontext()
