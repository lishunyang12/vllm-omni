# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
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
    """

    _profiler: profile | None = None
    _trace_template: str = ""

    @classmethod
    def start(cls, trace_path_template: str) -> str:
        # 1. Cleanup existing
        if cls._profiler is not None:
            logger.warning("[Rank %s] Stopping existing Torch profiler", cls._get_rank())
            cls._profiler.stop()
            cls._profiler = None

        rank = cls._get_rank()

        # 2. Setup Absolute Paths
        # We perform path resolution here to ensure the callback has correct paths
        trace_path_template = os.path.abspath(trace_path_template)
        cls._trace_template = trace_path_template

        json_file = f"{trace_path_template}_rank{rank}.json"
        table_file = f"{trace_path_template}_rank{rank}_table.txt"
        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        logger.info(f"[Rank {rank}] Starting End-to-End Torch profiler")

        # 3. Define the Handler
        # PyTorch calls this automatically when we call .stop()
        def trace_handler(p):
            # A. Export JSON Trace
            try:
                p.export_chrome_trace(json_file)
                logger.info(f"[Rank {rank}] Trace exported to {json_file}")
            except Exception as e:
                logger.warning(f"[Rank {rank}] Failed to export trace: {e}")

            # B. Export Text Table
            try:
                # Calculate averages inside the callback where data is guaranteed ready
                table_str = p.key_averages().table(sort_by="cuda_time_total", row_limit=50)
                with open(table_file, "w") as f:
                    f.write(table_str)
                logger.info(f"[Rank {rank}] Table exported to {table_file}")
            except Exception as e:
                logger.warning(f"[Rank {rank}] Failed to generate table: {e}")

        # 4. Initialize with Infinite Schedule + Callback
        cls._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=100000),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )

        # 5. Start
        cls._profiler.start()

        return json_file

    @classmethod
    def stop(cls) -> dict | None:
        if cls._profiler is None:
            return None

        rank = cls._get_rank()
        trace_path = f"{cls._trace_template}_rank{rank}.json"
        table_path = f"{cls._trace_template}_rank{rank}_table.txt"

        # 6. Stop the Profiler
        # This triggers the 'trace_handler' defined in start(), which saves both files.
        try:
            cls._profiler.stop()
        except Exception as e:
            logger.warning(f"Profiler stop failed: {e}")

        cls._profiler = None

        return {"trace": trace_path, "table": table_path}

    @classmethod
    def step(cls):
        """
        Marks steps in the timeline.
        """
        if cls._profiler is not None:
            cls._profiler.step()

    @classmethod
    def is_active(cls) -> bool:
        return cls._profiler is not None

    @classmethod
    def get_step_context(cls):
        return nullcontext()
