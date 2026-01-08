# vllm_omni/diffusion/profiler/torch_profiler.py

import os
from contextlib import nullcontext
from typing import Optional

import torch
from torch.profiler import ProfilerActivity, profile

from vllm.logger import init_logger
from .base import ProfilerBase

logger = init_logger(__name__)


class TorchProfiler(ProfilerBase):
    """
    Torch-based profiler using torch.profiler.profile.
    Exports Chrome traces (.json) viewable in chrome://tracing or Perfetto.
    """

    _profiler: Optional[profile] = None
    _trace_template: str = ""

    @classmethod
    def start(cls, trace_path_template: str) -> str:
        if cls._profiler is not None:
            logger.warning("[Rank %s] Stopping existing Torch profiler", cls._get_rank())
            cls.stop()

        rank = cls._get_rank()
        trace_file = f"{trace_path_template}_rank{rank}.json"
        os.makedirs(os.path.dirname(trace_file), exist_ok=True)

        logger.info(f"[Rank {rank}] Starting Torch profiler {trace_file}")

        cls._trace_template = trace_path_template

        cls._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=3,
                active=15,   # adjust based on typical num_inference_steps
                repeat=1,
            ),
            on_trace_ready=lambda prof: prof.export_chrome_trace(trace_file),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        cls._profiler.__enter__()

        return trace_file

    @classmethod
    def stop(cls) -> Optional[dict]:
        if cls._profiler is None:
            return None

        rank = cls._get_rank()
        
        # 1. Export the Chrome Trace manually before exiting context if needed
        # or rely on on_trace_ready. To be safe for a single generation:
        trace_path = f"{cls._trace_template}_rank{rank}.json"
        cls._profiler.export_chrome_trace(trace_path)

        # 2. Generate the Key Averages Table as a string
        table_str = cls._profiler.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=20
        )
        
        # 3. Save table to a .txt file
        table_path = f"{cls._trace_template}_rank{rank}_table.txt"
        with open(table_path, "w") as f:
            f.write(table_str)

        # Finalize the profiler
        cls._profiler.__exit__(None, None, None)
        cls._profiler = None
        
        return {"trace": trace_path, "table": table_path}
    
    @classmethod
    def step(cls):
        """Explicitly advance the profiler schedule."""
        if cls._profiler is not None:
            cls._profiler.step()

    @classmethod
    def is_active(cls) -> bool:
        return cls._profiler is not None