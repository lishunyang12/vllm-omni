# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
from torch.profiler import ProfilerActivity
from vllm.logger import init_logger

from .config import ProfilerConfig

logger = init_logger(__name__)


class TorchProfiler:
    """Instance-based torch profiler aligned with upstream vLLM 0.16.0.

    Mirrors upstream WorkerProfiler + TorchProfilerWrapper behavior with
    independent implementation. Uses ``tensorboard_trace_handler`` for trace
    export and supports delay/max iteration control.

    Args:
        config: ProfilerConfig with torch profiler settings.
        worker_name: Name used in trace file naming.
        local_rank: GPU rank for CUDA time stats output.
    """

    def __init__(
        self,
        config: ProfilerConfig,
        worker_name: str = "",
        local_rank: int = 0,
    ):
        self._config = config
        self._local_rank = local_rank
        self._delay_iters = config.delay_iterations
        self._max_iters = config.max_iterations
        self._active = False
        self._running = False
        self._active_iteration_count = 0
        self._profiling_for_iters = 0

        self._profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=config.torch_profiler_record_shapes,
            profile_memory=config.torch_profiler_with_memory,
            with_stack=config.torch_profiler_with_stack,
            with_flops=config.torch_profiler_with_flops,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                config.torch_profiler_dir,
                worker_name=worker_name,
                use_gzip=config.torch_profiler_use_gzip,
            ),
        )

    def start(self):
        """Start profiling, accounting for delayed starts."""
        if self._active:
            return
        self._active = True
        if self._delay_iters == 0:
            self._call_start()

    def stop(self):
        """Stop profiling."""
        if not self._active:
            return
        self._active = False
        self._active_iteration_count = 0
        self._profiling_for_iters = 0
        if self._running:
            self._call_stop()

    def step(self):
        """Per-iteration update for delay/max handling."""
        if not self._active:
            return
        self._active_iteration_count += 1
        if not self._running and self._delay_iters > 0 and self._active_iteration_count == self._delay_iters:
            self._call_start()
        if self._running:
            self._profiling_for_iters += 1
        if self._max_iters > 0 and self._running and self._profiling_for_iters > self._max_iters:
            self._call_stop()

    def shutdown(self):
        """Shutdown the profiler if running."""
        if self._running:
            self.stop()

    @property
    def is_running(self) -> bool:
        return self._running

    def _call_start(self):
        try:
            self._profiler.start()
            self._running = True
        except Exception as e:
            logger.warning("Failed to start profiler: %s", e)

    def _call_stop(self):
        try:
            self._profiler.stop()
            if self._config.torch_profiler_dump_cuda_time_total:
                table = self._profiler.key_averages().table(sort_by="self_cuda_time_total")
                out_file = os.path.join(
                    self._config.torch_profiler_dir,
                    f"profiler_out_{self._local_rank}.txt",
                )
                with open(out_file, "w") as f:
                    print(table, file=f)
                if self._local_rank == 0:
                    print(table)
        except Exception as e:
            logger.warning("Failed to stop profiler: %s", e)
        self._running = False
