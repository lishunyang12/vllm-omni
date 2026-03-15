# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU memory debugging utility for diffusion pipelines.

Enable by setting VLLM_DEBUG_MEMORY=1 before launching. Provides per-stage
GPU memory snapshots so developers can identify which pipeline stage
(text encoding, denoising, VAE decode, post-processing) consumes the most
memory and where OOM is likely to occur.

Usage:
    VLLM_DEBUG_MEMORY=1 python examples/offline_inference/text_to_video/text_to_video.py \\
        --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --height 480 --width 832 --num-frames 81
"""

import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

DEBUG_MEMORY = os.environ.get("VLLM_DEBUG_MEMORY", "0") == "1"

GiB = 1024**3


class MemoryDebugTracker:
    """Collects GPU memory snapshots at named pipeline stages."""

    def __init__(self, device: torch.device | None = None):
        self.stages: list[tuple[str, float, float, float]] = []
        self.device = device

    def snapshot(self, stage_name: str) -> None:
        if not DEBUG_MEMORY:
            return
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        free, _ = torch.cuda.mem_get_info(self.device)
        self.stages.append((stage_name, alloc, reserved, free))

    def report(self) -> None:
        if not self.stages:
            return
        _, total = torch.cuda.mem_get_info(self.device)
        base_alloc = self.stages[0][1]

        sep = "=" * 85
        dash = "-" * 85
        rows = "\n".join(
            f"{name:<20s} | {alloc / GiB:>9.2f} GiB | {reserved / GiB:>9.2f} GiB | "
            f"{free / GiB:>9.2f} GiB | {(alloc - base_alloc) / GiB if i > 0 else 0:>+10.2f} GiB"
            for i, (name, alloc, reserved, free) in enumerate(self.stages)
        )
        logger.info(
            f"\n{sep}\n"
            f"GPU MEMORY BY PIPELINE STAGE (device: {self.device}, total: {total / GiB:.2f} GiB)\n"
            f"{sep}\n"
            f"{'Stage':<20s} | {'Allocated':>12s} | {'Reserved':>12s} | "
            f"{'Free':>12s} | {'Delta':>13s}\n"
            f"{dash}\n"
            f"{rows}\n"
            f"{sep}"
        )

    def reset(self) -> None:
        self.stages.clear()


def log_tensor_memory(label: str, tensor: torch.Tensor) -> None:
    """Log memory state around a tensor, useful for before/after .cpu() moves."""
    if not DEBUG_MEMORY or not isinstance(tensor, torch.Tensor):
        return
    alloc = torch.cuda.memory_allocated()
    free, total = torch.cuda.mem_get_info()
    tensor_size = tensor.nelement() * tensor.element_size()
    logger.info(
        f"[MEMORY] {label}: allocated={alloc / GiB:.2f} GiB, "
        f"free={free / GiB:.2f} GiB, tensor={tensor_size / GiB:.2f} GiB, "
        f"device={tensor.device}"
    )
