# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for packing a dense (B, S, H, D) attention batch into the ragged/varlen
layout that flash-attn / trtllm-gen kernels consume."""

import torch


def varlen_cu_seqlens(batch_size: int, seqlen: int, device: torch.device) -> torch.Tensor:
    """cu_seqlens [0, S, 2S, ..., B*S] (int32) for a uniform (B, S) batch flattened to
    (B*S, H, D). Centralizes the convention so it can't drift across backends."""
    return torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device
    )
