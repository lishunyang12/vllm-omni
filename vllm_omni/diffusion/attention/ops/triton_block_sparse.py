# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""BF16 VSA on Ada and newer GPUs using FastVideo's portable Triton kernel."""

import math
from collections.abc import Callable
from functools import lru_cache

import torch

from vllm_omni.platforms import current_omni_platform

from .block_sparse import block_map_to_indices, validate_sparse_inputs


@lru_cache(maxsize=32)
def require_triton_sparse(device: torch.device) -> Callable:
    capability = current_omni_platform.get_device_capability(device.index) if device.type == "cuda" else None
    if capability is None or capability < (8, 0):
        raise ValueError(f"Triton BF16 block-sparse attention requires SM80 or newer; got {capability}")
    try:
        from fastvideo_kernel.block_sparse_attn import block_sparse_attn_triton
    except ImportError as exc:
        raise ImportError("Triton VSA requires fastvideo-kernel; install vllm-omni[vsa]") from exc
    return block_sparse_attn_triton


def triton_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    block_sizes: torch.Tensor | None,
    softmax_scale: float,
    block_size: int,
) -> torch.Tensor:
    if block_size not in (64, 128):
        raise ValueError("Triton VSA requires logical tile64 or tile128")
    if not math.isclose(softmax_scale, 128**-0.5, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Triton VSA requires head_size**-0.5 scaling")
    if query.shape[1] % block_size or key.shape[1] % block_size:
        raise ValueError("Triton VSA inputs must be padded to complete logical tiles")
    key_blocks = key.shape[1] // block_size
    if block_sizes is None or block_sizes.shape != (key_blocks,):
        raise ValueError("Triton VSA requires one valid-token count per KV tile")
    if block_map.shape[-1] != key_blocks:
        raise ValueError("Triton VSA block_map must match the KV tile count")
    indices, counts = block_map_to_indices(block_map)
    validate_sparse_inputs(query, key, value, indices, counts, block_sizes, block_size)
    if block_size == 128:
        # Both halves share the original tile selection and ONE softmax over
        # all selected keys. Pooling/top-k/compression stay at logical tile128.
        block_map = block_map.repeat_interleave(2, -2).repeat_interleave(2, -1)
        block_sizes = torch.stack((block_sizes.clamp(max=64), (block_sizes - 64).clamp(0, 64)), -1).flatten()
    block_map = block_map & (block_sizes > 0)[None, None, None, :]
    indices, counts = block_map_to_indices(block_map)
    kernel = require_triton_sparse(query.device)
    output, _ = kernel(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        indices,
        counts,
        block_sizes,
    )
    # The upstream kernel divides by a zero denominator for empty rows.
    # Define those rows as zero without hiding non-finite nonempty results.
    output = output.masked_fill((counts == 0).repeat_interleave(64, -1).unsqueeze(-1), 0)
    return output.transpose(1, 2).contiguous()
