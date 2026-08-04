# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused Q/K RMSNorm followed by packed non-interleaved RoPE.

The public contract is shared by diffusion attention implementations:

* ``q`` and ``k`` are ``[T, heads, head_dim]``;
* the norm weights are one-dimensional ``[head_dim]`` tensors;
* ``rope_table`` is ``[T, rotary_dim]`` and stores
  ``[cos(theta), sin(theta)]`` with ``theta`` of width ``rotary_dim // 2``.

The CUDA path uses a token-tiled Triton kernel.  The implementation is kept
here instead of in a model directory so models with the same RoPE contract do
not need to duplicate the custom-op registration or kernel dispatch.  Model-
specific positional-frequency construction remains in each model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.library import Library
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_TOKEN_BLOCK = 8
_NUM_WARPS = 1
_NUM_STAGES = 2


def _apply_rope_table(
    x: torch.Tensor,
    rope_table: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Reference implementation for the packed non-interleaved RoPE layout."""
    half = rotary_dim // 2
    cos = rope_table[..., :half].to(x.dtype).unsqueeze(1)
    sin = rope_table[..., half:].to(x.dtype).unsqueeze(1)
    x_first = x[..., :half]
    x_second = x[..., half:rotary_dim]
    rotated_first = (x_first * cos) - (x_second * sin)
    rotated_second = (x_second * cos) + (x_first * sin)
    return torch.cat((rotated_first, rotated_second, x[..., rotary_dim:]), dim=-1)


if HAS_TRITON:

    @triton.jit
    def _fused_qk_norm_rope_kernel(
        q_ptr,
        k_ptr,
        q_out_ptr,
        k_out_ptr,
        q_weight_ptr,
        k_weight_ptr,
        rope_table_ptr,
        q_stride_t,
        q_stride_h,
        q_stride_d,
        k_stride_t,
        k_stride_h,
        k_stride_d,
        q_out_stride_t,
        q_out_stride_h,
        q_out_stride_d,
        k_out_stride_t,
        k_out_stride_h,
        k_out_stride_d,
        rope_stride_t,
        num_q_heads,
        num_tokens,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        rotary_half_block: tl.constexpr,
        tail_block: tl.constexpr,
        token_block: tl.constexpr,
        eps: tl.constexpr,
        input_dtype: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * token_block + tl.arange(0, token_block)
        token_mask = token_offsets < num_tokens
        head = tl.program_id(1)
        is_k = head >= num_q_heads
        local_head = tl.where(is_k, head - num_q_heads, head)

        in_base = tl.where(
            is_k,
            k_ptr + local_head * k_stride_h,
            q_ptr + local_head * q_stride_h,
        )
        out_base = tl.where(
            is_k,
            k_out_ptr + local_head * k_out_stride_h,
            q_out_ptr + local_head * q_out_stride_h,
        )
        in_stride_d = tl.where(is_k, k_stride_d, q_stride_d)
        in_stride_t = tl.where(is_k, k_stride_t, q_stride_t)
        out_stride_d = tl.where(is_k, k_out_stride_d, q_out_stride_d)
        out_stride_t = tl.where(is_k, k_out_stride_t, q_out_stride_t)
        weight_ptr = tl.where(is_k, k_weight_ptr, q_weight_ptr)

        half_offsets = tl.arange(0, rotary_half_block)
        half_mask = half_offsets < (rotary_dim // 2)
        second_offsets = rotary_dim // 2 + half_offsets
        tail_offsets = rotary_dim + tl.arange(0, tail_block)
        tail_mask = tail_offsets < head_dim

        token_base = in_base + token_offsets[:, None] * in_stride_t
        first = tl.load(
            token_base + half_offsets[None, :] * in_stride_d,
            mask=token_mask[:, None] & half_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        second = tl.load(
            token_base + second_offsets[None, :] * in_stride_d,
            mask=token_mask[:, None] & half_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        tail = tl.load(
            token_base + tail_offsets[None, :] * in_stride_d,
            mask=token_mask[:, None] & tail_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        variance = (
            tl.sum(first * first, axis=1) + tl.sum(second * second, axis=1) + tl.sum(tail * tail, axis=1)
        ) / head_dim
        inv_rms = tl.rsqrt(variance + eps)

        first_weight = tl.load(
            weight_ptr + half_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        second_weight = tl.load(
            weight_ptr + second_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        tail_weight = tl.load(
            weight_ptr + tail_offsets,
            mask=tail_mask,
            other=0.0,
        ).to(tl.float32)
        first = (first * inv_rms[:, None] * first_weight[None, :]).to(input_dtype).to(tl.float32)
        second = (second * inv_rms[:, None] * second_weight[None, :]).to(input_dtype).to(tl.float32)
        tail = (tail * inv_rms[:, None] * tail_weight[None, :]).to(input_dtype)

        table_base = rope_table_ptr + token_offsets[:, None] * rope_stride_t
        cos = tl.load(
            table_base + half_offsets[None, :],
            mask=token_mask[:, None] & half_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        sin = tl.load(
            table_base + (rotary_dim // 2 + half_offsets)[None, :],
            mask=token_mask[:, None] & half_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        rotated_first = first * cos - second * sin
        rotated_second = second * cos + first * sin

        out_token_base = out_base + token_offsets[:, None] * out_stride_t
        tl.store(
            out_token_base + half_offsets[None, :] * out_stride_d,
            rotated_first,
            mask=token_mask[:, None] & half_mask[None, :],
        )
        tl.store(
            out_token_base + second_offsets[None, :] * out_stride_d,
            rotated_second,
            mask=token_mask[:, None] & half_mask[None, :],
        )
        tl.store(
            out_token_base + tail_offsets[None, :] * out_stride_d,
            tail,
            mask=token_mask[:, None] & tail_mask[None, :],
        )


def _triton_input_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise TypeError(f"Fused QK RMSNorm/RoPE only supports floating inputs, got {dtype}")


def _fused_qk_norm_rope_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not HAS_TRITON or not current_platform.is_cuda() or not q.is_cuda:
        q_norm = F.rms_norm(q, (head_dim,), q_weight, eps)
        k_norm = F.rms_norm(k, (head_dim,), k_weight, eps)
        return (
            _apply_rope_table(q_norm, rope_table, rotary_dim),
            _apply_rope_table(k_norm, rope_table, rotary_dim),
        )

    input_dtype = _triton_input_dtype(q.dtype)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    if q.shape[0] == 0:
        return q_out, k_out

    rotary_half_block = triton.next_power_of_2(rotary_dim // 2)
    tail_block = triton.next_power_of_2(max(1, head_dim - rotary_dim))
    grid = (triton.cdiv(q.shape[0], _TOKEN_BLOCK), q.shape[1] + k.shape[1])
    _fused_qk_norm_rope_kernel[grid](
        q,
        k,
        q_out,
        k_out,
        q_weight,
        k_weight,
        rope_table,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        rope_table.stride(0),
        num_q_heads=q.shape[1],
        num_tokens=q.shape[0],
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rotary_half_block=rotary_half_block,
        tail_block=tail_block,
        token_block=_TOKEN_BLOCK,
        eps=eps,
        input_dtype=input_dtype,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return q_out, k_out


def _fused_qk_norm_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, rope_table, eps, head_dim, rotary_dim
    return torch.empty_like(q), torch.empty_like(k)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "fused_qk_norm_rope"):
    direct_register_custom_op(
        op_name="fused_qk_norm_rope",
        op_func=_fused_qk_norm_rope_impl,
        fake_impl=_fused_qk_norm_rope_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    *,
    head_dim: int | None = None,
    rotary_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Q/K RMSNorm and packed non-interleaved RoPE in one custom op."""
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"q and k must be [T, heads, head_dim], got {q.shape} and {k.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k shapes are incompatible: {q.shape} and {k.shape}")
    if q.dtype != k.dtype:
        raise ValueError(f"q and k must have the same dtype, got {q.dtype} and {k.dtype}")
    if q.device != k.device:
        raise ValueError(f"q and k must be on the same device, got {q.device} and {k.device}")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"Fused QK RMSNorm/RoPE only supports floating inputs, got {q.dtype}")

    head_dim = q.shape[-1] if head_dim is None else head_dim
    rotary_dim = rope_table.shape[-1] if rotary_dim is None else rotary_dim
    if q.shape[-1] != head_dim:
        raise ValueError(f"Expected q/k head_dim={head_dim}, got {q.shape[-1]}")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and in [2, {head_dim}], got {rotary_dim}")
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError(
            f"Expected one norm weight of shape [{head_dim}], got {tuple(q_weight.shape)} and {tuple(k_weight.shape)}"
        )
    if q_weight.device != q.device or k_weight.device != q.device:
        raise ValueError("Q/K norm weights must be on the activation device")
    if rope_table.device != q.device:
        raise ValueError("RoPE table must be on the activation device")
    if rope_table.dtype != q.dtype:
        raise ValueError(f"RoPE table must have dtype {q.dtype}, got {rope_table.dtype}")
    if rope_table.shape != (q.shape[0], rotary_dim):
        raise ValueError(
            f"Expected rope_table [{q.shape[0]}, {rotary_dim}] containing [cos, sin], got {tuple(rope_table.shape)}"
        )
    if not q_weight.is_contiguous():
        q_weight = q_weight.contiguous()
    if not k_weight.is_contiguous():
        k_weight = k_weight.contiguous()
    if not rope_table.is_contiguous():
        rope_table = rope_table.contiguous()

    if not HAS_TRITON or not current_platform.is_cuda() or not q.is_cuda:
        return _fused_qk_norm_rope_impl(q, k, q_weight, k_weight, rope_table, eps, head_dim, rotary_dim)
    return torch.ops.vllm_omni.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        eps,
        head_dim,
        rotary_dim,
    )


__all__ = ["fused_qk_norm_rope"]
