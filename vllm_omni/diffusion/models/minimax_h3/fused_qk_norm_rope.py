# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused MiniMax H3 Q/K RMSNorm and 3D RoPE.

The H3 attention head is 128-wide and rotates the first 96 dimensions.  The
frequency tensor contains two identical 48-wide halves, so the compiled
request path stores only ``[cos(freqs[:48]), sin(freqs[:48])]``.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_H3_HEAD_DIM = 128
_H3_ROTARY_DIM = 96
_H3_ROTARY_HALF_DIM = _H3_ROTARY_DIM // 2


def _apply_rope_table(
    x: torch.Tensor,
    rope_table: torch.Tensor,
) -> torch.Tensor:
    """Reference H3 RoPE for a packed ``[cos, sin]`` table."""
    half = _H3_ROTARY_HALF_DIM
    cos = rope_table[..., :half].to(x.dtype).unsqueeze(1)
    sin = rope_table[..., half:].to(x.dtype).unsqueeze(1)
    x_rot = x[..., :_H3_ROTARY_DIM]
    x_first, x_second = x_rot[..., :half], x_rot[..., half:]
    rotated_first = (x_first * cos) - (x_second * sin)
    rotated_second = (x_second * cos) + (x_first * sin)
    return torch.cat((rotated_first, rotated_second, x[..., _H3_ROTARY_DIM:]), dim=-1)


if HAS_TRITON:

    @triton.jit
    def _h3_qk_norm_rope_kernel(
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
        num_q_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        eps: tl.constexpr,
        input_dtype: tl.constexpr,
        head_block: tl.constexpr,
        rot_half_block: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        is_k = head >= num_q_heads
        local_head = tl.where(is_k, head - num_q_heads, head)

        if is_k:
            in_base = k_ptr + token * k_stride_t + local_head * k_stride_h
            out_base = k_out_ptr + token * k_out_stride_t + local_head * k_out_stride_h
            weight_ptr = k_weight_ptr
            in_stride_d = k_stride_d
            out_stride_d = k_out_stride_d
        else:
            in_base = q_ptr + token * q_stride_t + local_head * q_stride_h
            out_base = q_out_ptr + token * q_out_stride_t + local_head * q_out_stride_h
            weight_ptr = q_weight_ptr
            in_stride_d = q_stride_d
            out_stride_d = q_out_stride_d

        head_offsets = tl.arange(0, head_block)
        head_mask = head_offsets < head_dim
        x = tl.load(
            in_base + head_offsets * in_stride_d,
            mask=head_mask,
            other=0.0,
        ).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / head_dim
        inv_rms = tl.rsqrt(variance + eps)

        # Keep the non-rotary tail in the same output allocation. The cast
        # before the store matches the eager RMSNorm BF16/FP16 boundary.
        tail_mask = head_mask & (head_offsets >= rotary_dim)
        tail_weight = tl.load(
            weight_ptr + head_offsets,
            mask=tail_mask,
            other=0.0,
        ).to(tl.float32)
        tail = (x * inv_rms * tail_weight).to(input_dtype)
        tl.store(out_base + head_offsets * out_stride_d, tail, mask=tail_mask)

        # Triton does not provide a cheap slice of a register block.  Reload
        # the rotated halves using the already computed inverse RMS.  This is
        # the same tradeoff used by the existing vLLM fused QK/RoPE kernel.
        rot_offsets = tl.arange(0, rot_half_block)
        rot_mask = rot_offsets < (rotary_dim // 2)
        first = tl.load(
            in_base + rot_offsets * in_stride_d,
            mask=rot_mask,
            other=0.0,
        ).to(tl.float32)
        second = tl.load(
            in_base + (rotary_dim // 2 + rot_offsets) * in_stride_d,
            mask=rot_mask,
            other=0.0,
        ).to(tl.float32)
        first_weight = tl.load(weight_ptr + rot_offsets, mask=rot_mask, other=0.0).to(tl.float32)
        second_weight = tl.load(
            weight_ptr + rotary_dim // 2 + rot_offsets,
            mask=rot_mask,
            other=0.0,
        ).to(tl.float32)
        first = (first * inv_rms * first_weight).to(input_dtype).to(tl.float32)
        second = (second * inv_rms * second_weight).to(input_dtype).to(tl.float32)

        table_base = rope_table_ptr + token * rope_stride_t
        cos = tl.load(table_base + rot_offsets, mask=rot_mask, other=0.0).to(tl.float32)
        sin = tl.load(
            table_base + rotary_dim // 2 + rot_offsets,
            mask=rot_mask,
            other=0.0,
        ).to(tl.float32)
        rotated_first = first * cos - second * sin
        rotated_second = second * cos + first * sin
        tl.store(out_base + rot_offsets * out_stride_d, rotated_first, mask=rot_mask)
        tl.store(
            out_base + (rotary_dim // 2 + rot_offsets) * out_stride_d,
            rotated_second,
            mask=rot_mask,
        )


def _h3_qk_norm_rope_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not HAS_TRITON or not q.is_cuda:
        q_norm = torch.nn.functional.rms_norm(q, (q.shape[-1],), q_weight, eps)
        k_norm = torch.nn.functional.rms_norm(k, (k.shape[-1],), k_weight, eps)
        return _apply_rope_table(q_norm, rope_table), _apply_rope_table(k_norm, rope_table)

    if q.dtype == torch.bfloat16:
        input_dtype = tl.bfloat16
    elif q.dtype == torch.float16:
        input_dtype = tl.float16
    elif q.dtype == torch.float32:
        input_dtype = tl.float32
    else:
        raise TypeError(f"MiniMax H3 fused QK/RoPE only supports floating inputs, got {q.dtype}")

    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if q.shape[0] == 0:
        return q_out, k_out

    head_block = triton.next_power_of_2(q.shape[-1])
    rot_half_block = triton.next_power_of_2(_H3_ROTARY_HALF_DIM)
    grid = (q.shape[0], q.shape[1] + k.shape[1])
    _h3_qk_norm_rope_kernel[grid](
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
        head_dim=q.shape[2],
        rotary_dim=_H3_ROTARY_DIM,
        eps=eps,
        input_dtype=input_dtype,
        head_block=head_block,
        rot_half_block=rot_half_block,
        num_warps=2,
        num_stages=2,
    )
    return q_out, k_out


def _h3_qk_norm_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, rope_table, eps
    return torch.empty_like(q), torch.empty_like(k)


direct_register_custom_op(
    op_name="minimax_h3_qk_norm_rope",
    op_func=_h3_qk_norm_rope_cuda,
    fake_impl=_h3_qk_norm_rope_fake,
    mutates_args=[],
)


def fused_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply H3 Q/K RMSNorm followed by 3D RoPE in one custom op."""
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"q and k must be [T, heads, head_dim], got {q.shape} and {k.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k shapes are incompatible: {q.shape} and {k.shape}")
    if q.dtype != k.dtype:
        raise ValueError(f"q and k must have the same dtype, got {q.dtype} and {k.dtype}")
    if q.device != k.device:
        raise ValueError(f"q and k must be on the same device, got {q.device} and {k.device}")
    if q.shape[2] != _H3_HEAD_DIM:
        raise ValueError(f"MiniMax H3 fused QK/RoPE expects head_dim=128, got {q.shape[2]}")
    if q_weight.shape != (_H3_HEAD_DIM,) or k_weight.shape != (_H3_HEAD_DIM,):
        raise ValueError(
            "MiniMax H3 fused QK/RoPE expects one 128-element weight per norm, "
            f"got {tuple(q_weight.shape)} and {tuple(k_weight.shape)}"
        )
    if q_weight.device != q.device or k_weight.device != q.device:
        raise ValueError("MiniMax H3 fused QK/RoPE weights must be on the activation device")
    if rope_table.device != q.device:
        raise ValueError("MiniMax H3 fused QK/RoPE table must be on the activation device")
    if rope_table.shape != (q.shape[0], _H3_ROTARY_DIM):
        raise ValueError(
            f"MiniMax H3 rope_table must be [T, 96] containing [cos(48), sin(48)], got {tuple(rope_table.shape)}"
        )
    if not rope_table.is_contiguous():
        rope_table = rope_table.contiguous()
    return torch.ops.vllm.minimax_h3_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps)


__all__ = ["fused_qk_rmsnorm_rope"]
