# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Q/K RMSNorm followed by packed non-interleaved RoPE.

The public contract is shared by diffusion attention implementations:

* ``q`` and ``k`` are ``[T, heads, head_dim]``;
* the norm weights are one-dimensional ``[head_dim]`` tensors;
* ``rope_table`` is ``[T, rotary_dim]`` and stores
  ``[cos(theta), sin(theta)]`` with ``theta`` of width ``rotary_dim // 2``.

For MiniMax H3's ``head_dim == 128`` BF16/FP16 path, the implementation keeps
RMSNorm in the native PyTorch kernel and uses two token-tiled Triton launches
for RoPE. The first launch stores each product in the input dtype; the second
launch performs the add/subtract. These explicit global boundaries preserve
bitwise equality with the eager reference. Other dtypes, shapes, and devices
use the eager implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.library import Library
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_LOSSLESS_HEAD_DIM = 128
_ROPE_TOKEN_BLOCK = 8
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
    def _rope_products_kernel(
        q_norm_ptr,
        k_norm_ptr,
        p1_ptr,
        p2_ptr,
        rope_table_ptr,
        q_norm_stride_t,
        q_norm_stride_h,
        q_norm_stride_d,
        k_norm_stride_t,
        k_norm_stride_h,
        k_norm_stride_d,
        p1_stride_t,
        p1_stride_h,
        p1_stride_d,
        p2_stride_t,
        p2_stride_h,
        p2_stride_d,
        rope_stride_t,
        num_q_heads,
        num_tokens,
        rotary_dim: tl.constexpr,
        rotary_half_block: tl.constexpr,
        token_block: tl.constexpr,
        input_dtype: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * token_block + tl.arange(0, token_block)
        token_mask = token_offsets < num_tokens
        head = tl.program_id(1)
        is_k = head >= num_q_heads
        local_head = tl.where(is_k, head - num_q_heads, head)

        x_base = tl.where(
            is_k,
            k_norm_ptr + local_head * k_norm_stride_h,
            q_norm_ptr + local_head * q_norm_stride_h,
        )
        x_stride_t = tl.where(is_k, k_norm_stride_t, q_norm_stride_t)
        x_stride_d = tl.where(is_k, k_norm_stride_d, q_norm_stride_d)

        half_offsets = tl.arange(0, rotary_half_block)
        half_mask = half_offsets < (rotary_dim // 2)
        second_offsets = rotary_dim // 2 + half_offsets
        token_base = x_base + token_offsets[:, None] * x_stride_t
        mask = token_mask[:, None] & half_mask[None, :]
        first = tl.load(
            token_base + half_offsets[None, :] * x_stride_d,
            mask=mask,
            other=0.0,
        ).to(input_dtype)
        second = tl.load(
            token_base + second_offsets[None, :] * x_stride_d,
            mask=mask,
            other=0.0,
        ).to(input_dtype)

        rope_base = rope_table_ptr + token_offsets[:, None] * rope_stride_t
        cos = tl.load(
            rope_base + half_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(input_dtype)
        sin = tl.load(
            rope_base + second_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(input_dtype)

        p1_base = p1_ptr + token_offsets[:, None] * p1_stride_t + head * p1_stride_h
        p2_base = p2_ptr + token_offsets[:, None] * p2_stride_t + head * p2_stride_h
        tl.store(
            p1_base + half_offsets[None, :] * p1_stride_d,
            first * cos,
            mask=mask,
        )
        tl.store(
            p1_base + second_offsets[None, :] * p1_stride_d,
            second * sin,
            mask=mask,
        )
        tl.store(
            p2_base + half_offsets[None, :] * p2_stride_d,
            second * cos,
            mask=mask,
        )
        tl.store(
            p2_base + second_offsets[None, :] * p2_stride_d,
            first * sin,
            mask=mask,
        )


    @triton.jit
    def _rope_combine_kernel(
        q_norm_ptr,
        k_norm_ptr,
        p1_ptr,
        p2_ptr,
        q_out_ptr,
        k_out_ptr,
        q_norm_stride_t,
        q_norm_stride_h,
        q_norm_stride_d,
        k_norm_stride_t,
        k_norm_stride_h,
        k_norm_stride_d,
        p1_stride_t,
        p1_stride_h,
        p1_stride_d,
        p2_stride_t,
        p2_stride_h,
        p2_stride_d,
        q_out_stride_t,
        q_out_stride_h,
        q_out_stride_d,
        k_out_stride_t,
        k_out_stride_h,
        k_out_stride_d,
        num_q_heads,
        num_tokens,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        rotary_half_block: tl.constexpr,
        tail_block: tl.constexpr,
        token_block: tl.constexpr,
        input_dtype: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * token_block + tl.arange(0, token_block)
        token_mask = token_offsets < num_tokens
        head = tl.program_id(1)
        is_k = head >= num_q_heads
        local_head = tl.where(is_k, head - num_q_heads, head)

        in_base = tl.where(
            is_k,
            k_norm_ptr + local_head * k_norm_stride_h,
            q_norm_ptr + local_head * q_norm_stride_h,
        )
        out_base = tl.where(
            is_k,
            k_out_ptr + local_head * k_out_stride_h,
            q_out_ptr + local_head * q_out_stride_h,
        )
        in_stride_t = tl.where(is_k, k_norm_stride_t, q_norm_stride_t)
        in_stride_d = tl.where(is_k, k_norm_stride_d, q_norm_stride_d)
        out_stride_t = tl.where(is_k, k_out_stride_t, q_out_stride_t)
        out_stride_d = tl.where(is_k, k_out_stride_d, q_out_stride_d)

        half_offsets = tl.arange(0, rotary_half_block)
        half_mask = half_offsets < (rotary_dim // 2)
        second_offsets = rotary_dim // 2 + half_offsets
        mask = token_mask[:, None] & half_mask[None, :]
        p1_base = p1_ptr + token_offsets[:, None] * p1_stride_t + head * p1_stride_h
        p2_base = p2_ptr + token_offsets[:, None] * p2_stride_t + head * p2_stride_h
        first_cos = tl.load(
            p1_base + half_offsets[None, :] * p1_stride_d,
            mask=mask,
            other=0.0,
        ).to(input_dtype)
        second_sin = tl.load(
            p1_base + second_offsets[None, :] * p1_stride_d,
            mask=mask,
            other=0.0,
        ).to(input_dtype)
        second_cos = tl.load(
            p2_base + half_offsets[None, :] * p2_stride_d,
            mask=mask,
            other=0.0,
        ).to(input_dtype)
        first_sin = tl.load(
            p2_base + second_offsets[None, :] * p2_stride_d,
            mask=mask,
            other=0.0,
        ).to(input_dtype)

        out_base = out_base + token_offsets[:, None] * out_stride_t
        tl.store(
            out_base + half_offsets[None, :] * out_stride_d,
            first_cos - second_sin,
            mask=mask,
        )
        tl.store(
            out_base + second_offsets[None, :] * out_stride_d,
            second_cos + first_sin,
            mask=mask,
        )

        tail_offsets = rotary_dim + tl.arange(0, tail_block)
        tail_mask = token_mask[:, None] & (tail_offsets[None, :] < head_dim)
        in_base = in_base + token_offsets[:, None] * in_stride_t
        tail = tl.load(
            in_base + tail_offsets[None, :] * in_stride_d,
            mask=tail_mask,
            other=0.0,
        ).to(input_dtype)
        tl.store(
            out_base + tail_offsets[None, :] * out_stride_d,
            tail,
            mask=tail_mask,
        )


def _triton_input_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    raise TypeError(f"Lossless fused QK RMSNorm/RoPE only supports BF16/FP16, got {dtype}")


def _lossless_cuda_supported(q: torch.Tensor, head_dim: int) -> bool:
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and head_dim == _LOSSLESS_HEAD_DIM
    )


def _eager_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_norm = F.rms_norm(q, (head_dim,), q_weight, eps)
    k_norm = F.rms_norm(k, (head_dim,), k_weight, eps)
    return (
        _apply_rope_table(q_norm, rope_table, rotary_dim),
        _apply_rope_table(k_norm, rope_table, rotary_dim),
    )


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
    if not _lossless_cuda_supported(q, head_dim):
        return _eager_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps, head_dim, rotary_dim)

    # Keep RMSNorm native so its reduction and dtype behavior are exactly the
    # same as the eager reference on every CUDA architecture.
    q_norm = F.rms_norm(q, (head_dim,), q_weight, eps)
    k_norm = F.rms_norm(k, (head_dim,), k_weight, eps)
    if q.shape[0] == 0:
        return q_norm, k_norm

    num_q_heads = q.shape[1]
    num_heads = num_q_heads + k.shape[1]
    input_dtype = _triton_input_dtype(q.dtype)
    rotary_half_block = triton.next_power_of_2(rotary_dim // 2)
    tail_block = triton.next_power_of_2(max(1, head_dim - rotary_dim))
    p1 = torch.empty(
        (q.shape[0], num_heads, rotary_dim),
        device=q.device,
        dtype=q.dtype,
    )
    p2 = torch.empty_like(p1)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    grid = (triton.cdiv(q.shape[0], _ROPE_TOKEN_BLOCK), num_heads)

    _rope_products_kernel[grid](
        q_norm,
        k_norm,
        p1,
        p2,
        rope_table,
        q_norm.stride(0),
        q_norm.stride(1),
        q_norm.stride(2),
        k_norm.stride(0),
        k_norm.stride(1),
        k_norm.stride(2),
        p1.stride(0),
        p1.stride(1),
        p1.stride(2),
        p2.stride(0),
        p2.stride(1),
        p2.stride(2),
        rope_table.stride(0),
        num_q_heads,
        q.shape[0],
        rotary_dim=rotary_dim,
        rotary_half_block=rotary_half_block,
        token_block=_ROPE_TOKEN_BLOCK,
        input_dtype=input_dtype,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _rope_combine_kernel[grid](
        q_norm,
        k_norm,
        p1,
        p2,
        q_out,
        k_out,
        q_norm.stride(0),
        q_norm.stride(1),
        q_norm.stride(2),
        k_norm.stride(0),
        k_norm.stride(1),
        k_norm.stride(2),
        p1.stride(0),
        p1.stride(1),
        p1.stride(2),
        p2.stride(0),
        p2.stride(1),
        p2.stride(2),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        num_q_heads,
        q.shape[0],
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rotary_half_block=rotary_half_block,
        tail_block=tail_block,
        token_block=_ROPE_TOKEN_BLOCK,
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
    """Apply Q/K RMSNorm and packed non-interleaved RoPE in one custom op.

    The CUDA Triton implementation is lossless for the validated H3 geometry;
    all other inputs use the eager reference path.
    """
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
