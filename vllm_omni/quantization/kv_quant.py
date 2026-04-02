# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 quantization utilities for diffusion attention tensors.

Provides per-tensor dynamic quantization of Q/K/V tensors to
float8_e4m3fn format. Designed for diffusion models where Q/K/V are
computed fresh each forward pass (no persistent KV cache).

Uses vLLM's fused CUDA kernel (scaled_fp8_quant) for efficient
amax+scale+cast in a single kernel launch.
"""

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Try to use vLLM's fused CUDA kernel; fall back to PyTorch ops.
try:
    from vllm._custom_ops import scaled_fp8_quant as _vllm_scaled_fp8_quant

    _HAS_FUSED_QUANT = True
except ImportError:
    _HAS_FUSED_QUANT = False
    logger.warning_once(
        "vLLM scaled_fp8_quant not available, using PyTorch ops fallback. "
        "FP8 attention will work but with higher quantization overhead."
    )


def _quantize_tensor_fp8(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a single tensor to FP8 with per-tensor dynamic scaling.

    Uses vLLM's fused CUDA kernel when available (single kernel launch
    for amax reduction + scale computation + FP8 cast). Falls back to
    3 separate PyTorch ops otherwise.

    Returns:
        ``(fp8_tensor, inv_scale)`` where inv_scale is the dequant scale.
    """
    if _HAS_FUSED_QUANT:
        # scaled_fp8_quant requires 2D input [M, N]
        orig_shape = tensor.shape
        flat = tensor.reshape(-1, orig_shape[-1])
        # Dynamic per-tensor quantization: scale=None
        fp8_flat, scale = _vllm_scaled_fp8_quant(flat)
        fp8_out = fp8_flat.reshape(orig_shape)
        # scale from vLLM is 1/scale (inv_scale / dequant scale)
        return fp8_out, scale
    else:
        finfo = torch.finfo(torch.float8_e4m3fn)
        amax = tensor.abs().amax().clamp(min=1e-12)
        scale_factor = finfo.max / amax
        fp8 = (tensor * scale_factor).clamp(finfo.min, finfo.max).to(
            torch.float8_e4m3fn
        )
        inv_scale = amax / finfo.max
        return fp8, inv_scale


def quantize_qkv_fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Quantize Q/K/V tensors to float8_e4m3fn with dynamic per-tensor scaling.

    Args:
        query: Query tensor in BF16/FP16, shape ``(B, S, H, D)``
        key: Key tensor in BF16/FP16, shape ``(B, S, H, D)``
        value: Value tensor in BF16/FP16, shape ``(B, S, H, D)``

    Returns:
        ``(fp8_query, fp8_key, fp8_value, q_scale, k_scale, v_scale)``
        where scales are inverse (dequant) scales.
        Pass as ``descale_q/k/v`` to FA3 or use :func:`dequantize_fp8`.
    """
    fp8_q, q_scale = _quantize_tensor_fp8(query)
    fp8_k, k_scale = _quantize_tensor_fp8(key)
    fp8_v, v_scale = _quantize_tensor_fp8(value)
    return fp8_q, fp8_k, fp8_v, q_scale, k_scale, v_scale


def quantize_kv_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K/V tensors to float8_e4m3fn (joint attention path).

    Returns:
        ``(fp8_key, fp8_value, k_scale, v_scale)``
    """
    fp8_k, k_scale = _quantize_tensor_fp8(key)
    fp8_v, v_scale = _quantize_tensor_fp8(value)
    return fp8_k, fp8_v, k_scale, v_scale


def dequantize_fp8(
    tensor: torch.Tensor,
    inv_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize an FP8 tensor back to the given dtype.

    Args:
        tensor: FP8-quantized tensor (float8_e4m3fn).
        inv_scale: Inverse scale (dequant scale).
        output_dtype: Target dtype (e.g. ``torch.bfloat16``).

    Returns:
        Dequantized tensor: ``tensor.to(output_dtype) * inv_scale``.
    """
    return (tensor.to(output_dtype) * inv_scale).to(output_dtype)
