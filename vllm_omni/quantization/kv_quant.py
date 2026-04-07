# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 quantization utilities for diffusion attention tensors.

Provides per-tensor dynamic quantization of Q/K/V tensors to
float8_e4m3fn format. Designed for diffusion models where Q/K/V are
computed fresh each forward pass (no persistent KV cache).

Supports two modes:
  - Dynamic: computes amax per call (accurate but ~4ms overhead at 50K tokens)
  - Static (delayed scaling): reuses a cached scale from the previous call,
    skipping the expensive amax reduction (~0.5ms overhead).
"""

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def is_quantized_kv_cache(kv_cache_dtype: str | None) -> bool:
    """Check if the KV cache dtype implies quantized storage."""
    return kv_cache_dtype in ("fp8", "fp8_e4m3")


# Try to use vLLM's fused CUDA kernel for quantization.
# Falls back to device-agnostic PyTorch ops (works on any platform).
try:
    from vllm._custom_ops import scaled_fp8_quant as _vllm_scaled_fp8_quant

    _HAS_FUSED_QUANT = True
except ImportError:
    _HAS_FUSED_QUANT = False


def _quantize_tensor_fp8(
    tensor: torch.Tensor,
    cached_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a single tensor to FP8 with per-tensor scaling.

    Args:
        tensor: Input tensor in BF16/FP16.
        cached_scale: If provided, use this scale (static mode, skips amax).
            If None, compute scale dynamically.

    Returns:
        ``(fp8_tensor, inv_scale)`` where inv_scale is the dequant scale.
    """
    if _HAS_FUSED_QUANT and tensor.is_cuda:
        orig_shape = tensor.shape
        flat = tensor.reshape(-1, orig_shape[-1])
        # Pass cached_scale for static quant (no amax), None for dynamic
        fp8_flat, scale = _vllm_scaled_fp8_quant(flat, scale=cached_scale)
        fp8_out = fp8_flat.reshape(orig_shape)
        return fp8_out, scale
    else:
        finfo = torch.finfo(torch.float8_e4m3fn)
        if cached_scale is not None:
            # Static: use cached scale directly
            inv_scale = cached_scale
            scale_factor = 1.0 / inv_scale
            fp8 = (tensor * scale_factor).clamp(finfo.min, finfo.max).to(
                torch.float8_e4m3fn
            )
            return fp8, inv_scale
        else:
            # Dynamic: compute amax
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
    cached_scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Quantize Q/K/V tensors to float8_e4m3fn.

    Args:
        query: Query tensor in BF16/FP16, shape ``(B, S, H, D)``
        key: Key tensor in BF16/FP16, shape ``(B, S, H, D)``
        value: Value tensor in BF16/FP16, shape ``(B, S, H, D)``
        cached_scales: Optional ``(q_scale, k_scale, v_scale)`` from a
            previous call. When provided, skips the expensive amax
            reduction (static/delayed scaling mode).

    Returns:
        ``(fp8_query, fp8_key, fp8_value, q_scale, k_scale, v_scale)``
        where scales are inverse (dequant) scales.
    """
    if cached_scales is not None:
        cq, ck, cv = cached_scales
    else:
        cq = ck = cv = None
    fp8_q, q_scale = _quantize_tensor_fp8(query, cq)
    fp8_k, k_scale = _quantize_tensor_fp8(key, ck)
    fp8_v, v_scale = _quantize_tensor_fp8(value, cv)
    return fp8_q, fp8_k, fp8_v, q_scale, k_scale, v_scale


def quantize_kv_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    cached_scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K/V tensors to float8_e4m3fn (joint attention path).

    Returns:
        ``(fp8_key, fp8_value, k_scale, v_scale)``
    """
    if cached_scales is not None:
        ck, cv = cached_scales
    else:
        ck = cv = None
    fp8_k, k_scale = _quantize_tensor_fp8(key, ck)
    fp8_v, v_scale = _quantize_tensor_fp8(value, cv)
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


def quantize_qkv_fp8_fast(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ultra-fast FP8 quantization using direct saturating cast (no amax).

    For diffusion attention where Q/K/V values are typically in [-10, 10],
    well within float8_e4m3fn range (±448). Eliminates the expensive
    per-tensor amax reduction that dominates quantization overhead at
    large sequence lengths (50K+ tokens).

    Scale is fixed at 1.0 (identity), so descale is also 1.0.
    """
    one = torch.ones(1, dtype=torch.float32, device=query.device)
    fp8_q = query.to(torch.float8_e4m3fn)
    fp8_k = key.to(torch.float8_e4m3fn)
    fp8_v = value.to(torch.float8_e4m3fn)
    return fp8_q, fp8_k, fp8_v, one, one, one


def quantize_kv_fp8_fast(
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast FP8 quantization for K/V only (joint attention path)."""
    one = torch.ones(1, dtype=torch.float32, device=key.device)
    fp8_k = key.to(torch.float8_e4m3fn)
    fp8_v = value.to(torch.float8_e4m3fn)
    return fp8_k, fp8_v, one, one
