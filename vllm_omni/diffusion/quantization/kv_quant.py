# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 quantization utilities for diffusion model KV tensors.

Provides per-tensor dynamic quantization of Key and Value tensors to
float8_e4m3fn format. Designed for diffusion models where K/V are computed
fresh each forward pass (no persistent KV cache).
"""

import torch


def quantize_kv_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K/V tensors to float8_e4m3fn with dynamic per-tensor scaling.

    Uses the same absmax scaling pattern as vLLM's ``input_to_float8``
    (see ``vllm/model_executor/layers/quantization/utils/fp8_utils.py``).

    Args:
        key: Key tensor in BF16/FP16, shape ``(B, S, H, D)``
        value: Value tensor in BF16/FP16, shape ``(B, S, H, D)``

    Returns:
        A tuple of ``(fp8_key, fp8_value, k_scale, v_scale)`` where scales
        are *inverse* (dequant) scales: ``inv_scale = amax / FP8_MAX``.
        Pass these scales as ``descale_k`` / ``descale_v`` to FA3 or use
        :func:`dequantize_fp8` to convert back.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)

    # Key
    k_amax = key.abs().amax().clamp(min=1e-12)
    k_scale_factor = finfo.max / k_amax
    fp8_key = (key * k_scale_factor).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    k_inv_scale = k_amax / finfo.max  # dequant scale

    # Value
    v_amax = value.abs().amax().clamp(min=1e-12)
    v_scale_factor = finfo.max / v_amax
    fp8_value = (value * v_scale_factor).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    v_inv_scale = v_amax / finfo.max  # dequant scale

    return fp8_key, fp8_value, k_inv_scale, v_inv_scale


def dequantize_fp8(
    tensor: torch.Tensor,
    inv_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize an FP8 tensor back to the given dtype.

    Args:
        tensor: FP8-quantized tensor (float8_e4m3fn).
        inv_scale: Inverse scale (dequant scale) produced by :func:`quantize_kv_fp8`.
        output_dtype: Target dtype (e.g. ``torch.bfloat16``).

    Returns:
        Dequantized tensor: ``tensor.to(output_dtype) * inv_scale``.
    """
    return tensor.to(output_dtype) * inv_scale
