# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference for packed NVFP4 weights with FlashInfer's 128x4 scale layout."""

from functools import lru_cache

import torch
from torch import nn

from vllm_omni.platforms import current_omni_platform


@lru_cache(maxsize=32)
def require_nvfp4_linear(device: torch.device) -> None:
    capability = current_omni_platform.get_device_capability(device.index) if device.type == "cuda" else None
    if capability not in ((10, 0), (10, 3), (12, 0), (12, 1)):
        raise ValueError(f"Packed NVFP4 linear currently requires SM100/SM103/SM120/SM121; got {capability}")
    try:
        from flashinfer import mm_fp4, nvfp4_quantize
    except ImportError as exc:
        raise ImportError("Packed NVFP4 linear requires FlashInfer with nvfp4_quantize and mm_fp4") from exc
    if not callable(mm_fp4) or not callable(nvfp4_quantize):
        raise ImportError("FlashInfer does not expose the required NVFP4 APIs")


@torch.library.custom_op("vllm_omni::packed_nvfp4_linear", mutates_args=())
def _packed_nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    alpha: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> torch.Tensor:
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    require_nvfp4_linear(x.device)
    rows = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).contiguous()
    packed, input_scales = nvfp4_quantize(rows, input_global_scale, sfLayout=SfLayout.layout_128x4, enable_pdl=False)
    result = mm_fp4(
        packed,
        weight.T,
        input_scales,
        scales.T,
        alpha,
        out_dtype=torch.bfloat16,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cutlass",
        enable_pdl=False,
    )
    return result.reshape(*x.shape[:-1], weight.shape[0])


@_packed_nvfp4_linear.register_fake
def _(x, weight, scales, alpha, input_global_scale):
    return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)


class PackedNVFP4Linear(nn.Module):
    """Keep exported E2M1 weights and swizzled E4M3 scales without requantizing."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        weight_inv_scale: torch.Tensor,
        weight_global_sf: torch.Tensor,
        bias: torch.Tensor,
        input_amax: torch.Tensor,
    ) -> None:
        super().__init__()
        if packed_weight.ndim != 2 or packed_weight.dtype != torch.uint8:
            raise ValueError("NVFP4 packed_weight must be a UINT8 matrix")
        self.out_features, packed_k = packed_weight.shape
        self.in_features = packed_k * 2
        if self.out_features % 128 or self.in_features % 64:
            raise ValueError("NVFP4 128x4 layout requires N divisible by 128 and K divisible by 64")
        if (
            weight_inv_scale.dtype != torch.uint8
            or weight_inv_scale.numel() != self.out_features * self.in_features // 16
        ):
            raise ValueError("NVFP4 weight_inv_scale must contain swizzled UINT8 E4M3 scales")
        if weight_global_sf.dtype != torch.float32 or weight_global_sf.numel() != 1:
            raise ValueError("NVFP4 weight_global_sf must be a scalar FP32 tensor")
        if bias.dtype != torch.bfloat16 or bias.shape != (self.out_features,):
            raise ValueError("NVFP4 bias must be BF16 with one value per output")
        if input_amax.dtype != torch.float32 or input_amax.numel() != 1:
            raise ValueError("NVFP4 input_amax must be scalar FP32")
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_inv_scale", weight_inv_scale)
        self.register_buffer("weight_global_sf", weight_global_sf)
        self.register_buffer("bias", bias)
        self.register_buffer("input_amax", input_amax)
        # The released VAE uses a unit activation global scale.
        self.register_buffer("input_global_scale", torch.ones_like(weight_global_sf), persistent=False)
        self.register_buffer("alpha", weight_global_sf.reciprocal(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError("NVFP4 input width does not match the packed weight")
        return (
            _packed_nvfp4_linear(x, self.packed_weight, self.weight_inv_scale, self.alpha, self.input_global_scale)
            + self.bias
        )
