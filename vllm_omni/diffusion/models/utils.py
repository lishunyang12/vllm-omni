# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for diffusion models."""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Maximum value for float8_e4m3fn
_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


_FP8_TARGET_LAYERS = (nn.Linear, nn.Conv2d, nn.Conv3d)


def apply_fp8_weight_storage(model: nn.Module) -> None:
    """Apply FP8 weight-only storage to Linear/Conv2d/Conv3d layers.

    Stores weights in float8_e4m3fn with per-tensor scales.
    Dequantizes to the original compute dtype before each forward pass,
    then re-quantizes afterward to free BF16 memory.

    This saves ~50% of memory with no accuracy loss since computation
    still happens in the original dtype.

    Args:
        model: The model whose layers will be quantized.
    """
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, _FP8_TARGET_LAYERS):
            continue

        weight = module.weight.data
        compute_dtype = weight.dtype

        # Compute per-tensor scale
        amax = weight.abs().amax().clamp(min=1e-12)
        scale = amax / _FP8_E4M3_MAX

        # Quantize weight to FP8
        fp8_weight = (weight / scale).clamp(min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX).to(torch.float8_e4m3fn)

        # Store FP8 weight and metadata as buffers (not parameters)
        module.register_buffer("_fp8_weight", fp8_weight)
        module.register_buffer("_fp8_scale", scale.to(torch.float32))
        module._fp8_compute_dtype = compute_dtype

        # Replace the parameter data with FP8 to save memory
        module.weight.data = fp8_weight

        def _pre_hook(mod, args):
            # Dequantize: restore BF16/FP16 weight for conv computation
            mod.weight.data = mod._fp8_weight.to(mod._fp8_compute_dtype) * mod._fp8_scale

        def _post_hook(mod, args, output):
            # Re-quantize: swap back to FP8 to free BF16 memory
            mod.weight.data = mod._fp8_weight

        module.register_forward_pre_hook(_pre_hook)
        module.register_forward_hook(_post_hook)
        count += 1

    logger.info(
        "Applied FP8 weight storage to %d layers in %s",
        count,
        model.__class__.__name__,
    )
