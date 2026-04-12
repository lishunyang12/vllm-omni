# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone NVFP4 LinearMethod for BFL FLUX.2 checkpoints.

Ported 1:1 from sgl-project/sglang main branch (PR #20137 + #22064).
Uses flashinfer's raw FP4 APIs (fp4_quantize + mm_fp4) directly instead
of vLLM's wrappers, because the two frameworks have different conventions
for transpose and scale layout:

  - SGLang / flashinfer.mm_fp4: weight and weight_scale are TRANSPOSED
  - vLLM / flashinfer_scaled_fp4_mm: weight and weight_scale are NOT transposed

Using the wrong convention produces numerically-stable but semantically
garbage output (orange textured squares).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)

logger = logging.getLogger(__name__)


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _pad_nvfp4_weight(
    weight: torch.Tensor,
    alignment: int = 32,
) -> tuple[torch.Tensor, int]:
    """Pad packed NVFP4 weight [N, K//2] to alignment boundaries.

    Ported from SGLang's ``pad_nvfp4_weight``.
    """
    n_rows = weight.shape[0]
    k_bytes = weight.shape[1]
    k_elements = k_bytes * 2

    pad_rows = 0
    if n_rows % alignment != 0:
        pad_rows = _round_up(n_rows, alignment) - n_rows

    pad_cols_bytes = 0
    if k_elements % alignment != 0:
        pad_cols = _round_up(k_elements, alignment) - k_elements
        pad_cols_bytes = pad_cols // 2

    if pad_rows > 0 or pad_cols_bytes > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_cols_bytes, 0, pad_rows)).contiguous()

    return weight, pad_cols_bytes


def _pad_nvfp4_activation(x_fp4: torch.Tensor, padding_cols: int) -> torch.Tensor:
    """Pad activation to match weight K-dimension padding."""
    if padding_cols > 0:
        x_fp4 = torch.nn.functional.pad(x_fp4, (0, padding_cols)).contiguous()
    return x_fp4


def _slice_output(out: torch.Tensor, output_size: int) -> torch.Tensor:
    """Remove N-dimension padding from output."""
    if out.shape[-1] != output_size:
        out = out[..., :output_size]
    return out


@lru_cache(maxsize=1)
def _get_flashinfer_ops():
    """Get flashinfer's raw FP4 quantize and GEMM ops.

    Returns (fp4_quantize, mm_fp4, backend_str).
    """
    try:
        from flashinfer import fp4_quantize, mm_fp4

        # On Blackwell (SM120), cuDNN backend is preferred by SGLang.
        # For safety, use "cutlass" which works across SM100/SM120.
        backend = "cutlass"
        return fp4_quantize, mm_fp4, backend
    except ImportError:
        logger.error("flashinfer not available — cannot run NVFP4 inference")
        return None, None, None


class FluxNvFp4LinearMethod(QuantizeMethodBase):
    """Standalone NVFP4 linear method for BFL FLUX.2 checkpoints.

    Ported from SGLang's ``ModelOptFp4LinearMethod``. Uses flashinfer's
    raw ``fp4_quantize`` + ``mm_fp4`` APIs with the SGLang calling
    convention (transposed weight/scale for the flashinfer backend).
    """

    def __init__(self, quant_config: Any) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del input_size, output_size

        if input_size_per_partition % 16 != 0:
            raise ValueError(f"NVFP4 requires input_size divisible by 16, got {input_size_per_partition}")

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        input_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_scale", input_scale)

        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)

        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Exact port of SGLang's ModelOptFp4LinearMethod.process_weights_after_loading."""

        # 1. Global scales → alpha + input_scale_inv
        input_scale_2 = layer.input_scale.max().to(torch.float32)
        weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)

        layer.alpha = Parameter(
            (input_scale_2 * weight_scale_2).to(torch.float32),
            requires_grad=False,
        )
        layer.input_scale_inv = Parameter(
            (1.0 / input_scale_2).to(torch.float32),
            requires_grad=False,
        )

        del layer.input_scale
        del layer.weight_scale_2

        layer.output_size_per_partition = layer.weight.shape[0]

        # 2. Nibble swap (BFL packs high/low nibble in opposite order)
        w = layer.weight.data
        w_swapped = ((w >> 4) | (w << 4)).contiguous()

        # 3. Pad weight to 32-element alignment
        weight, weights_padding_cols = _pad_nvfp4_weight(w_swapped)
        layer.weights_padding_cols = weights_padding_cols
        layer.weight = Parameter(weight, requires_grad=False)

        # 4. Pad + blockwise-interleave weight_scale for CUTLASS TMA
        scales = layer.weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        assert scales.ndim == 3
        B, M, K = scales.shape
        M_padded = _round_up(M, 128)
        K_padded = _round_up(K, 4)
        padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype)
        padded_scales[:B, :M, :K] = scales
        # Blockwise interleave for CUTLASS TMA layout (from SGLang #22064)
        padded_scales = padded_scales.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
        padded_scales = padded_scales.permute(0, 1, 4, 3, 2, 5)
        padded_scales = padded_scales.contiguous().cuda()
        padded_scales = (
            padded_scales.reshape(M_padded, K_padded)
            if scale_ndim == 2
            else padded_scales.reshape(B, M_padded, K_padded)
        )
        # Store as separate attribute (SGLang convention)
        layer.weight_scale_interleaved = Parameter(padded_scales, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exact port of SGLang's ModelOptFp4LinearMethod.apply."""

        fp4_quantize, mm_fp4, backend = _get_flashinfer_ops()
        if fp4_quantize is None or mm_fp4 is None:
            raise RuntimeError("flashinfer FP4 ops not available")

        output_dtype = x.dtype
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        output_size = layer.output_size_per_partition
        output_shape = list(input_shape[:-1]) + [output_size]

        # Quantize activations to FP4 using flashinfer's native op
        x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)

        # Pad activations to match weight K-dimension padding
        weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
        x_fp4 = _pad_nvfp4_activation(x_fp4, weights_padding_cols)

        w = layer.weight
        w_scale_interleaved = layer.weight_scale_interleaved

        # Cast block scales to FP8 if needed
        if x_scale_interleaved.dtype == torch.uint8:
            x_scale_interleaved = x_scale_interleaved.view(torch.float8_e4m3fn)
        if w_scale_interleaved.dtype == torch.uint8:
            w_scale_interleaved = w_scale_interleaved.view(torch.float8_e4m3fn)

        # flashinfer.mm_fp4 expects TRANSPOSED weight and weight_scale
        # (SGLang convention, different from vLLM's flashinfer_scaled_fp4_mm)
        out = mm_fp4(
            x_fp4,
            w.T,
            x_scale_interleaved,
            w_scale_interleaved.T,
            layer.alpha,
            output_dtype,
            backend=backend,
        )

        out = _slice_output(out, output_size)

        if bias is not None:
            out = out + bias

        return out.view(*output_shape)
