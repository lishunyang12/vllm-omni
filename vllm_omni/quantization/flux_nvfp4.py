# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone NVFP4 LinearMethod for BFL FLUX.2 checkpoints.

Ported from sgl-project/sglang#20137 + #22064. The upstream vLLM
``ModelOptNvFp4LinearMethod`` is designed for LLM ModelOpt checkpoints and
makes assumptions about weight/scale layout that don't hold for BFL's
FLUX.2-NVFP4 family. Rather than patching upstream, this module implements
the full create → process → apply cycle independently, calling vLLM's
low-level CUTLASS/FlashInfer FP4 ops directly.

Key differences from upstream:
  1. **FP4 nibble swap** — BFL packs nibbles in opposite byte order.
  2. **Scale swizzle source** — BFL ships scales in cuBLAS-tiled layout;
     we must re-swizzle for the FlashInfer CUTLASS kernel.
  3. **No delegation to ``self.kernel``** — we handle weight reformatting
     ourselves and call ``apply_nvfp4_linear`` from upstream's utils.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import Parameter
from vllm._custom_ops import scaled_fp4_quant
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight_for_cutlass,
    select_nvfp4_linear_backend,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm

logger = init_logger(__name__)


class FluxNvFp4LinearMethod(QuantizeMethodBase):
    """Standalone NVFP4 linear method for BFL FLUX.2 checkpoints.

    This is NOT a subclass of upstream's ``ModelOptNvFp4LinearMethod`` — it
    implements the same interface (``create_weights``, ``process_weights_after_loading``,
    ``apply``) but controls the entire weight pipeline so there are no
    hidden interactions with upstream's kernel backend.
    """

    def __init__(self, quant_config: Any) -> None:
        self.quant_config = quant_config
        self.backend = select_nvfp4_linear_backend()

    # ------------------------------------------------------------------
    # create_weights — allocate parameter slots matching the on-disk layout
    # ------------------------------------------------------------------
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

        # Packed FP4 weight: 2 values per byte → input_dim // 2.
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

        # Per-tensor input activation scale (one per output partition).
        input_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_scale", input_scale)

        # Per-tensor weight global scale (one per output partition).
        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)

        # Per-block FP8 weight scale: one E4M3 value per group of 16 elements.
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

    # ------------------------------------------------------------------
    # process_weights_after_loading — full standalone processing
    # ------------------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # 1. Global scales → alpha + input_scale_inv.
        input_global_scale = layer.input_scale.max().to(torch.float32)
        weight_global_scale = layer.weight_scale_2.max().to(torch.float32)

        layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
        layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
        layer.alpha = Parameter(
            (input_global_scale * weight_global_scale).to(torch.float32),
            requires_grad=False,
        )
        layer.input_global_scale_inv = Parameter(
            (1.0 / input_global_scale).to(torch.float32),
            requires_grad=False,
        )

        # Clean up raw per-partition scales (upstream convention).
        del layer.input_scale
        del layer.weight_scale_2

        # 2. FP4 nibble swap — BFL packs high/low nibble in opposite order.
        #    Confirmed needed by sgl-project/sglang#20137.
        w = layer.weight.data
        w_swapped = ((w >> 4) | ((w & 0x0F) << 4)).to(torch.uint8).contiguous()

        # 3. Pad weight + swizzle block-scales for CUTLASS/FlashInfer kernel.
        #    We call the SAME vLLM utility functions that upstream uses — the
        #    only delta is our pre-swap of the weight bytes in step 2.
        padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(w_swapped)
        swizzled_weight_scale = swizzle_blockscale(layer.weight_scale.data)

        layer.weight = Parameter(padded_weight, requires_grad=False)
        layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols
        layer.output_size_per_partition = padded_weight.shape[0]

    # ------------------------------------------------------------------
    # apply — forward pass calling flashinfer directly (SGLang style)
    # ------------------------------------------------------------------
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_dtype = x.dtype
        output_size = layer.output_size_per_partition
        output_shape = [*x.shape[:-1], output_size]

        x = x.view(-1, x.shape[-1])

        # Quantize activations to FP4 with swizzled block scales.
        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend=self.backend.value,
        )

        # Pad activations to match weight K-dimension padding.
        weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
        x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

        w = layer.weight
        w_scale = layer.weight_scale

        # Cast block scales to the expected dtype if needed.
        if x_blockscale.dtype == torch.uint8:
            x_blockscale = x_blockscale.view(torch.float8_e4m3fn)
        if w_scale.dtype == torch.uint8:
            w_scale = w_scale.view(torch.float8_e4m3fn)

        # SGLang discovered that flashinfer's scaled_fp4_mm expects
        # transposed weight + weight_scale for the CUTLASS backend.
        # See sgl-project/sglang#20137.
        backend_name = self.backend.value
        if backend_name.startswith("flashinfer-"):
            backend_name = backend_name[len("flashinfer-") :]
        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            w.T,
            x_blockscale,
            w_scale.T,
            layer.alpha,
            output_dtype,
            backend=backend_name,
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias

        return out.view(*output_shape)
