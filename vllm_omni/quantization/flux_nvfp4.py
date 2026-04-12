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
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    apply_nvfp4_linear,
    pad_nvfp4_weight_for_cutlass,
    select_nvfp4_linear_backend,
    swizzle_blockscale,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)

logger = init_logger(__name__)


class FluxNvFp4LinearMethod:
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
    # apply — forward pass using upstream's utility
    # ------------------------------------------------------------------
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_nvfp4_linear(
            backend=self.backend,
            layer=layer,
            x=x,
            bias=bias,
        )
