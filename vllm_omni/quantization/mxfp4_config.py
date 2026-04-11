# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 (W4A16) quantization for diffusion transformers.

Uses Marlin kernel on SM 75+, pure-PyTorch emulation fallback otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

try:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        is_fp4_marlin_supported,
        prepare_fp4_layer_for_marlin,
    )

    _HAS_MARLIN_FP4 = True
except ImportError:
    _HAS_MARLIN_FP4 = False

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
_FP4_MAX = 6.0


def quantize_weight_mxfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize float weight to packed MXFP4 uint8 + uint8 block scales.

    Returns:
        packed: uint8 (N, K//2).
        scales: uint8 (N, K//32).
    """
    block_size = OCP_MX_BLOCK_SIZE
    N, K = weight.shape
    assert K % block_size == 0, f"in_features ({K}) must be divisible by MX block size ({block_size})"
    num_blocks = K // block_size
    w_blocked = weight.float().reshape(N, num_blocks, block_size)

    block_amax = w_blocked.abs().amax(dim=-1)
    raw_scale = (block_amax / _FP4_MAX).clamp(min=2**-127)
    exponent = torch.floor(torch.log2(raw_scale)) + 127.0
    exponent = exponent.clamp(0, 254)
    scales = exponent.to(torch.uint8)
    descale = torch.exp2(exponent - 127.0).unsqueeze(-1)

    w_scaled = w_blocked / descale
    fp4_lut = _FP4_VALUES.to(weight.device)
    signs = w_scaled.sign()
    diffs = (w_scaled.abs().unsqueeze(-1) - fp4_lut).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)

    sign_bits = (signs < 0).to(torch.uint8) << 3
    nibbles = (sign_bits | indices).reshape(N, K)
    packed = (nibbles[:, 1::2] << 4) | nibbles[:, 0::2]
    return packed, scales


def dequant_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize packed MXFP4 weights. Accepts uint8 or float8_e8m0fnu scales."""
    block_size = OCP_MX_BLOCK_SIZE
    N, K_half = packed.shape
    K = K_half * 2

    low = (packed & 0x0F).to(torch.int32)
    high = ((packed >> 4) & 0x0F).to(torch.int32)
    nibbles = torch.stack([low, high], dim=-1).reshape(N, K)

    sign_bits = nibbles >> 3
    mag_idx = nibbles & 0x07

    fp4_lut = _FP4_VALUES.to(packed.device)
    magnitudes = fp4_lut[mag_idx.long()]
    signs = 1.0 - 2.0 * sign_bits.float()
    values = signs * magnitudes

    num_blocks = K // block_size
    values = values.reshape(N, num_blocks, block_size)
    if scales.dtype != torch.uint8:
        scales = scales.view(torch.uint8)
    descale = torch.exp2(scales.float() - 127.0).unsqueeze(-1)
    values = values * descale
    return values.reshape(N, K).to(out_dtype)


def qdq_mxfp4(x: torch.Tensor) -> torch.Tensor:
    """Quantize-then-dequantize activation to simulate MXFP4 noise."""
    block_size = OCP_MX_BLOCK_SIZE
    orig_shape = x.shape
    K = orig_shape[-1]

    pad = (block_size - K % block_size) % block_size
    if pad:
        x = F.pad(x, (0, pad))

    flat = x.float().reshape(-1, x.shape[-1])
    M, K_padded = flat.shape
    num_blocks = K_padded // block_size

    blocked = flat.reshape(M, num_blocks, block_size)
    block_amax = blocked.abs().amax(dim=-1)

    raw_scale = (block_amax / _FP4_MAX).clamp(min=2**-127)
    exponent = torch.floor(torch.log2(raw_scale)) + 127.0
    exponent = exponent.clamp(0, 254)
    descale = torch.exp2(exponent - 127.0).unsqueeze(-1)

    scaled = blocked / descale
    fp4_lut = _FP4_VALUES.to(x.device)
    diffs = (scaled.abs().unsqueeze(-1) - fp4_lut).abs()
    nearest = fp4_lut[diffs.argmin(dim=-1)]

    dequantized = (scaled.sign() * nearest * descale).reshape(M, K_padded)
    if pad:
        dequantized = dequantized[..., :K]

    return dequantized.reshape(orig_shape).to(x.dtype)


def _use_marlin() -> bool:
    return _HAS_MARLIN_FP4 and is_fp4_marlin_supported()


class Mxfp4OnlineLinearMethod(LinearMethodBase):
    """MXFP4 online quantization: loads BF16/FP16 weights, quantizes post-load.

    Uses Marlin W4A16 kernel on SM 75+, emulation fallback otherwise.
    """

    def __init__(self, quant_config: DiffusionMxfp4Config):
        self.quant_config = quant_config
        self.use_marlin = _use_marlin()

    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.params_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: Module) -> None:
        packed, scales = quantize_weight_mxfp4(layer.weight.data)
        layer.weight = torch.nn.Parameter(packed, requires_grad=False)

        if self.use_marlin:
            if hasattr(torch, "float8_e8m0fnu"):
                scales = scales.view(torch.float8_e8m0fnu)
            layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
            prepare_fp4_layer_for_marlin(layer)
        else:
            layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)

    def apply(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_marlin:
            return apply_fp4_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                weight_global_scale=None,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        dq_w = dequant_mxfp4(layer.weight, layer.weight_scale, x.dtype)
        qdq_x = qdq_mxfp4(x)
        return F.linear(qdq_x, dq_w, bias)


class DiffusionMxfp4Config(QuantizationConfig):
    """MXFP4 quantization config for diffusion transformers.

    Online quantization from BF16/FP16 checkpoints using OCP Microscaling FP4
    (E2M1 values, E8M0 shared exponents, block size 32).
    """

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        if activation_scheme != "dynamic":
            raise ValueError(f"MXFP4 only supports dynamic activation scheme, got {activation_scheme!r}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionMxfp4Config:
        activation_scheme = cls.get_from_keys_or(
            config,
            ["activation_scheme"],
            "dynamic",
        )
        ignored_layers = cls.get_from_keys_or(
            config,
            ["ignored_layers"],
            None,
        )
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config,
                ["modules_to_not_convert"],
                None,
            )
        return cls(
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(
                self.ignored_layers,
            )

    def get_quant_method(
        self,
        layer: Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return Mxfp4OnlineLinearMethod(self)
        return None
