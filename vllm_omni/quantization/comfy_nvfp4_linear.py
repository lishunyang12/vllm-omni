# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comfy-kitchen-native NVFP4 Linear for BFL FLUX.2.

BFL's FLUX.2-dev-NVFP4 release is (per the model card and
Comfy-Org/comfy-kitchen README) compatible only with comfy-kitchen-style
kernels. The scale layout and weight packing diverge from what vLLM's
ModelOpt FP4 / flashinfer mm_fp4 path expects, and SGLang's upstream PR
#20137 explicitly documents that only the ``ComfyUIFp4LinearMethod``
backed by ``comfy_kitchen`` produces correct output for this checkpoint.

This module is a direct port of
https://github.com/Comfy-Org/comfy-kitchen/blob/main/samples/nvfp4_linear.py
adapted to:

  - Match the return-tuple shape vLLM parallel linears expose
    (``(output, bias_or_None)`` when ``return_bias`` is True).
  - Be a drop-in replacement for ``ColumnParallelLinear`` /
    ``MergedColumnParallelLinear`` / ``QKVParallelLinear`` /
    ``RowParallelLinear`` at ``tensor_parallel_size=1``. Output
    partition sizes collapse to a single fused output dim, which
    lines up exactly with how BFL NVFP4 stores its fused weights
    (fused Q+K+V, fused gate+up, etc.).
  - Hook into vLLM's weight-loader protocol (each param exposes a
    ``weight_loader`` attribute) so the existing
    ``Flux2Transformer2DModel.load_weights`` path — which already
    applies the BFL → diffusers name mapping — works unchanged.

Requires ``comfy-kitchen[cublas]`` on Blackwell (SM ≥ 10.0):

    pip install comfy-kitchen[cublas]
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def _get_quantized_tensor():
    """Return (QuantizedTensor, LayoutName) or raise if comfy_kitchen missing."""
    try:
        from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout
    except ImportError as e:
        raise RuntimeError(
            "comfy-kitchen is not installed. For BFL FLUX.2-dev-NVFP4 "
            "inference, install with: pip install comfy-kitchen[cublas]"
        ) from e
    return QuantizedTensor, TensorCoreNVFP4Layout


def _copy_weight_loader(param: nn.Parameter, loaded: torch.Tensor) -> None:
    """Minimal weight_loader that just copies the loaded tensor in place.

    BFL NVFP4 ckps store every quantized linear as a single fused weight
    (no Q/K/V shards), so we never need vLLM's stacked-shard handling.
    """
    param.data.copy_(loaded)


class NVFP4Linear(nn.Module):
    """NVFP4-quantized Linear using comfy-kitchen's ``QuantizedTensor``.

    Parameters match the on-disk BFL NVFP4 convention:

      - ``weight``          : ``uint8``  [out, in // 2]    packed FP4
      - ``weight_scale``    : ``uint8``  [out, in // 16]   (viewed as FP8_E4M3)
      - ``weight_scale_2``  : ``float32`` scalar           per-tensor scale
      - ``input_scale``     : ``float32`` scalar           activation scale
      - ``bias`` (optional) : ``compute_dtype``

    At ``tensor_parallel_size=1`` this replaces any of vLLM's
    ColumnParallel / RowParallel / QKVParallel / MergedColumnParallel
    layers. For fused layers (QKV, gate+up) the BFL checkpoint already
    stores them pre-fused, so ``out_features`` is the fused size.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        return_bias: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
        group_size: int = 16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.return_bias = return_bias
        self.compute_dtype = compute_dtype
        self.group_size = group_size

        # Packed FP4 weight.
        weight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        weight.weight_loader = _copy_weight_loader
        self.weight = weight

        # Per-block weight scales; viewed as FP8_E4M3 when wrapped in QT.
        weight_scale = nn.Parameter(
            torch.empty(
                out_features, in_features // group_size, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        weight_scale.weight_loader = _copy_weight_loader
        self.weight_scale = weight_scale

        # Per-tensor global scales.
        weight_scale_2 = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        weight_scale_2.weight_loader = _copy_weight_loader
        self.weight_scale_2 = weight_scale_2

        input_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        input_scale.weight_loader = _copy_weight_loader
        self.input_scale = input_scale

        if bias:
            bias_param = nn.Parameter(
                torch.empty(out_features, dtype=compute_dtype), requires_grad=False
            )
            bias_param.weight_loader = _copy_weight_loader
            self.bias = bias_param
        else:
            self.register_parameter("bias", None)

        # Set after load (see ``wrap_weight_as_quantized_tensor``).
        self._weight_wrapped = False

    @torch.no_grad()
    def wrap_weight_as_quantized_tensor(self) -> None:
        """Replace the raw uint8 ``weight`` with a ``QuantizedTensor``.

        Called once after ``load_weights`` populates the raw tensors.
        Matches the post-load step in comfy-kitchen's sample
        ``NVFP4Linear._load_from_state_dict``.
        """
        if self._weight_wrapped:
            return
        QuantizedTensor, TensorCoreNVFP4Layout = _get_quantized_tensor()
        params = TensorCoreNVFP4Layout.Params(
            scale=self.weight_scale_2.data,
            orig_dtype=self.compute_dtype,
            orig_shape=(self.out_features, self.in_features),
            block_scale=self.weight_scale.data.view(torch.float8_e4m3fn),
        )
        self.weight = nn.Parameter(
            QuantizedTensor(
                self.weight.data, "TensorCoreNVFP4Layout", params
            ),
            requires_grad=False,
        )
        self._weight_wrapped = True

    def forward(self, x: torch.Tensor):
        if not self._weight_wrapped:
            self.wrap_weight_as_quantized_tensor()

        QuantizedTensor, _ = _get_quantized_tensor()

        input_shape = x.shape
        x_2d = x.reshape(-1, input_shape[-1])

        if not isinstance(x_2d, QuantizedTensor):
            x_qt = QuantizedTensor.from_float(
                x_2d, "TensorCoreNVFP4Layout", scale=self.input_scale
            )
        else:
            x_qt = x_2d

        out = F.linear(x_qt, self.weight, self.bias)
        out = out.reshape(*input_shape[:-1], self.out_features)

        if self.return_bias:
            return out, None
        return out


def wrap_all_nvfp4_weights(root: nn.Module) -> int:
    """Walk ``root`` and finalize every ``NVFP4Linear`` after weight load.

    Returns the number of layers wrapped. Safe to call multiple times.
    """
    count = 0
    for _, m in root.named_modules():
        if isinstance(m, NVFP4Linear) and not m._weight_wrapped:
            m.wrap_weight_as_quantized_tensor()
            count += 1
    return count
