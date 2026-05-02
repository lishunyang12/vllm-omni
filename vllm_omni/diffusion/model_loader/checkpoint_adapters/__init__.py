# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint adapter selection for ModelOpt-quantized diffusion checkpoints.

A checkpoint adapter wraps the raw safetensors weights iterator and rewrites
names / handles scale tensors / dequantizes-on-load as needed before the
model's ``load_weights`` sees the stream.  Adapters are stateless w.r.t. the
loader and carry their own per-call state.

Selection is by ``is_compatible(source, quant_config, use_safetensors)`` -
the first adapter that matches wins.  Currently FP8 takes precedence over
NVFP4 (mutually exclusive in practice since the quant_config name differs).
"""

from __future__ import annotations

from typing import Protocol

from torch import nn

from .modelopt_fp8 import ModelOptFp8CheckpointAdapter
from .modelopt_nvfp4 import ModelOptNvFp4CheckpointAdapter

__all__ = [
    "CheckpointAdapter",
    "ModelOptFp8CheckpointAdapter",
    "ModelOptNvFp4CheckpointAdapter",
    "get_checkpoint_adapter",
]


class CheckpointAdapter(Protocol):
    @classmethod
    def is_compatible(
        cls,
        source: object,
        quant_config: object | None,
        use_safetensors: bool,
    ) -> bool: ...

    def adapt(self, weights): ...


_REGISTRY: tuple[type, ...] = (
    ModelOptFp8CheckpointAdapter,
    ModelOptNvFp4CheckpointAdapter,
)


def get_checkpoint_adapter(
    source: object,
    quant_config: object | None,
    model: nn.Module,
    use_safetensors: bool = True,
) -> CheckpointAdapter | None:
    """Return the first compatible checkpoint adapter, or None.

    Order in ``_REGISTRY`` is the precedence; FP8 first since it has a
    stricter compatibility check (requires ``is_checkpoint_fp8_serialized``).
    """
    for adapter_cls in _REGISTRY:
        if adapter_cls.is_compatible(source, quant_config, use_safetensors):
            return adapter_cls(model, source)
    return None
