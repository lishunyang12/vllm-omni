# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from torch import nn

from .modelopt_fp8 import ModelOptFp8CheckpointAdapter
from .modelopt_nvfp4 import ModelOptNvFp4CheckpointAdapter

Adapter = ModelOptFp8CheckpointAdapter | ModelOptNvFp4CheckpointAdapter


def get_checkpoint_adapter(
    model: nn.Module,
    source: object,
    quant_config: object | None,
    use_safetensors: bool,
) -> Adapter | None:
    if ModelOptNvFp4CheckpointAdapter.is_compatible(source, quant_config, use_safetensors):
        return ModelOptNvFp4CheckpointAdapter(model, source)
    if ModelOptFp8CheckpointAdapter.is_compatible(source, quant_config, use_safetensors):
        return ModelOptFp8CheckpointAdapter(model, source)
    return None


__all__ = [
    "ModelOptFp8CheckpointAdapter",
    "ModelOptNvFp4CheckpointAdapter",
    "get_checkpoint_adapter",
]
