# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator, Iterable

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.utils import get_packed_modules_mapping

logger = init_logger(__name__)

MODEL_OPT_SCALE_SUFFIXES = (".input_scale", ".weight_scale", ".weight_scale_inv")
DEFAULT_PACKED_MODULES_MAPPING = {
    "to_qkv": ("to_q", "to_k", "to_v"),
    "add_kv_proj": ("add_q_proj", "add_k_proj", "add_v_proj"),
    "w13": ("w1", "w3"),
}
FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)


class ModelOptFp8CheckpointAdapter:
    def __init__(self, model: nn.Module, source: object):
        self._loadable_tensors = self._get_model_loadable_tensors(model)
        self._loadable_names = set(self._loadable_tensors)
        self._packed_modules_mapping = self._get_packed_modules_mapping(model)
        self._source_label = getattr(source, "prefix", "") or getattr(source, "subfolder", None) or "model"

    @classmethod
    def is_compatible(
        cls,
        source: object,
        quant_config: object | None,
        use_safetensors: bool,
    ) -> bool:
        return use_safetensors and cls._is_transformer_source(source) and cls._is_checkpoint_quant_config(quant_config)

    @staticmethod
    def _is_transformer_source(source: object) -> bool:
        if getattr(source, "subfolder", None) == "transformer":
            return True
        return str(getattr(source, "prefix", "")).startswith("transformer.")

    @staticmethod
    def _is_checkpoint_quant_config(quant_config: object | None) -> bool:
        return (
            quant_config is not None
            and hasattr(quant_config, "get_name")
            and quant_config.get_name() == "modelopt"
            and bool(getattr(quant_config, "is_checkpoint_fp8_serialized", False))
        )

    @staticmethod
    def _get_model_loadable_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
        loadable_tensors: dict[str, torch.Tensor] = {name: param for name, param in model.named_parameters()}
        loadable_tensors.update({name: buffer for name, buffer in model.named_buffers()})
        return loadable_tensors

    @staticmethod
    def _is_scale(name: str) -> bool:
        return name.endswith(MODEL_OPT_SCALE_SUFFIXES)

    @staticmethod
    def _is_fp8_tensor(tensor: torch.Tensor) -> bool:
        return tensor.dtype in FP8_DTYPES

    @staticmethod
    def _get_weight_scale_name(weight_name: str) -> str | None:
        if weight_name.endswith(".weight"):
            return weight_name[: -len(".weight")] + ".weight_scale"
        return None

    @staticmethod
    def _replace_module_name(name: str, old: str, new: str) -> str:
        if name.startswith(f"{old}."):
            return f"{new}.{name[len(old) + 1 :]}"
        return name.replace(f".{old}.", f".{new}.")

    @staticmethod
    def _get_packed_modules_mapping(model: nn.Module) -> dict[str, tuple[str, ...]]:
        mapping = {
            packed_name: tuple(shard_names) for packed_name, shard_names in DEFAULT_PACKED_MODULES_MAPPING.items()
        }
        mapping.update(
            {
                str(packed_name): tuple(str(shard_name) for shard_name in shard_names)
                for packed_name, shard_names in get_packed_modules_mapping(model).items()
            }
        )
        return mapping

    def _resolve_target_name(self, name: str) -> str | None:
        if name in self._loadable_names:
            return name

        if ".to_out.0." in name:
            candidate = name.replace(".to_out.0.", ".to_out.")
            if candidate in self._loadable_names:
                return candidate

        for packed_name, shard_names in self._packed_modules_mapping.items():
            for shard_name in shard_names:
                candidate = self._replace_module_name(name, shard_name, packed_name)
                if candidate != name and candidate in self._loadable_names:
                    return candidate
        return None

    @staticmethod
    def _reshape_weight_scale(scale: torch.Tensor, weight_shape: torch.Size) -> torch.Tensor:
        if scale.numel() == 1:
            return scale.reshape(())
        if len(weight_shape) == 2 and scale.ndim == 1 and scale.shape[0] == weight_shape[0]:
            return scale.reshape(-1, 1)
        if tuple(scale.shape) == tuple(weight_shape):
            return scale
        if (
            len(weight_shape) == 2
            and scale.ndim == 4
            and scale.shape[1] == 1
            and scale.shape[3] == 1
            and weight_shape[0] % scale.shape[0] == 0
            and weight_shape[1] % scale.shape[2] == 0
        ):
            block_n = weight_shape[0] // scale.shape[0]
            block_k = weight_shape[1] // scale.shape[2]
            return scale.expand(scale.shape[0], block_n, scale.shape[2], block_k).reshape(weight_shape)
        raise ValueError(f"Unsupported ModelOpt FP8 weight_scale shape {tuple(scale.shape)} for weight {weight_shape}")

    def _dequantize_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        scale_tensors: dict[str, torch.Tensor],
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        scale_name = self._get_weight_scale_name(name)
        if scale_name is None or scale_name not in scale_tensors:
            raise ValueError(f"Missing ModelOpt FP8 weight_scale for full-precision target weight {name!r}")

        weight = loaded_weight.to(dtype=torch.float32)
        scale = scale_tensors[scale_name].to(dtype=torch.float32, device=weight.device)
        scale = self._reshape_weight_scale(scale, loaded_weight.shape)
        return (weight * scale).to(dtype=target_dtype)

    def adapt(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        scale_tensors: dict[str, torch.Tensor] = {}
        pending_weights: dict[str, list[tuple[str, torch.Tensor, torch.dtype]]] = {}

        skipped_scales = 0
        dequantized_weights = 0
        for name, tensor in weights:
            target_name = self._resolve_target_name(name)
            if self._is_scale(name):
                scale_tensors[name] = tensor
                if target_name is None:
                    skipped_scales += 1
                else:
                    yield name, tensor

                for weight_name, weight_tensor, target_dtype in pending_weights.pop(name, []):
                    yield (
                        weight_name,
                        self._dequantize_weight(
                            weight_name,
                            weight_tensor,
                            scale_tensors,
                            target_dtype,
                        ),
                    )
                    dequantized_weights += 1
                continue

            if self._is_fp8_tensor(tensor) and target_name is not None:
                target_tensor = self._loadable_tensors[target_name]
                if target_tensor.dtype not in FP8_DTYPES:
                    scale_name = self._get_weight_scale_name(name)
                    if scale_name is None:
                        raise ValueError(f"Missing ModelOpt FP8 weight_scale name for weight {name!r}")
                    if scale_name in scale_tensors:
                        tensor = self._dequantize_weight(name, tensor, scale_tensors, target_tensor.dtype)
                        dequantized_weights += 1
                    else:
                        pending_weights.setdefault(scale_name, []).append((name, tensor, target_tensor.dtype))
                        continue
            yield name, tensor

        if pending_weights:
            missing_scale_names = ", ".join(repr(name) for name in sorted(pending_weights))
            raise ValueError(
                f"Missing ModelOpt FP8 weight_scale for full-precision target weights: {missing_scale_names}"
            )

        if skipped_scales or dequantized_weights:
            logger.info_once(
                "Adapted ModelOpt FP8 %s weights: dequantized %d full-precision weights, skipped %d scale tensors",
                self._source_label,
                dequantized_weights,
                skipped_scales,
            )
