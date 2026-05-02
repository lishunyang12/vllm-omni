# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt NVFP4 checkpoint adapter for diffusion transformers.

Streams ModelOpt NVFP4 W4A8 checkpoints into a diffusers-style transformer:
weights stay packed FP4 (uint8) for quantized layers, FP8 weight_scales,
FP32 weight_scale_2 and input_scale tensors are routed to the matching
parameter on the model, and BF16 weights for excluded layers (e.g.
proj_out, norms) pass through unchanged.

The adapter does NOT dequantize on the fly.  The model is expected to have
NVFP4 linear layers (from ``ModelOptNvFp4Config.get_quant_method``) for the
quantized positions and plain ``nn.Linear`` for the excluded ones; both
shapes match what the calibration script writes out.

If a checkpoint has a packed FP4 tensor that maps to a BF16 ``nn.Linear``
on the model, this adapter raises a clear error rather than silently
dropping the weight: that indicates a calibration/serving config drift
(an exclude-list mismatch) that would otherwise produce zero-init weights.
"""

from collections.abc import Generator, Iterable
from dataclasses import dataclass, field

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.utils import get_packed_modules_mapping

logger = init_logger(__name__)

# Scale suffixes written by ModelOpt's NVFP4 export.  Order matters: more
# specific suffixes (weight_scale_2) must come before their prefixes
# (weight_scale) when used with str.endswith() for routing logic.
NVFP4_SCALE_SUFFIXES = (
    ".input_scale",
    ".weight_scale_2",
    ".weight_scale",
)
DEFAULT_PACKED_MODULES_MAPPING = {
    "to_qkv": ("to_q", "to_k", "to_v"),
    "add_kv_proj": ("add_q_proj", "add_k_proj", "add_v_proj"),
    "w13": ("w1", "w3"),
}


@dataclass
class _AdaptState:
    skipped_scales: int = 0
    routed_scales: int = 0
    routed_weights: int = 0


class ModelOptNvFp4CheckpointAdapter:
    def __init__(self, model: nn.Module, source: object):
        self._loadable_tensors = self._get_model_loadable_tensors(model)
        self._weights_mapper = self._get_weights_mapper(model)
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
        subfolder = getattr(source, "subfolder", None)
        if subfolder in ("transformer", "transformer_2"):
            return True
        prefix = str(getattr(source, "prefix", ""))
        return prefix.startswith("transformer.") or prefix.startswith("transformer_2.")

    @staticmethod
    def _is_checkpoint_quant_config(quant_config: object | None) -> bool:
        if quant_config is None or not hasattr(quant_config, "get_name"):
            return False
        return quant_config.get_name() in ("modelopt_fp4", "nvfp4")

    @staticmethod
    def _get_model_loadable_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
        loadable_tensors: dict[str, torch.Tensor] = {name: param for name, param in model.named_parameters()}
        loadable_tensors.update({name: buffer for name, buffer in model.named_buffers()})
        return loadable_tensors

    @staticmethod
    def _is_scale(name: str) -> bool:
        return name.endswith(NVFP4_SCALE_SUFFIXES)

    @classmethod
    def _get_weights_mapper(cls, model: nn.Module) -> WeightsMapper:
        mapping = {
            packed_name: tuple(shard_names) for packed_name, shard_names in DEFAULT_PACKED_MODULES_MAPPING.items()
        }
        mapping.update(
            {
                str(packed_name): tuple(str(shard_name) for shard_name in shard_names)
                for packed_name, shard_names in get_packed_modules_mapping(model).items()
            }
        )

        orig_to_new_substr = {".to_out.0.": ".to_out."}
        orig_to_new_prefix: dict[str, str] = {}
        for packed_name, shard_names in mapping.items():
            for shard_name in shard_names:
                orig_to_new_substr[f".{shard_name}."] = f".{packed_name}."
                orig_to_new_prefix[f"{shard_name}."] = f"{packed_name}."

        # Walk submodules to pick up `hf_to_vllm_mapper` defined on the
        # transformer (the adapter may be invoked with the whole pipeline).
        collected: set[int] = set()
        for m in (model, *(sm for _, sm in model.named_modules())):
            mp = getattr(m, "hf_to_vllm_mapper", None)
            if mp is None or id(mp) in collected:
                continue
            collected.add(id(mp))
            orig_to_new_substr.update(getattr(mp, "orig_to_new_substr", None) or {})
            orig_to_new_prefix.update(getattr(mp, "orig_to_new_prefix", None) or {})

        return WeightsMapper(
            orig_to_new_substr=orig_to_new_substr,
            orig_to_new_prefix=orig_to_new_prefix,
        )

    def _resolve_target_name(self, name: str) -> str | None:
        if name in self._loadable_tensors:
            return name
        for candidate in self._weights_mapper.apply_list([name]):
            if candidate != name and candidate in self._loadable_tensors:
                return candidate
        return None

    def _check_weight_target_compat(self, name: str, tensor: torch.Tensor, target_name: str) -> None:
        """Detect packed FP4 -> BF16 nn.Linear mismatches.

        The calibration script writes packed FP4 (dtype uint8, last dim halved)
        only for layers actually quantized.  If the corresponding model layer
        is plain BF16 nn.Linear (because serving has it in its exclude_modules
        but calibration didn't), copying a uint8 tensor into a BF16 param
        would silently fail or produce garbage.  Catch this early.
        """
        target = self._loadable_tensors[target_name]
        if tensor.dtype != torch.uint8:
            return
        if target.dtype == torch.uint8:
            return
        raise ValueError(
            f"NVFP4 packed weight {name!r} (uint8) cannot load into BF16 target "
            f"{target_name!r} (dtype={target.dtype}). Calibration quantized this "
            f"layer but the runtime quant_config has it in `exclude_modules`. "
            f"Align the exclude lists or recalibrate."
        )

    def _log_adaptation_summary(self, state: _AdaptState) -> None:
        if not (state.skipped_scales or state.routed_scales or state.routed_weights):
            return
        logger.info_once(
            "Adapted ModelOpt NVFP4 %s weights: routed %d scales, %d weights; skipped %d scales (no target)",
            self._source_label,
            state.routed_scales,
            state.routed_weights,
            state.skipped_scales,
        )

    def adapt(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        state = _AdaptState()

        for name, tensor in weights:
            target_name = self._resolve_target_name(name)
            is_scale = self._is_scale(name)

            if target_name is None:
                if is_scale:
                    state.skipped_scales += 1
                    if state.skipped_scales <= 3:
                        suffix = name.split(".")[-1]
                        similar = [k for k in self._loadable_tensors if k.endswith(suffix)][:3]
                        logger.warning(
                            "ModelOpt NVFP4 adapter: skipping scale %r (no target). "
                            "Similar loadable params by suffix: %r. "
                            "Hint: the checkpoint key uses a name that doesn't match any model parameter.",
                            name,
                            similar,
                        )
                    continue
                # Non-scale tensor without a target - let the model's load_weights
                # decide; it may have its own weight loader that knows about it.
                yield name, tensor
                continue

            if is_scale:
                state.routed_scales += 1
                yield target_name, tensor
                continue

            self._check_weight_target_compat(name, tensor, target_name)
            state.routed_weights += 1
            # Emit under the resolved name so the model's load_weights sees the
            # vLLM-side parameter name when packing/remapping is in play.
            yield target_name, tensor

        self._log_adaptation_summary(state)
