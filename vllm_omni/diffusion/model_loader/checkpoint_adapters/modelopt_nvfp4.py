# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator, Iterable

from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.utils import get_packed_modules_mapping

logger = init_logger(__name__)

# NVFP4 serialized checkpoints expose these auxiliary tensors per quantized Linear:
#   .input_scale       -> F32 scalar (activation per-tensor scale)
#   .weight_scale      -> F8_E4M3 [out_dim, in_dim // group_size] (per-group weight scale)
#   .weight_scale_2    -> F32 scalar (per-tensor global weight scale)
# Weights themselves are U8 (two FP4 values packed per byte) along the input dim.
MODEL_OPT_SCALE_SUFFIXES = (
    ".input_scale",
    ".weight_scale",
    ".weight_scale_2",
)

DEFAULT_PACKED_MODULES_MAPPING = {
    "to_qkv": ("to_q", "to_k", "to_v"),
    "add_kv_proj": ("add_q_proj", "add_k_proj", "add_v_proj"),
    "w13": ("w1", "w3"),
}


class ModelOptNvFp4CheckpointAdapter:
    """Diffusers ↔ vllm-omni name remapping for ModelOpt NVFP4 checkpoints.

    Unlike the FP8 adapter, this one does not need a dequantization path: layers
    that we exclude from quantization during calibration are saved at full
    precision (BF16), not as packed FP4 that would need unpacking to land in a
    BF16 model parameter. The job here is pure name mapping so diffusers-style
    keys (e.g. `ffn.net.0.proj.weight_scale`) route to the matching vllm-omni
    packed-module keys (e.g. `ffn.net_0.proj.weight_scale`).

    Serving the resulting weights still requires Blackwell (sm_100+ / sm_120) —
    this adapter only handles load-time key translation.
    """

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
        if getattr(source, "subfolder", None) == "transformer":
            return True
        return str(getattr(source, "prefix", "")).startswith("transformer.")

    @staticmethod
    def _is_checkpoint_quant_config(quant_config: object | None) -> bool:
        if quant_config is None or not hasattr(quant_config, "get_name"):
            return False
        # vLLM's ModelOptNvFp4Config registers itself as "modelopt_fp4".
        if quant_config.get_name() != "modelopt_fp4":
            return False
        return bool(getattr(quant_config, "is_checkpoint_nvfp4_serialized", False))

    @staticmethod
    def _get_model_loadable_tensors(model: nn.Module) -> dict[str, "object"]:
        loadable_tensors: dict = {name: param for name, param in model.named_parameters()}
        loadable_tensors.update({name: buffer for name, buffer in model.named_buffers()})
        return loadable_tensors

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

        # Collect `hf_to_vllm_mapper` from `model` AND any submodule that defines one
        # (same as the FP8 adapter — the transformer submodule inside a Pipeline
        # is where model-specific mappers live, e.g. WanTransformer3DModel).
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

    @staticmethod
    def _is_scale(name: str) -> bool:
        return name.endswith(MODEL_OPT_SCALE_SUFFIXES)

    def adapt(
        self,
        weights: Iterable[tuple[str, "object"]],
    ) -> Generator[tuple[str, "object"], None, None]:
        skipped_scales = 0
        passed_scales = 0
        passed_weights = 0

        for name, tensor in weights:
            target_name = self._resolve_target_name(name)
            if self._is_scale(name):
                if target_name is None:
                    skipped_scales += 1
                    if skipped_scales <= 3:
                        similar = [k for k in self._loadable_tensors if k.endswith(name.split(".")[-1])][:3]
                        logger.warning(
                            "ModelOpt NVFP4 adapter: skipping scale %r (no target). "
                            "Similar loadable params by suffix: %r. "
                            "Hint: checkpoint key doesn't match any model parameter. "
                            "Check hf_to_vllm_mapper on the model class.",
                            name,
                            similar,
                        )
                    continue
                passed_scales += 1
                yield target_name, tensor
                continue

            # Non-scale tensor: just pass through (name mapping handled by the
            # outer loader which also calls the weights_mapper). We don't need
            # to unpack U8 FP4 here — ignored (full-precision) layers are stored
            # as BF16, and quantized weights stay packed for the NVFP4 kernel.
            passed_weights += 1
            yield name, tensor

        if skipped_scales or passed_scales or passed_weights:
            logger.info_once(
                "Adapted ModelOpt NVFP4 %s weights: %d weights passed through, "
                "%d scale tensors remapped, %d scale tensors skipped (no target)",
                self._source_label,
                passed_weights,
                passed_scales,
                skipped_scales,
            )
