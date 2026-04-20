# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator, Iterable

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4LinearMethod
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.utils import get_packed_modules_mapping

logger = init_logger(__name__)


def _patch_nvfp4_apply_for_nd_input() -> None:
    """Patch ModelOptNvFp4LinearMethod.apply to handle N-D activation tensors.

    vLLM's NVFP4 path was designed for LLMs, which flatten activations to 2D
    (batch*seq, hidden) before every Linear. Diffusion transformers pass 3-D
    tensors (batch, seq, hidden) directly. Wrap `.apply` to flatten leading
    dims, call the original, and restore the original shape. Idempotent.
    """
    if getattr(ModelOptNvFp4LinearMethod, "_vllm_omni_nd_input_patched", False):
        return

    original_apply = ModelOptNvFp4LinearMethod.apply

    def _nd_aware_apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() <= 2:
            return original_apply(self, layer, x, bias)
        leading = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        out_2d = original_apply(self, layer, x_2d, bias)
        return out_2d.reshape(*leading, out_2d.shape[-1])

    ModelOptNvFp4LinearMethod.apply = _nd_aware_apply
    ModelOptNvFp4LinearMethod._vllm_omni_nd_input_patched = True


def _patch_nvfp4_process_weights_for_fused_qkv() -> None:
    """Patch process_weights_after_loading to correctly fuse per-shard scales.

    For a fused QKV Linear, ModelOpt calibration produces three different
    per-shard `weight_scale_2` scalars (one per Q/K/V) and three per-group
    `weight_scale` blocks. vLLM's original path does:

        weight_global_scale = layer.weight_scale_2.max()

    which KEEPS only the largest of the three scalars. The per-group
    `weight_scale` for K/V was calibrated relative to their own smaller
    `weight_scale_2`, so substituting Q's larger value at inference makes
    K's and V's effective weights too large (often by 1.5-3x). For LLMs
    where Q/K/V weight magnitudes are similar this is ~OK; for diffusion
    transformers the magnitudes are more dispersed and the output becomes
    noise.

    Fix: before `.max()` collapses the per-shard vector, re-normalize each
    shard's per-group `weight_scale` by `weight_scale_2[shard] / max`. That
    bakes the per-shard information into the per-group scales, so the
    subsequent `.max()` + matmul produces the correct effective per-group
    scale for every shard.
    """
    if getattr(ModelOptNvFp4LinearMethod, "_vllm_omni_fused_scale_patched", False):
        return

    original_process = ModelOptNvFp4LinearMethod.process_weights_after_loading

    def _fused_scale_aware_process(self, layer: nn.Module) -> None:
        logical_widths = getattr(layer, "logical_widths", None)
        weight_scale_2 = getattr(layer, "weight_scale_2", None)
        weight_scale = getattr(layer, "weight_scale", None)
        if (
            logical_widths is not None
            and len(logical_widths) > 1
            and weight_scale_2 is not None
            and weight_scale is not None
            and weight_scale_2.numel() == len(logical_widths)
        ):
            ws2_fp32 = weight_scale_2.detach().to(torch.float32)
            max_ws2 = ws2_fp32.max()
            # Avoid divide-by-zero on pathological calibration.
            if float(max_ws2) > 0:
                ratios = (ws2_fp32 / max_ws2).clamp(max=1.0)
                original_dtype = weight_scale.dtype
                ws_fp32 = weight_scale.detach().to(torch.float32)
                offset = 0
                for ratio, width in zip(ratios.tolist(), logical_widths):
                    if ratio < 1.0:
                        ws_fp32[offset : offset + width] = ws_fp32[offset : offset + width] * ratio
                    offset += width
                weight_scale.data.copy_(ws_fp32.to(original_dtype))
        return original_process(self, layer)

    ModelOptNvFp4LinearMethod.process_weights_after_loading = _fused_scale_aware_process
    ModelOptNvFp4LinearMethod._vllm_omni_fused_scale_patched = True
    logger.info_once(
        "Patched ModelOptNvFp4LinearMethod.process_weights_after_loading to "
        "absorb per-shard weight_scale_2 into per-group weight_scale before "
        "the `.max()` reduction. Required when Q/K/V (or gate/up) scales "
        "differ by more than a few percent — common in diffusion transformers."
    )


# Apply both patches at import time — covers all workers that load this module.
_patch_nvfp4_apply_for_nd_input()
_patch_nvfp4_process_weights_for_fused_qkv()

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
            # For both weights and scales we yield the ORIGINAL checkpoint name.
            # The outer model loader (e.g. HunyuanVideo15 load_weights) handles
            # packed-module name remapping itself via its stacked_params_mapping
            # + shard_id path. Pre-remapping here to e.g. `to_qkv.weight_scale`
            # makes the outer loader mis-split names on substring overlaps
            # (`.to_q` is a substring of `.to_qkv`) and silently drop scales.
            if self._is_scale(name):
                target_name = self._resolve_target_name(name)
                if target_name is None and skipped_scales < 3:
                    similar = [k for k in self._loadable_tensors if k.endswith(name.split(".")[-1])][:3]
                    logger.warning(
                        "ModelOpt NVFP4 adapter: scale %r has no matching model param. "
                        "Similar loadable params by suffix: %r. "
                        "Hint: checkpoint key doesn't align with any model parameter. "
                        "Check hf_to_vllm_mapper on the model class.",
                        name,
                        similar,
                    )
                if target_name is None:
                    skipped_scales += 1
                    continue
                passed_scales += 1
                yield name, tensor
                continue

            # Non-scale tensor: pass through with original name. Quantized
            # weights stay U8-packed for the NVFP4 kernel; ignored layers are
            # already stored as BF16, so nothing to unpack on this side.
            passed_weights += 1
            yield name, tensor

        if skipped_scales or passed_scales or passed_weights:
            logger.info_once(
                "Adapted ModelOpt NVFP4 %s weights: %d weights passed through, "
                "%d scale tensors passed through, %d scale tensors skipped (no target)",
                self._source_label,
                passed_weights,
                passed_scales,
                skipped_scales,
            )
