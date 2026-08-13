# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Streaming adapter for the official LTX-2.5 split checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from pathlib import Path

import torch
from torch import nn

_RAW_SOURCE_FILENAMES = {
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "ltx-2.5-audio-vae-bf16.safetensors",
    "ltx-2.5-video-vae-conv-bf16.safetensors",
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}


def _get_source_mapper(filename: str) -> Callable[[str], str | None]:
    # Import lazily: diffusers_loader imports checkpoint_adapters while the LTX
    # package imports DiffusersPipelineLoader. Importing the model package here
    # at module scope would therefore break spawned diffusion workers.
    from vllm_omni.diffusion.models.ltx2.ltx2_raw_checkpoint import (
        map_ltx2_audio_file_weight,
        map_ltx2_text_encoder_file_weight,
        map_ltx2_transformer_file_weight,
        map_ltx2_upsampler_weight,
        map_ltx2_video_vae_weight,
    )

    source_mappers = {
        "ltx-2.5-22b-distilled-transformer-bf16.safetensors": map_ltx2_transformer_file_weight,
        "ltx-2.5-22b-dev-transformer-bf16.safetensors": map_ltx2_transformer_file_weight,
        "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors": map_ltx2_text_encoder_file_weight,
        "ltx-2.5-audio-vae-bf16.safetensors": map_ltx2_audio_file_weight,
        "ltx-2.5-video-vae-conv-bf16.safetensors": map_ltx2_video_vae_weight,
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": map_ltx2_upsampler_weight,
    }
    return source_mappers[filename]


_EMBEDDED_GEMMA_ASSET_KEYS = {
    "tokenizer_json",
    "hf_asset__tokenizer_config.json",
    "hf_asset__processor_config.json",
    "hf_asset__chat_template.jinja",
    "hf_asset__generation_config.json",
}


def _source_filename(source: object) -> str | None:
    patterns = getattr(source, "allow_patterns_overrides", None)
    if not isinstance(patterns, list) or len(patterns) != 1:
        return None
    return Path(patterns[0]).name


class LTX2RawCheckpointAdapter:
    """Map each official raw tensor to its vLLM-Omni component name.

    Only embedded tokenizer assets are intentionally omitted. Every other
    unmapped tensor is an implementation error and fails before generation.
    """

    def __init__(self, model: nn.Module, source: object):
        filename = _source_filename(source)
        if filename not in _RAW_SOURCE_FILENAMES:
            raise ValueError(f"Unsupported LTX-2.5 raw checkpoint source: {filename!r}")
        self._filename = filename
        self._mapper = _get_source_mapper(filename)
        named_parameters = getattr(model, "named_parameters", None)
        named_buffers = getattr(model, "named_buffers", None)
        self._available_names = (
            {name for name, _ in named_parameters()} | {name for name, _ in named_buffers()}
            if callable(named_parameters) and callable(named_buffers)
            else set()
        )

    @classmethod
    def is_compatible(
        cls,
        model: nn.Module,
        source: object,
        quant_config: object | None,
        use_safetensors: bool,
    ) -> bool:
        del quant_config
        return (
            use_safetensors
            and bool(getattr(model, "_ltx2_raw_checkpoint", False))
            and not getattr(source, "prefix", "")
            and _source_filename(source) in _RAW_SOURCE_FILENAMES
        )

    def _resolve_gemma_target_name(self, mapped_name: str) -> str:
        if not self._available_names or mapped_name in self._available_names:
            return mapped_name
        candidates = [mapped_name]
        if ".model.vision_embedder." in mapped_name:
            candidates.append(mapped_name.replace(".model.vision_embedder.", ".model.embed_vision."))
        candidates.extend(
            name.replace(
                ".model.embed_vision.embedding_projection.",
                ".model.embed_vision.multimodal_embedder.embedding_projection.",
            )
            for name in tuple(candidates)
        )
        return next((name for name in candidates if name in self._available_names), mapped_name)

    def adapt(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        mapped_names: set[str] = set()
        for raw_name, tensor in weights:
            mapped_name = self._mapper(raw_name)
            if mapped_name is not None and self._filename == "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors":
                mapped_name = self._resolve_gemma_target_name(mapped_name)
            if mapped_name is None:
                if self._filename == "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" and (
                    raw_name in _EMBEDDED_GEMMA_ASSET_KEYS
                ):
                    continue
                raise ValueError(
                    f"Unmapped required tensor {raw_name!r} in official LTX-2.5 source {self._filename!r}."
                )
            if mapped_name in mapped_names:
                raise ValueError(
                    f"Official LTX-2.5 source {self._filename!r} maps multiple tensors to {mapped_name!r}."
                )
            mapped_names.add(mapped_name)
            yield mapped_name, tensor
