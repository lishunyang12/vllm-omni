# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utilities for the official LTX-2.5 split-checkpoint layout.

The official repository stores model components in a handful of large
``safetensors`` files instead of a Diffusers directory.  This module contains
only layout inspection, metadata/config conversion, embedded tokenizer asset
handling, and pure weight-name mappings.  Loading the mapped tensors into a
pipeline is intentionally left to the model loader.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

LTX2CheckpointVariant = Literal["distilled", "full", "unknown"]
LTX2VideoDecoderType = Literal["diffusion", "conv"]

_TRANSFORMER_FILENAMES = {
    "distilled": "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "full": "ltx-2.5-22b-dev-transformer-bf16.safetensors",
}
_VIDEO_VAE_FILENAMES = {
    "diffusion": "ltx-2.5-video-vae-bf16.safetensors",
    "conv": "ltx-2.5-video-vae-conv-bf16.safetensors",
}
_TEXT_ENCODER_FILENAME = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
_AUDIO_VAE_FILENAME = "ltx-2.5-audio-vae-bf16.safetensors"
_LATENT_UPSAMPLER_FILENAME = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

_GEMMA_ASSET_KEYS = {
    "tokenizer_json": "tokenizer.json",
    "hf_asset__tokenizer_config.json": "tokenizer_config.json",
    "hf_asset__processor_config.json": "processor_config.json",
    "hf_asset__chat_template.jinja": "chat_template.jinja",
    "hf_asset__generation_config.json": "generation_config.json",
}


@dataclass(frozen=True)
class LTX2RawCheckpointLayout:
    """Resolved files in an official LTX-2.5 split checkpoint."""

    root: Path
    transformer: Path
    checkpoint_variant: Literal["distilled", "full"]
    video_decoder_type: LTX2VideoDecoderType
    text_encoder: Path
    audio_vae: Path
    video_vae: Path
    latent_upsampler: Path | None = None


@dataclass(frozen=True)
class LTX2RawCheckpointMetadata:
    """Small JSON metadata read from the split checkpoint headers."""

    model_version: str | None
    checkpoint_variant: LTX2CheckpointVariant
    transformer: dict[str, Any]
    connectors: dict[str, Any]
    scheduler: dict[str, Any]
    gemma: dict[str, Any]
    audio_vae: dict[str, Any]
    vocoder: dict[str, Any]
    video_vae: dict[str, Any]
    latent_upsampler: dict[str, Any] | None


def materialize_ltx2_raw_checkpoint(
    model_path: str | PathLike[str],
    *,
    checkpoint_variant: Literal["distilled", "full"],
    video_decoder_type: LTX2VideoDecoderType = "conv",
    require_latent_upsampler: bool = False,
    revision: str | None = None,
    cache_dir: str | PathLike[str] | None = None,
) -> LTX2RawCheckpointLayout:
    """Resolve a local snapshot, downloading only the selected raw files.

    The official repository contains multiple mutually exclusive transformer
    and VAE variants. Exact allow patterns keep selection deterministic and
    ensure ``revision`` pins both component metadata and tensor payloads.
    """

    if Path(model_path).is_dir():
        return resolve_ltx2_raw_checkpoint_layout(
            model_path,
            checkpoint_variant=checkpoint_variant,
            video_decoder_type=video_decoder_type,
            require_latent_upsampler=require_latent_upsampler,
        )

    selected = [
        f"diffusion_models/{_TRANSFORMER_FILENAMES[checkpoint_variant]}",
        f"text_encoders/{_TEXT_ENCODER_FILENAME}",
        f"vae/{_AUDIO_VAE_FILENAME}",
        f"vae/{_VIDEO_VAE_FILENAMES[video_decoder_type]}",
    ]
    if require_latent_upsampler:
        selected.append(f"latent_upscale_models/{_LATENT_UPSAMPLER_FILENAME}")
    snapshot = snapshot_download(
        repo_id=str(model_path),
        revision=revision,
        cache_dir=None if cache_dir is None else str(cache_dir),
        allow_patterns=selected,
    )
    return resolve_ltx2_raw_checkpoint_layout(
        snapshot,
        checkpoint_variant=checkpoint_variant,
        video_decoder_type=video_decoder_type,
        require_latent_upsampler=require_latent_upsampler,
    )


def _require_file(path: Path, component: str) -> Path:
    if not path.is_file():
        raise ValueError(f"Missing LTX-2.5 {component} checkpoint: {path}")
    return path


def resolve_ltx2_raw_checkpoint_layout(
    model_path: str | PathLike[str],
    *,
    checkpoint_variant: Literal["distilled", "full"],
    video_decoder_type: LTX2VideoDecoderType,
    require_latent_upsampler: bool = False,
    transformer_filename: str | None = None,
    video_vae_filename: str | None = None,
) -> LTX2RawCheckpointLayout:
    """Resolve a local snapshot of ``Lightricks/LTX-2.5``.

    Selection is explicit because an official snapshot contains both dev/full
    and distilled transformers and both DiffVAE and ConvVAE checkpoints.  The
    caller's task/profile flags are authoritative; filenames never select a
    variant implicitly.  The optional filename overrides are intended for a
    pinned compatible checkpoint, not wildcard discovery.
    """

    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(f"LTX-2.5 raw checkpoint path is not a directory: {root}")

    transformer = root / "diffusion_models" / (transformer_filename or _TRANSFORMER_FILENAMES[checkpoint_variant])
    video_vae = root / "vae" / (video_vae_filename or _VIDEO_VAE_FILENAMES[video_decoder_type])
    latent_upsampler = root / "latent_upscale_models" / _LATENT_UPSAMPLER_FILENAME
    if not latent_upsampler.is_file():
        if require_latent_upsampler:
            raise ValueError(f"LTX-2.5 two-stage execution requires checkpoint: {latent_upsampler}")
        latent_upsampler = None

    return LTX2RawCheckpointLayout(
        root=root,
        transformer=_require_file(transformer, f"{checkpoint_variant} transformer"),
        checkpoint_variant=checkpoint_variant,
        video_decoder_type=video_decoder_type,
        text_encoder=_require_file(root / "text_encoders" / _TEXT_ENCODER_FILENAME, "text encoder"),
        audio_vae=_require_file(root / "vae" / _AUDIO_VAE_FILENAME, "audio VAE/vocoder"),
        video_vae=_require_file(video_vae, f"{video_decoder_type} video VAE"),
        latent_upsampler=latent_upsampler,
    )


def is_ltx2_raw_checkpoint_layout(
    model_path: str | PathLike[str],
    *,
    checkpoint_variant: Literal["distilled", "full"],
    video_decoder_type: LTX2VideoDecoderType,
) -> bool:
    """Return whether ``model_path`` unambiguously has the official layout."""

    try:
        resolve_ltx2_raw_checkpoint_layout(
            model_path,
            checkpoint_variant=checkpoint_variant,
            video_decoder_type=video_decoder_type,
        )
    except ValueError:
        return False
    return True


def read_safetensors_metadata(path: str | PathLike[str]) -> dict[str, str]:
    """Read only the safetensors header metadata, not tensor payloads."""

    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        return dict(checkpoint.metadata() or {})


def parse_json_metadata(metadata: Mapping[str, str], key: str, *, source: str | Path = "checkpoint") -> dict[str, Any]:
    """Parse a required JSON object from safetensors metadata."""

    if key not in metadata:
        raise ValueError(f"Missing {key!r} metadata in {source}.")
    try:
        value = json.loads(metadata[key])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {key!r} metadata from {source}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected {key!r} metadata in {source} to contain a JSON object.")
    return value


def infer_ltx2_checkpoint_variant(
    metadata: Mapping[str, str],
    *,
    filename: str | Path | None = None,
) -> LTX2CheckpointVariant:
    """Infer distilled/full weights without treating the schedule as proof.

    Current official transformer metadata identifies the model version but has
    no explicit distilled/full field.  Explicit metadata is preferred if a
    future checkpoint adds it; current checkpoints therefore use their
    published filename.  The ``LinearQuadratic`` sampler is deliberately not
    used because it describes a schedule, not the trained weight variant.
    """

    candidates: list[str] = []
    for key in ("checkpoint_variant", "variant", "weight_variant", "task_type"):
        value = metadata.get(key)
        if value:
            candidates.append(value)
    config_value = metadata.get("config")
    if config_value:
        try:
            config = json.loads(config_value)
        except json.JSONDecodeError:
            config = None
        if isinstance(config, dict):
            for key in ("checkpoint_variant", "variant", "weight_variant", "task_type"):
                value = config.get(key)
                if isinstance(value, str):
                    candidates.append(value)
    if filename is not None:
        candidates.append(Path(filename).name)

    for value in candidates:
        tokens = {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}
        if "distilled" in tokens or "distill" in tokens:
            return "distilled"
        if tokens.intersection({"full", "dev", "sft"}):
            return "full"
    return "unknown"


def inspect_ltx2_raw_checkpoint(layout: LTX2RawCheckpointLayout) -> LTX2RawCheckpointMetadata:
    """Read and normalize component configuration from checkpoint headers."""

    transformer_metadata = read_safetensors_metadata(layout.transformer)
    transformer_config = parse_json_metadata(transformer_metadata, "config", source=layout.transformer)
    text_metadata = read_safetensors_metadata(layout.text_encoder)
    audio_metadata = read_safetensors_metadata(layout.audio_vae)
    audio_config = parse_json_metadata(audio_metadata, "config", source=layout.audio_vae)
    video_metadata = read_safetensors_metadata(layout.video_vae)
    video_config = parse_json_metadata(video_metadata, "config", source=layout.video_vae)
    upsampler_config = None
    if layout.latent_upsampler is not None:
        upsampler_metadata = read_safetensors_metadata(layout.latent_upsampler)
        upsampler_config = convert_ltx2_upsampler_config(
            parse_json_metadata(upsampler_metadata, "config", source=layout.latent_upsampler)
        )

    model_versions = {
        value
        for value in (
            transformer_metadata.get("model_version"),
            audio_metadata.get("model_version"),
            video_metadata.get("model_version"),
        )
        if value is not None
    }
    if len(model_versions) > 1:
        raise ValueError(f"LTX-2.5 split components disagree on model_version: {sorted(model_versions)}")

    detected_variant = infer_ltx2_checkpoint_variant(transformer_metadata, filename=layout.transformer)
    if detected_variant not in ("unknown", layout.checkpoint_variant):
        raise ValueError(
            f"Selected {layout.checkpoint_variant!r} LTX-2.5 weights, but {layout.transformer.name!r} "
            f"identifies itself as {detected_variant!r}."
        )
    raw_transformer = _required_mapping(transformer_config, "transformer")
    raw_audio_vae = _required_mapping(audio_config, "audio_vae")
    raw_vocoder = _required_mapping(audio_config, "vocoder")
    raw_video_vae = _required_mapping(video_config, "vae")
    gemma = parse_json_metadata(text_metadata, "gemma_config", source=layout.text_encoder)
    normalized_video_vae = (
        convert_ltx2_diffusion_video_vae_config(raw_video_vae)
        if layout.video_decoder_type == "diffusion"
        else convert_ltx2_video_vae_config(raw_video_vae)
    )
    return LTX2RawCheckpointMetadata(
        model_version=next(iter(model_versions), None),
        checkpoint_variant=layout.checkpoint_variant,
        transformer=convert_ltx2_transformer_config(raw_transformer),
        connectors=convert_ltx2_connectors_config(raw_transformer, gemma),
        scheduler=dict(_required_mapping(transformer_config, "scheduler")),
        gemma=gemma,
        audio_vae=convert_ltx2_audio_vae_config(raw_audio_vae),
        vocoder=convert_ltx2_vocoder_config(raw_vocoder),
        video_vae=normalized_video_vae,
        latent_upsampler=upsampler_config,
    )


def convert_ltx2_scheduler_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert official metadata to the Diffusers scheduler state.

    Official denoise recipes provide their sigma sequences explicitly. These
    values match the converted LTX-2.5 scheduler for all remaining state.
    """

    return {
        "num_train_timesteps": config.get("num_train_timesteps", 1000),
        "shift": 1.0,
        "use_dynamic_shifting": False,
        "base_shift": 0.95,
        "max_shift": 2.05,
        "base_image_seq_len": 1024,
        "max_image_seq_len": 4096,
        "invert_sigmas": False,
        "shift_terminal": None,
        "use_karras_sigmas": False,
        "use_exponential_sigmas": False,
        "use_beta_sigmas": False,
        "time_shift_type": "exponential",
        "stochastic_sampling": False,
    }


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"Expected raw LTX-2.5 config to contain a {key!r} object.")
    return result


def convert_ltx2_transformer_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert official ``AVTransformer3DModel`` metadata to Diffusers config."""

    positions = tuple(config.get("positional_embedding_max_pos", (20, 2048, 2048)))
    audio_positions = tuple(config.get("audio_positional_embedding_max_pos", (20,)))
    qk_norm = config.get("qk_norm", "rms_norm")
    if qk_norm == "rms_norm":
        qk_norm = "rms_norm_across_heads"
    return {
        "in_channels": config.get("in_channels", 128),
        "out_channels": config.get("out_channels", 128),
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": config.get("num_attention_heads", 32),
        "attention_head_dim": config.get("attention_head_dim", 128),
        "cross_attention_dim": config.get("cross_attention_dim", 4096),
        "vae_scale_factors": (8, 32, 32),
        "pos_embed_max_pos": positions[0],
        "base_height": positions[1],
        "base_width": positions[2],
        "gated_attn": bool(config.get("apply_gated_attention", True)),
        "cross_attn_mod": bool(config.get("cross_attention_adaln", True)),
        "audio_in_channels": config.get("audio_out_channels", 128),
        "audio_out_channels": config.get("audio_out_channels", 128),
        "audio_patch_size": 1,
        "audio_patch_size_t": 1,
        "audio_num_attention_heads": config.get("audio_num_attention_heads", 32),
        "audio_attention_head_dim": config.get("audio_attention_head_dim", 64),
        "audio_cross_attention_dim": config.get("audio_cross_attention_dim", 2048),
        "audio_scale_factor": 4,
        "audio_pos_embed_max_pos": audio_positions[0],
        "audio_sampling_rate": 16000,
        "audio_hop_length": 160,
        "audio_gated_attn": bool(config.get("apply_gated_attention", True)),
        "audio_cross_attn_mod": bool(config.get("cross_attention_adaln", True)),
        "num_layers": config.get("num_layers", 48),
        "activation_fn": config.get("activation_fn", "gelu-approximate"),
        "qk_norm": qk_norm,
        "norm_elementwise_affine": bool(config.get("norm_elementwise_affine", False)),
        "norm_eps": config.get("norm_eps", 1e-6),
        "caption_channels": config.get("caption_channels", 3840),
        "attention_bias": bool(config.get("attention_bias", True)),
        "attention_out_bias": True,
        "rope_theta": config.get("positional_embedding_theta", 10000.0),
        "rope_double_precision": config.get("frequencies_precision") == "float64",
        "causal_offset": 1,
        "timestep_scale_multiplier": config.get("timestep_scale_multiplier", 1000),
        "cross_attn_timestep_scale_multiplier": config.get("av_ca_timestep_scale_multiplier", 1000),
        "rope_type": config.get("rope_type", "split"),
        "use_prompt_embeddings": False,
        "perturbed_attn": True,
        "ff_bias": bool(config.get("ff_bias", False)),
        "audio_ff_bias": True,
        "use_prompt_adaln_single": True,
        "use_keyframes_abs_pos_embedding": bool(config.get("use_keyframes_abs_pos_embedding", False)),
    }


def convert_ltx2_connectors_config(
    transformer_config: Mapping[str, Any],
    gemma_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build connector config from transformer and packed Gemma metadata."""

    text_config = _required_mapping(gemma_config, "text_config")
    return {
        "caption_channels": text_config["hidden_size"],
        "text_proj_in_factor": text_config["num_hidden_layers"] + 1,
        "video_connector_num_attention_heads": transformer_config["connector_num_attention_heads"],
        "video_connector_attention_head_dim": transformer_config["connector_attention_head_dim"],
        "video_connector_num_layers": transformer_config["connector_num_layers"],
        "video_connector_num_learnable_registers": transformer_config["connector_num_learnable_registers"],
        "video_gated_attn": bool(transformer_config.get("connector_apply_gated_attention", True)),
        "audio_connector_num_attention_heads": transformer_config["audio_connector_num_attention_heads"],
        "audio_connector_attention_head_dim": transformer_config["audio_connector_attention_head_dim"],
        "audio_connector_num_layers": transformer_config["connector_num_layers"],
        "audio_connector_num_learnable_registers": transformer_config["connector_num_learnable_registers"],
        "audio_gated_attn": bool(transformer_config.get("connector_apply_gated_attention", True)),
        "connector_rope_base_seq_len": transformer_config["connector_positional_embedding_max_pos"][0],
        "rope_theta": transformer_config.get("positional_embedding_theta", 10000.0),
        "rope_double_precision": transformer_config.get("frequencies_precision") == "float64",
        # Diffusers connectors receive token positions directly; this differs
        # from the official wrapper's internal temporal-positioning flag.
        "causal_temporal_positioning": False,
        "rope_type": transformer_config.get("rope_type", "split"),
        "per_modality_projections": True,
        "video_hidden_dim": transformer_config["cross_attention_dim"],
        "audio_hidden_dim": transformer_config["audio_cross_attention_dim"],
        "proj_bias": True,
    }


def convert_ltx2_audio_vae_config(config: Mapping[str, Any]) -> dict[str, Any]:
    params = _required_mapping(_required_mapping(config, "model"), "params")
    ddconfig = _required_mapping(params, "ddconfig")
    preprocessing = _required_mapping(config, "preprocessing")
    stft = _required_mapping(preprocessing, "stft")
    return {
        "base_channels": ddconfig["ch"],
        "output_channels": ddconfig["out_ch"],
        "ch_mult": tuple(ddconfig["ch_mult"]),
        "num_res_blocks": ddconfig["num_res_blocks"],
        "attn_resolutions": tuple(ddconfig["attn_resolutions"]) or None,
        "in_channels": ddconfig["in_channels"],
        "resolution": ddconfig["resolution"],
        "latent_channels": ddconfig["z_channels"],
        "norm_type": ddconfig["norm_type"],
        "causality_axis": ddconfig["causality_axis"],
        "dropout": ddconfig["dropout"],
        "mid_block_add_attention": bool(ddconfig["mid_block_add_attention"]),
        "sample_rate": params["sampling_rate"],
        "mel_hop_length": stft["hop_length"],
        "is_causal": bool(stft["causal"]),
        "mel_bins": ddconfig["mel_bins"],
        "double_z": bool(ddconfig["double_z"]),
    }


def convert_ltx2_vocoder_config(config: Mapping[str, Any]) -> dict[str, Any]:
    vocoder = _required_mapping(config, "vocoder")
    bwe = _required_mapping(config, "bwe")
    stereo_channels = 2 if vocoder.get("stereo", True) else 1
    bwe_channels = 2 if bwe.get("stereo", True) else 1
    return {
        "in_channels": 128,
        "hidden_channels": vocoder["upsample_initial_channel"],
        "out_channels": stereo_channels,
        "upsample_kernel_sizes": list(vocoder["upsample_kernel_sizes"]),
        "upsample_factors": list(vocoder["upsample_rates"]),
        "resnet_kernel_sizes": list(vocoder["resblock_kernel_sizes"]),
        "resnet_dilations": list(vocoder["resblock_dilation_sizes"]),
        "act_fn": vocoder["activation"],
        "leaky_relu_negative_slope": 0.1,
        "antialias": True,
        "antialias_ratio": 2,
        "antialias_kernel_size": 12,
        "final_act_fn": "tanh" if vocoder.get("use_tanh_at_final", False) else None,
        "final_bias": bool(vocoder.get("use_bias_at_final", False)),
        "bwe_in_channels": 128,
        "bwe_hidden_channels": bwe["upsample_initial_channel"],
        "bwe_out_channels": bwe_channels,
        "bwe_upsample_kernel_sizes": list(bwe["upsample_kernel_sizes"]),
        "bwe_upsample_factors": list(bwe["upsample_rates"]),
        "bwe_resnet_kernel_sizes": list(bwe["resblock_kernel_sizes"]),
        "bwe_resnet_dilations": list(bwe["resblock_dilation_sizes"]),
        "bwe_act_fn": bwe["activation"],
        "bwe_leaky_relu_negative_slope": 0.1,
        "bwe_antialias": True,
        "bwe_antialias_ratio": 2,
        "bwe_antialias_kernel_size": 12,
        "bwe_final_act_fn": "tanh" if bwe.get("use_tanh_at_final", False) else None,
        "bwe_final_bias": bool(bwe.get("use_bias_at_final", False)),
        "filter_length": bwe["n_fft"],
        "hop_length": bwe["hop_length"],
        "window_length": bwe["win_size"],
        "num_mel_channels": bwe["num_mels"],
        "input_sampling_rate": bwe["input_sampling_rate"],
        "output_sampling_rate": bwe["output_sampling_rate"],
    }


def _compression_kind(name: str) -> str:
    if "space" in name and "time" not in name and "all" not in name:
        return "spatial"
    if "time" in name and "space" not in name and "all" not in name:
        return "temporal"
    if "all" in name:
        return "spatiotemporal"
    raise ValueError(f"Unsupported LTX video VAE compression block: {name!r}")


def _split_vae_blocks(blocks: Any) -> tuple[list[tuple[str, Mapping[str, Any]]], list[tuple[str, Mapping[str, Any]]]]:
    residuals: list[tuple[str, Mapping[str, Any]]] = []
    compressions: list[tuple[str, Mapping[str, Any]]] = []
    for entry in blocks:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[1], Mapping):
            raise ValueError(f"Malformed LTX video VAE block metadata: {entry!r}")
        item = (str(entry[0]), entry[1])
        if item[0] == "res_x":
            residuals.append(item)
        elif item[0].startswith("compress_"):
            compressions.append(item)
        else:
            raise ValueError(f"Unsupported LTX video VAE block: {item[0]!r}")
    return residuals, compressions


def convert_ltx2_video_vae_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert official convolutional video VAE metadata to Diffusers config."""

    encoder_residuals, encoder_compressions = _split_vae_blocks(config["encoder_blocks"])
    decoder_residuals, decoder_compressions = _split_vae_blocks(config["decoder_blocks"])

    def cumulative_channels(base: int, blocks: list[tuple[str, Mapping[str, Any]]]) -> tuple[int, ...]:
        channels = base
        result = []
        for _, params in blocks:
            channels *= int(params.get("multiplier", 1))
            result.append(channels)
        return tuple(result)

    encoder_types = tuple(_compression_kind(name) for name, _ in encoder_compressions)
    decoder_types = tuple(_compression_kind(name) for name, _ in reversed(decoder_compressions))
    spatial_compressions = sum(kind in ("spatial", "spatiotemporal") for kind in encoder_types)
    temporal_compressions = sum(kind in ("temporal", "spatiotemporal") for kind in encoder_types)
    return {
        "in_channels": config["in_channels"],
        "out_channels": config["out_channels"],
        "latent_channels": config["latent_channels"],
        "block_out_channels": cumulative_channels(config["encoder_base_channels"], encoder_compressions),
        "down_block_types": tuple("LTX2VideoDownBlock3D" for _ in encoder_compressions),
        "decoder_block_out_channels": cumulative_channels(config["decoder_base_channels"], decoder_compressions),
        "layers_per_block": tuple(params["num_layers"] for _, params in encoder_residuals),
        "decoder_layers_per_block": tuple(params["num_layers"] for _, params in decoder_residuals),
        "spatio_temporal_scaling": tuple(True for _ in encoder_compressions),
        "decoder_spatio_temporal_scaling": tuple(True for _ in decoder_compressions),
        "decoder_inject_noise": tuple(False for _ in decoder_residuals),
        "downsample_type": encoder_types,
        "upsample_type": decoder_types,
        "upsample_residual": tuple(name.endswith("_res") for name, _ in decoder_compressions),
        "upsample_factor": tuple(int(params.get("multiplier", 1)) for _, params in decoder_compressions),
        "timestep_conditioning": bool(config.get("timestep_conditioning", False)),
        "patch_size": config.get("patch_size", 4),
        "patch_size_t": 1,
        "resnet_norm_eps": 1e-6,
        "encoder_causal": True,
        "decoder_causal": bool(config.get("causal_decoder", False)),
        "encoder_spatial_padding_mode": config.get("spatial_padding_mode", "zeros"),
        "decoder_spatial_padding_mode": config.get("spatial_padding_mode", "zeros"),
        "spatial_compression_ratio": int(config.get("patch_size", 4)) * 2**spatial_compressions,
        "temporal_compression_ratio": 2**temporal_compressions,
        "scaling_factor": config.get("scaling_factor", 1.0),
    }


def convert_ltx2_diffusion_video_vae_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the official LTX-2.5 DiffVAE decoder config."""

    class_name = str(config.get("_class_name", ""))
    if class_name and "diffusion" not in class_name.lower():
        raise ValueError(f"Selected the diffusion video decoder, but checkpoint metadata declares {class_name!r}.")
    return {
        "out_channels": config.get("out_channels", 3),
        "latent_channels": config.get("latent_channels", 128),
        "patch_size": config.get("patch_size", 4),
        "decoder_head_dim": 64,
        "decoder_stage_channels": (2048, 1024, 512, 512, 256),
        "decoder_stage_depths": (4, 6, 4, 2, 8),
        "decoder_stage_kernels": ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)),
        "decoder_upsample_strides": ((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
        "decoder_upsample_channel_reductions": (2, 2, 1, 2),
        "decoder_stage5_kernel": (11, 11, 11),
        "decoder_t_emb_dim": 384,
        "decoder_timestep_scale_multiplier": 1000.0,
        "decoder_model_output_type": "x0",
        "decoder_num_inference_steps": 1,
        "spatial_compression_ratio": 32,
        "temporal_compression_ratio": 8,
    }


def convert_ltx2_upsampler_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.pop("_class_name", None)
    result["rational_spatial_scale"] = result.pop("spatial_scale")
    result["use_rational_resampler"] = result.pop("rational_resampler")
    return result


def decode_ltx2_embedded_asset(tensor: torch.Tensor) -> bytes:
    """Decode an official uint8/int8 tensor-backed embedded file."""

    if tensor.ndim != 1 or tensor.dtype not in (torch.uint8, torch.int8):
        raise ValueError("Embedded Gemma assets must be one-dimensional uint8 or int8 tensors.")
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def load_ltx2_embedded_gemma_assets(path: str | PathLike[str]) -> dict[str, bytes]:
    """Load only tokenizer/processor assets from the packed Gemma checkpoint."""

    assets: dict[str, bytes] = {}
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for tensor_name, asset_name in _GEMMA_ASSET_KEYS.items():
            if tensor_name in keys:
                assets[asset_name] = decode_ltx2_embedded_asset(checkpoint.get_tensor(tensor_name))
    if "tokenizer.json" not in assets:
        raise ValueError(f"Packed Gemma checkpoint {path} does not contain tokenizer_json.")
    return assets


def map_ltx2_transformer_weight(key: str) -> str | None:
    prefix = "model.diffusion_model."
    if not key.startswith(prefix):
        return None
    name = key.removeprefix(prefix)
    if name.startswith(("video_embeddings_connector.", "audio_embeddings_connector.")):
        return None
    replacements = (
        ("audio_patchify_proj", "audio_proj_in"),
        ("patchify_proj", "proj_in"),
        ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
        ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
        ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
        ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
        ("audio_prompt_adaln_single", "audio_prompt_adaln"),
        ("prompt_adaln_single", "prompt_adaln"),
        ("audio_adaln_single", "audio_time_embed"),
        ("adaln_single", "time_embed"),
        ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
        ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
    )
    for old, new in replacements:
        name = name.replace(old, new)
    name = name.replace(".q_norm.", ".norm_q.").replace(".k_norm.", ".norm_k.")
    return f"transformer.{name}"


def map_ltx2_connector_weight(key: str) -> str | None:
    transformer_prefixes = {
        "model.diffusion_model.video_embeddings_connector.": "video_connector.",
        "model.diffusion_model.audio_embeddings_connector.": "audio_connector.",
    }
    text_prefixes = {
        "text_embedding_projection.video_aggregate_embed.": "video_text_proj_in.",
        "text_embedding_projection.audio_aggregate_embed.": "audio_text_proj_in.",
    }
    name = None
    for prefix, replacement in (*transformer_prefixes.items(), *text_prefixes.items()):
        if key.startswith(prefix):
            name = replacement + key.removeprefix(prefix)
            break
    if name is None:
        return None
    name = name.replace("transformer_1d_blocks", "transformer_blocks")
    name = name.replace(".q_norm.", ".norm_q.").replace(".k_norm.", ".norm_k.")
    return f"connectors.{name}"


def map_ltx2_text_encoder_weight(key: str) -> str | None:
    if key in _GEMMA_ASSET_KEYS or key.startswith("hf_asset__") or key.startswith("text_embedding_projection."):
        return None
    prefixes = {
        "model.layers.": "model.language_model.layers.",
        "model.embed_tokens.": "model.language_model.embed_tokens.",
        "model.norm.": "model.language_model.norm.",
        "vision_model.": "model.vision_embedder.",
        "multi_modal_projector.": "model.embed_vision.",
        "audio_projector.": "model.embed_audio.",
    }
    for prefix, replacement in prefixes.items():
        if key.startswith(prefix):
            return f"text_encoder.{replacement}{key.removeprefix(prefix)}"
    return None


def map_ltx2_audio_vae_weight(key: str) -> str | None:
    if not key.startswith("audio_vae."):
        return None
    name = key.removeprefix("audio_vae.")
    name = name.replace("per_channel_statistics.mean-of-means", "latents_mean")
    name = name.replace("per_channel_statistics.std-of-means", "latents_std")
    return f"audio_vae.{name}"


def map_ltx2_vocoder_weight(key: str) -> str | None:
    if not key.startswith("vocoder."):
        return None
    name = key.removeprefix("vocoder.")
    name = name.replace("conv_pre", "conv_in").replace("conv_post", "conv_out")
    name = name.replace("resblocks", "resnets").replace("act_post", "act_out")
    name = name.replace(".downsample.lowpass.filter", ".downsample.filter")
    name = name.replace(".ups.", ".upsamplers.")
    return f"vocoder.{name}"


def map_ltx2_video_vae_weight(key: str) -> str | None:
    statistics = {
        "per_channel_statistics.mean-of-means": "vae.latents_mean",
        "per_channel_statistics.std-of-means": "vae.latents_std",
    }
    if key in statistics:
        return statistics[key]
    if key.startswith("per_channel_statistics."):
        return None

    name = key
    encoder_match = re.match(r"^encoder\.down_blocks\.(\d+)\.(.+)$", name)
    if encoder_match:
        index, suffix = int(encoder_match.group(1)), encoder_match.group(2)
        if index == 8:
            name = f"encoder.mid_block.{suffix}"
        elif index % 2 == 0:
            name = f"encoder.down_blocks.{index // 2}.{suffix}"
        else:
            name = f"encoder.down_blocks.{(index - 1) // 2}.downsamplers.0.{suffix}"
    decoder_match = re.match(r"^decoder\.up_blocks\.(\d+)\.(.+)$", name)
    if decoder_match:
        index, suffix = int(decoder_match.group(1)), decoder_match.group(2)
        if index == 0:
            name = f"decoder.mid_block.{suffix}"
        elif index % 2 == 0:
            name = f"decoder.up_blocks.{index // 2 - 1}.{suffix}"
        else:
            name = f"decoder.up_blocks.{(index - 1) // 2}.upsamplers.0.{suffix}"
    name = name.replace("res_blocks", "resnets")
    name = name.replace("last_time_embedder", "time_embedder")
    name = name.replace("last_scale_shift_table", "scale_shift_table")
    return f"vae.{name}"


def map_ltx2_upsampler_weight(key: str) -> str:
    return f"latent_upsampler.{key}"


def map_ltx2_transformer_file_weight(key: str) -> str | None:
    """Map a tensor from the raw transformer file to a pipeline key."""

    return map_ltx2_connector_weight(key) or map_ltx2_transformer_weight(key)


def map_ltx2_text_encoder_file_weight(key: str) -> str | None:
    """Map a tensor from the packed Gemma file to a pipeline key."""

    return map_ltx2_connector_weight(key) or map_ltx2_text_encoder_weight(key)


def map_ltx2_audio_file_weight(key: str) -> str | None:
    """Map a tensor from the combined audio-VAE/vocoder file."""

    return map_ltx2_audio_vae_weight(key) or map_ltx2_vocoder_weight(key)
