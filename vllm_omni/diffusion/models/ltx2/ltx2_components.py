# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared component construction helpers for the LTX model family."""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.video_processor import VideoProcessor
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import (
    CONFIG_MAPPING,
    AutoModelForImageTextToText,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    PreTrainedTokenizerFast,
)

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention as OmniAttention
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import DistributedAutoencoderKLLTX2Video
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.utils.hf_utils import is_ltx25_raw_checkpoint

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .ltx2_raw_checkpoint import (
    LTX2RawCheckpointLayout,
    convert_ltx2_scheduler_config,
    inspect_ltx2_raw_checkpoint,
    load_ltx2_embedded_gemma_assets,
    materialize_ltx2_raw_checkpoint,
)
from .ltx2_transformer import (
    LTX2VideoTransformer3DModel,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
    to_ltx_padding_mask,
)

try:
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
except ImportError:
    LTX2VocoderWithBWE = None

try:
    from transformers import Gemma4UnifiedForConditionalGeneration as _Gemma4UnifiedForConditionalGeneration
except ImportError:
    _Gemma4UnifiedForConditionalGeneration = None


_LTX25_TEXT_ENCODER_CLS = AutoModelForImageTextToText if _Gemma4UnifiedForConditionalGeneration is not None else None

_LTX_COMPONENT_SUBFOLDERS = (
    "tokenizer",
    "text_encoder",
    "connectors",
    "vae",
    "audio_vae",
    "vocoder",
    "scheduler",
    "latent_upsampler",
)
logger = logging.getLogger(__name__)

LTXCheckpointVariant = Literal["full", "distilled"]


@dataclass(frozen=True)
class LTXComponentProfile:
    """Component construction and discovery contract for an LTX variant."""

    name: str
    dit_modules: tuple[str, ...]
    encoder_modules: tuple[str, ...]
    vae_modules: tuple[str, ...]
    resident_modules: tuple[str, ...] = ()
    video_vae_cls: type = AutoencoderKLLTX2Video
    vocoder_cls: type = LTX2Vocoder
    text_encoder_cls: type | None = Gemma3ForConditionalGeneration
    vocoder_fallback_cls: type | None = None
    transformer_subfolder: str = "transformer"
    scheduler_use_dynamic_shifting: bool = False
    scheduler_shift_terminal: float | None = None


LTX2_COMPONENT_PROFILE = LTXComponentProfile(
    name="ltx2",
    dit_modules=("transformer",),
    encoder_modules=("text_encoder", "connectors"),
    vae_modules=("vae", "audio_vae"),
    resident_modules=("vocoder",),
    video_vae_cls=DistributedAutoencoderKLLTX2Video,
)

LTX23_COMPONENT_PROFILE = LTXComponentProfile(
    name="ltx2_3",
    dit_modules=("transformer",),
    encoder_modules=("text_encoder", "connectors"),
    vae_modules=("vae", "audio_vae"),
    resident_modules=("vocoder",),
    video_vae_cls=DistributedAutoencoderKLLTX2Video,
    vocoder_cls=LTX2VocoderWithBWE or LTX2Vocoder,
    vocoder_fallback_cls=LTX2Vocoder,
)

# The converted LTX-2.5 checkpoint stores the official dev/SFT weights in
# transformer_full/. Restore the official LTX2Scheduler defaults that its
# distilled model_index disables.
LTX25_FULL_COMPONENT_PROFILE = LTXComponentProfile(
    name="ltx2_5_full",
    dit_modules=("transformer",),
    encoder_modules=("text_encoder", "connectors"),
    vae_modules=("vae", "audio_vae"),
    resident_modules=("vocoder",),
    video_vae_cls=DistributedAutoencoderKLLTX2Video,
    vocoder_cls=LTX2VocoderWithBWE or LTX2Vocoder,
    vocoder_fallback_cls=LTX2Vocoder,
    text_encoder_cls=_LTX25_TEXT_ENCODER_CLS,
    transformer_subfolder="transformer_full",
    scheduler_use_dynamic_shifting=True,
    scheduler_shift_terminal=0.1,
)


LTX25_FULL_TWO_STAGE_COMPONENT_PROFILE = LTXComponentProfile(
    name="ltx2_5_full_two_stage",
    dit_modules=("transformer",),
    encoder_modules=("text_encoder", "connectors"),
    vae_modules=("vae", "audio_vae"),
    resident_modules=("vocoder", "latent_upsampler"),
    video_vae_cls=DistributedAutoencoderKLLTX2Video,
    vocoder_cls=LTX2VocoderWithBWE or LTX2Vocoder,
    vocoder_fallback_cls=LTX2Vocoder,
    text_encoder_cls=_LTX25_TEXT_ENCODER_CLS,
    transformer_subfolder="transformer_full",
    scheduler_use_dynamic_shifting=True,
    scheduler_shift_terminal=0.1,
)


LTX2_DISTILLED_COMPONENT_PROFILE = LTXComponentProfile(
    name="ltx2_distilled",
    dit_modules=("transformer",),
    encoder_modules=("text_encoder", "connectors"),
    vae_modules=("vae", "audio_vae"),
    resident_modules=("vocoder", "latent_upsampler"),
    video_vae_cls=DistributedAutoencoderKLLTX2Video,
)

LTX25_DISTILLED_COMPONENT_PROFILE = LTXComponentProfile(
    name="ltx2_5_distilled",
    dit_modules=("transformer",),
    encoder_modules=("text_encoder", "connectors"),
    vae_modules=("vae", "audio_vae"),
    resident_modules=("vocoder", "latent_upsampler"),
    video_vae_cls=DistributedAutoencoderKLLTX2Video,
    vocoder_cls=LTX2VocoderWithBWE or LTX2Vocoder,
    vocoder_fallback_cls=LTX2Vocoder,
    text_encoder_cls=_LTX25_TEXT_ENCODER_CLS,
)


_COMPONENT_PROFILES: dict[tuple[str, str, LTXCheckpointVariant], LTXComponentProfile] = {
    ("one_stage", "2", "full"): LTX2_COMPONENT_PROFILE,
    ("one_stage", "2.3", "full"): LTX23_COMPONENT_PROFILE,
    ("one_stage", "2.5", "full"): LTX25_FULL_COMPONENT_PROFILE,
    ("two_stage", "2", "distilled"): LTX2_DISTILLED_COMPONENT_PROFILE,
    ("two_stage", "2.5", "distilled"): LTX25_DISTILLED_COMPONENT_PROFILE,
    ("two_stage", "2.5", "full"): LTX25_FULL_TWO_STAGE_COMPONENT_PROFILE,
    ("dmd2", "2", "distilled"): LTX2_COMPONENT_PROFILE,
    ("dmd2", "2.3", "distilled"): LTX23_COMPONENT_PROFILE,
}


def resolve_ltx_checkpoint_variant(
    pipeline_kind: str,
    model_version: str,
    task_type: str | None,
    *,
    compatibility_override: LTXCheckpointVariant | None = None,
) -> LTXCheckpointVariant:
    """Resolve checkpoint weights independently from pipeline topology."""
    normalized = None if task_type is None else str(task_type).strip().lower().replace("-", "_")
    aliases: dict[str, LTXCheckpointVariant] = {
        "full": "full",
        "dev": "full",
        "sft": "full",
        "distilled": "distilled",
        "distill": "distilled",
    }
    if normalized in (None, "", "auto"):
        explicit = None
    else:
        try:
            explicit = aliases[normalized]
        except KeyError as exc:
            raise ValueError(
                "LTX task_type must select checkpoint weights with one of "
                f"auto, full/dev/sft, or distilled; got {task_type!r}."
            ) from exc

    if compatibility_override is not None and explicit is not None and compatibility_override != explicit:
        raise ValueError(
            f"{pipeline_kind} compatibility entry requires {compatibility_override!r} weights, "
            f"but task_type selected {explicit!r}."
        )
    variant = explicit or compatibility_override
    if variant is None:
        variant = "distilled" if pipeline_kind in ("two_stage", "dmd2") else "full"

    if pipeline_kind == "one_stage" and variant == "distilled":
        raise ValueError(
            "Official LTX does not define a distilled one-stage pipeline. Use "
            "LTX2TwoStagePipeline with task_type='distilled', or select task_type='full' for LTX2Pipeline."
        )
    if model_version != "2.5" and pipeline_kind == "two_stage" and variant == "full":
        raise ValueError("Full guided two-stage execution is currently supported for LTX-2.5 checkpoints only.")
    return variant


def resolve_ltx_component_profile(
    pipeline_kind: str,
    model_version: str,
    checkpoint_variant: LTXCheckpointVariant,
) -> LTXComponentProfile:
    """Resolve component construction independently from execution recipes."""
    try:
        return _COMPONENT_PROFILES[(pipeline_kind, model_version, checkpoint_variant)]
    except KeyError as exc:
        raise ValueError(
            "Unsupported LTX component topology/version/checkpoint variant: "
            f"{pipeline_kind!r}/{model_version!r}/{checkpoint_variant!r}."
        ) from exc


def _load_ltx_metadata_json(model: str, filename: str, revision: str | None = None) -> dict[str, Any]:
    """Load small checkpoint metadata without relying on repository names."""
    if os.path.isdir(model):
        path = os.path.join(model, filename)
        if not os.path.isfile(path):
            return {}
    else:
        try:
            path = hf_hub_download(repo_id=model, filename=filename, revision=revision)
        except Exception:
            return {}
    try:
        with open(path) as config_file:
            value = json.load(config_file)
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def detect_ltx_model_version(model: str, revision: str | None = None) -> str:
    """Detect the LTX model version from checkpoint component metadata.

    Official checkpoints use ``model_version`` metadata. Diffusers repositories
    expose LTX-2.5 through the Gemma4 Unified text encoder and LTX-2.3 through
    the BWE vocoder. Unknown conversions retain the official LTX-2 fallback.
    """
    if is_ltx25_raw_checkpoint(model, revision=revision):
        return "2.5"

    model_index = _load_ltx_metadata_json(model, "model_index.json", revision)
    model_version = str(model_index.get("model_version", ""))
    if model_version.startswith("2.5"):
        return "2.5"
    text_encoder_entry = model_index.get("text_encoder")
    if isinstance(text_encoder_entry, (list, tuple)) and text_encoder_entry:
        text_encoder_class = str(text_encoder_entry[-1])
    elif isinstance(text_encoder_entry, dict):
        text_encoder_class = str(text_encoder_entry.get("_class_name", ""))
    else:
        text_encoder_class = ""
    if text_encoder_class == "Gemma4UnifiedForConditionalGeneration":
        return "2.5"

    text_encoder_config = _load_ltx_metadata_json(model, "text_encoder/config.json", revision)
    if text_encoder_config.get("model_type") in ("gemma4_unified", "gemma4"):
        return "2.5"

    # Explicit checkpoint metadata takes precedence over structural heuristics.
    if model_version.startswith("2.3"):
        return "2.3"

    # Converted checkpoints may record an AutoModel class in model_index.json
    # instead of the concrete Gemma4 class. The 2.5 transformer drops the
    # video FFN bias; this is the only transformer-config delta from 2.3 that
    # is present in both 2.5 and 2.5.1+ conversions.
    transformer_config = _load_ltx_metadata_json(model, "transformer/config.json", revision)
    if transformer_config.get("ff_bias") is False:
        return "2.5"

    vocoder_entry = model_index.get("vocoder")
    if isinstance(vocoder_entry, (list, tuple)) and vocoder_entry:
        vocoder_class = str(vocoder_entry[-1])
    elif isinstance(vocoder_entry, dict):
        vocoder_class = str(vocoder_entry.get("_class_name", ""))
    else:
        vocoder_class = ""
    if vocoder_class == "LTX2VocoderWithBWE":
        return "2.3"

    vocoder_config = _load_ltx_metadata_json(model, "vocoder/config.json", revision)
    if str(vocoder_config.get("model_version", "")).startswith("2.3"):
        return "2.3"
    if vocoder_config.get("_class_name") == "LTX2VocoderWithBWE":
        return "2.3"

    logger.info("Using LTX-2 defaults for checkpoint %s", model)
    return "2"


class _LTXConnectorAttnProcessor:
    """Preserve official connector math around Omni attention dispatch."""

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Offload hooks may execute affine-free Q/K norms in FP32. Restore the
        # projection dtype before attention, matching the fully resident path.
        query = attn.norm_q(query).to(dtype=value.dtype)
        key = attn.norm_k(key).to(dtype=value.dtype)

        if query_rotary_emb is not None:
            # Diffusers builds connector RoPE in FP32, while the official
            # connector materializes it in the hidden-state dtype.
            query_rotary_emb = tuple(component.to(value.dtype) for component in query_rotary_emb)
            key_rotary_emb = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
            key_rotary_emb = tuple(component.to(value.dtype) for component in key_rotary_emb)
            if attn.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(key, key_rotary_emb)
            elif attn.rope_type == "split":
                query = apply_split_rotary_emb(query, query_rotary_emb, head_dim=attn.head_dim)
                key = apply_split_rotary_emb(key, key_rotary_emb, head_dim=attn.head_dim)
            else:
                raise ValueError(f"Unsupported LTX connector RoPE type: {attn.rope_type}")

        # Keep Q/K in the projection dtype expected by the attention backend.
        query = query.to(dtype=value.dtype)
        key = key.to(dtype=value.dtype)

        batch_size, _, inner_dim = query.shape
        head_dim = inner_dim // attn.heads
        kv_heads = attn.inner_kv_dim // attn.head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Official learned-register connectors keep the resulting all-zero
        # additive mask and therefore use the masked SDPA path. Preserve that
        # dispatch; Flash backends consume the equivalent 2D all-keep mask.
        if attention_mask is not None and attn.omni_attention.attn_backend.get_name().upper() == "FLASH_ATTN":
            attention_mask = to_ltx_padding_mask(attention_mask)
        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        hidden_states = attn.omni_attention(query, key, value, attn_metadata)
        hidden_states = hidden_states.reshape(batch_size, -1, inner_dim)

        if gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            hidden_states = hidden_states * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


def _install_connector_attention(connectors: LTX2TextConnectors) -> None:
    for connector_name in ("video_connector", "audio_connector"):
        connector = getattr(connectors, connector_name, None)
        for block_index, block in enumerate(getattr(connector, "transformer_blocks", ())):
            attention = getattr(block, "attn1", None)
            if attention is not None:
                attention.omni_attention = OmniAttention(
                    num_heads=attention.heads,
                    head_size=attention.head_dim,
                    num_kv_heads=attention.inner_kv_dim // attention.head_dim,
                    softmax_scale=1.0 / (attention.head_dim**0.5),
                    causal=False,
                    prefix=f"connectors.{connector_name}.transformer_blocks.{block_index}.attn1",
                    role="ltx2.connector",
                    role_category="self",
                    skip_sequence_parallel=True,
                    disable_kv_quant=True,
                )
                attention.set_processor(_LTXConnectorAttnProcessor())


def _detect_vocoder_output_sample_rate(model: str, revision: str | None = None) -> int | None:
    """Read the generated waveform sample rate from the vocoder config."""
    # The official split LTX-2.5 checkpoint embeds the vocoder config in the
    # safetensors header instead of publishing ``vocoder/config.json``. Its
    # BWE vocoder has a fixed 48 kHz output rate; avoid downloading the large
    # weight file merely to inspect that header during post-processing setup.
    if is_ltx25_raw_checkpoint(model, revision=revision):
        return 48_000

    vocoder_config_path = os.path.join(model, "vocoder", "config.json")
    if not os.path.exists(vocoder_config_path):
        try:
            vocoder_config_path = hf_hub_download(model, "vocoder/config.json", revision=revision)
        except Exception:
            return None
    try:
        with open(vocoder_config_path) as config_file:
            return json.load(config_file).get("output_sampling_rate")
    except Exception:
        return None


def get_ltx2_post_process_func(od_config: Any):
    """Build the common LTX engine-output adapter."""
    output_sample_rate = _detect_vocoder_output_sample_rate(
        od_config.model,
        revision=getattr(od_config, "revision", None),
    )

    def post_process_func(output: tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
        if not (isinstance(output, tuple) and len(output) == 2):
            return output
        video, audio = output
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu()
        result: dict[str, Any] = {"video": video, "audio": audio}
        if output_sample_rate is not None:
            result["audio_sample_rate"] = output_sample_rate
        return result

    return post_process_func


def _load_component(
    component_cls: type,
    model: str,
    subfolder: str,
    *,
    local_files_only: bool,
    dtype: torch.dtype,
    revision: str | None,
) -> Any:
    return from_pretrained_with_prefetch(
        component_cls.from_pretrained,
        model,
        subfolder=subfolder,
        prefetch_list=_LTX_COMPONENT_SUBFOLDERS,
        local_files_only=local_files_only,
        revision=revision,
        torch_dtype=dtype,
    )


def _place_aux_components(pipeline: Any) -> None:
    parallel_config = getattr(pipeline.od_config, "parallel_config", None)
    use_managed_placement = bool(
        getattr(pipeline.od_config, "enable_cpu_offload", False)
        or getattr(pipeline.od_config, "enable_layerwise_offload", False)
        or getattr(parallel_config, "use_hsdp", False)
    )
    if use_managed_placement:
        return

    modules = ModuleDiscovery.discover(pipeline)
    for module in (*modules.encoders, *modules.vaes, *modules.resident_modules):
        module.to(pipeline.device)


def _raw_component_source(
    layout: LTX2RawCheckpointLayout,
    path: os.PathLike[str],
    revision: str | None,
) -> DiffusersPipelineLoader.ComponentSource:
    relative_path = os.path.relpath(path, layout.root)
    return DiffusersPipelineLoader.ComponentSource(
        model_or_path=str(layout.root),
        subfolder=os.path.dirname(relative_path),
        revision=revision,
        prefix="",
        fall_back_to_pt=False,
        allow_patterns_overrides=[os.path.basename(relative_path)],
    )


def _build_ltx2_raw_weight_sources(
    layout: LTX2RawCheckpointLayout,
    revision: str | None,
    *,
    include_latent_upsampler: bool,
) -> list[DiffusersPipelineLoader.ComponentSource]:
    paths = [layout.transformer, layout.text_encoder, layout.audio_vae, layout.video_vae]
    if include_latent_upsampler:
        if layout.latent_upsampler is None:
            raise ValueError("LTX-2.5 two-stage execution requires the latent upsampler weights.")
        paths.append(layout.latent_upsampler)
    return [_raw_component_source(layout, path, revision) for path in paths]


def _build_ltx2_raw_tokenizer(assets: dict[str, bytes]) -> PreTrainedTokenizerFast:
    tokenizer_config = json.loads(assets.get("tokenizer_config.json", b"{}"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_buffer(assets["tokenizer.json"]),
        **tokenizer_config,
    )
    if chat_template := assets.get("chat_template.jinja"):
        tokenizer.chat_template = chat_template.decode("utf-8")
    return tokenizer


def _initialize_raw_pipeline_components(pipeline: Any, od_config: Any, dtype: torch.dtype) -> None:
    """Construct components for the official split BF16 checkpoint.

    Only the official ConvVAE tensor layout is mapped today. The separate
    DiffVAE decoder requires QKV splitting and gated-convolution folding and is
    rejected instead of being advertised as a working raw-checkpoint path.
    """

    decoder_type = getattr(od_config, "ltx2_video_decoder_type", "conv")
    if decoder_type != "conv":
        raise ValueError(
            "Official raw LTX-2.5 checkpoints currently support only the ConvVAE decoder; "
            "DiffVAE tensor mapping is not implemented."
        )
    if pipeline.component_profile.text_encoder_cls is None:
        raise ImportError("LTX-2.5 requires Gemma4UnifiedForConditionalGeneration. Install transformers>=5.10.1,<5.15.")
    if LTX2VocoderWithBWE is None:
        raise ImportError("Official LTX-2.5 audio decoding requires diffusers with LTX2VocoderWithBWE support.")

    revision = getattr(od_config, "revision", None)
    require_upsampler = "latent_upsampler" in pipeline.component_profile.resident_modules
    layout = materialize_ltx2_raw_checkpoint(
        od_config.model,
        checkpoint_variant=pipeline.checkpoint_variant,
        video_decoder_type="conv",
        require_latent_upsampler=require_upsampler,
        revision=revision,
    )
    metadata = inspect_ltx2_raw_checkpoint(layout)
    if metadata.model_version is not None and not metadata.model_version.startswith("2.5"):
        raise ValueError(f"Expected LTX-2.5 raw components, found model_version={metadata.model_version!r}.")

    pipeline._ltx2_raw_checkpoint = True
    pipeline._ltx2_raw_checkpoint_layout = layout
    pipeline.weights_sources = _build_ltx2_raw_weight_sources(
        layout, revision, include_latent_upsampler=require_upsampler
    )
    pipeline.tokenizer = _build_ltx2_raw_tokenizer(load_ltx2_embedded_gemma_assets(layout.text_encoder))

    model_type = metadata.gemma.get("model_type")
    if not isinstance(model_type, str) or model_type not in CONFIG_MAPPING:
        raise ValueError(f"Unsupported embedded LTX-2.5 text encoder model_type: {model_type!r}.")
    gemma_config = CONFIG_MAPPING[model_type].from_dict(metadata.gemma)
    with torch.device("cpu"):
        pipeline.text_encoder = pipeline.component_profile.text_encoder_cls.from_config(gemma_config, dtype=dtype)

    pipeline.connectors = LTX2TextConnectors.from_config(metadata.connectors)
    _install_connector_attention(pipeline.connectors)
    pipeline.vae = DistributedAutoencoderKLLTX2Video.from_config(metadata.video_vae)
    pipeline.vae.init_distributed()
    pipeline.audio_vae = AutoencoderKLLTX2Audio.from_config(metadata.audio_vae)
    pipeline.vocoder = LTX2VocoderWithBWE.from_config(metadata.vocoder)

    if require_upsampler:
        if metadata.latent_upsampler is None:
            raise ValueError("Official LTX-2.5 two-stage execution requires latent upsampler metadata.")
        with torch.device("cpu"):
            pipeline.latent_upsampler = LTX2LatentUpsamplerModel.from_config(metadata.latent_upsampler)

    quant_config = getattr(od_config, "quantization_config", None)
    pipeline.transformer = create_transformer_from_config(metadata.transformer, quant_config=quant_config)
    _place_aux_components(pipeline)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(convert_ltx2_scheduler_config(metadata.scheduler))
    if pipeline.component_profile.scheduler_use_dynamic_shifting:
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            use_dynamic_shifting=True,
            shift_terminal=pipeline.component_profile.scheduler_shift_terminal,
        )


def initialize_pipeline_components(pipeline: Any, od_config: Any) -> None:
    """Build the common LTX component graph selected by ``component_profile``."""
    profile: LTXComponentProfile = pipeline.component_profile
    pipeline.od_config = od_config
    pipeline.device = get_local_device()
    dtype = getattr(od_config, "dtype", torch.bfloat16)
    model = od_config.model
    revision = getattr(od_config, "revision", None)
    if is_ltx25_raw_checkpoint(model, revision=revision):
        _initialize_raw_pipeline_components(pipeline, od_config, dtype)
        _finalize_pipeline_components(pipeline)
        return
    local_files_only = os.path.exists(model)

    pipeline.weights_sources = [
        DiffusersPipelineLoader.ComponentSource(
            model_or_path=model,
            subfolder=profile.transformer_subfolder,
            revision=revision,
            prefix="transformer.",
            fall_back_to_pt=True,
        ),
    ]
    prefetch_subfolders(model, _LTX_COMPONENT_SUBFOLDERS, local_files_only=local_files_only, revision=revision)

    pipeline.tokenizer = AutoTokenizer.from_pretrained(
        model,
        subfolder="tokenizer",
        local_files_only=local_files_only,
        revision=revision,
    )
    if profile.text_encoder_cls is None:
        raise ImportError("LTX-2.5 requires Gemma4UnifiedForConditionalGeneration. Install transformers>=5.10.1,<5.15.")
    with torch.device("cpu"):
        pipeline.text_encoder = _load_component(
            profile.text_encoder_cls,
            model,
            "text_encoder",
            local_files_only=local_files_only,
            dtype=dtype,
            revision=revision,
        )
    pipeline.connectors = _load_component(
        LTX2TextConnectors,
        model,
        "connectors",
        local_files_only=local_files_only,
        dtype=dtype,
        revision=revision,
    )
    _install_connector_attention(pipeline.connectors)
    pipeline.vae = _load_component(
        profile.video_vae_cls,
        model,
        "vae",
        local_files_only=local_files_only,
        dtype=dtype,
        revision=revision,
    )
    pipeline.audio_vae = _load_component(
        AutoencoderKLLTX2Audio,
        model,
        "audio_vae",
        local_files_only=local_files_only,
        dtype=dtype,
        revision=revision,
    )
    try:
        pipeline.vocoder = _load_component(
            profile.vocoder_cls,
            model,
            "vocoder",
            local_files_only=local_files_only,
            dtype=dtype,
            revision=revision,
        )
    except (TypeError, OSError, ValueError):
        if profile.vocoder_fallback_cls is None or profile.vocoder_fallback_cls is profile.vocoder_cls:
            raise
        pipeline.vocoder = _load_component(
            profile.vocoder_fallback_cls,
            model,
            "vocoder",
            local_files_only=local_files_only,
            dtype=dtype,
            revision=revision,
        )

    if "latent_upsampler" in profile.resident_modules:
        # BlurDownsample constructs an integer kernel that must be initialized
        # on CPU; component placement is handled uniformly after construction.
        with torch.device("cpu"):
            pipeline.latent_upsampler = _load_component(
                LTX2LatentUpsamplerModel,
                model,
                "latent_upsampler",
                local_files_only=local_files_only,
                dtype=dtype,
                revision=revision,
            )

    transformer_config = load_transformer_config(
        model, profile.transformer_subfolder, local_files_only, revision=revision
    )
    quant_config = getattr(od_config, "quantization_config", None)
    pipeline.transformer = create_transformer_from_config(transformer_config, quant_config=quant_config)
    _place_aux_components(pipeline)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model,
        subfolder="scheduler",
        local_files_only=local_files_only,
        revision=revision,
    )
    if profile.scheduler_use_dynamic_shifting:
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            use_dynamic_shifting=True,
            shift_terminal=profile.scheduler_shift_terminal,
        )

    _finalize_pipeline_components(pipeline)


def _finalize_pipeline_components(pipeline: Any) -> None:
    pipeline.vae_spatial_compression_ratio = pipeline.vae.spatial_compression_ratio
    pipeline.vae_temporal_compression_ratio = pipeline.vae.temporal_compression_ratio
    pipeline.audio_vae_mel_compression_ratio = pipeline.audio_vae.mel_compression_ratio
    pipeline.audio_vae_temporal_compression_ratio = pipeline.audio_vae.temporal_compression_ratio
    pipeline.transformer_spatial_patch_size = pipeline.transformer.config.patch_size
    pipeline.transformer_temporal_patch_size = pipeline.transformer.config.patch_size_t
    pipeline.audio_sampling_rate = pipeline.audio_vae.config.sample_rate
    pipeline.audio_hop_length = pipeline.audio_vae.config.mel_hop_length
    pipeline.video_processor = VideoProcessor(vae_scale_factor=pipeline.vae_spatial_compression_ratio)

    tokenizer_max_length = pipeline.tokenizer.model_max_length
    if tokenizer_max_length is None or tokenizer_max_length > 100000:
        encoder_config = getattr(pipeline.text_encoder, "config", None)
        tokenizer_max_length = getattr(encoder_config, "max_position_embeddings", None)
        if tokenizer_max_length is None:
            tokenizer_max_length = getattr(encoder_config, "max_seq_len", None)
    pipeline.tokenizer_max_length = int(tokenizer_max_length or 1024)

    pipeline._interrupt = False


def load_transformer_config(
    model_path: str,
    subfolder: str = "transformer",
    local_files_only: bool = True,
    *,
    revision: str | None = None,
) -> dict:
    """Load an LTX transformer config from a local model or the HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"LTX transformer config not found: {config_path}")
    else:
        config_path = hf_hub_download(
            repo_id=model_path,
            filename=f"{subfolder}/config.json",
            revision=revision,
        )
    with open(config_path) as config_file:
        return json.load(config_file)


def create_transformer_from_config(
    config: dict,
    quant_config: QuantizationConfig | None = None,
) -> LTX2VideoTransformer3DModel:
    """Construct the shared LTX transformer from a Diffusers config."""
    if not config and quant_config is None:
        return LTX2VideoTransformer3DModel()

    signature = inspect.signature(LTX2VideoTransformer3DModel.__init__)
    allowed_keys = set(signature.parameters)
    kwargs = {key: value for key, value in config.items() if key in allowed_keys}
    if quant_config is not None:
        kwargs["quant_config"] = quant_config

    return LTX2VideoTransformer3DModel(**kwargs)
