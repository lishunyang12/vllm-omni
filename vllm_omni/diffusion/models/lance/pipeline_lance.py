# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LancePipeline — Lance (ByteDance) packaged for the vLLM-Omni diffusion engine.

Lance is BAGEL-lineage (Qwen2-MoT unified AR+diffusion), so the transformer
core and the entire generation/forward machinery are inherited unchanged from
:class:`BagelPipeline`. Only model *construction* differs, in three
well-localized places:

  1. Checkpoint layout — the HF repo ``bytedance-research/Lance`` bundles
     ``Lance_3B/`` (image) and ``Lance_3B_Video/`` (video) LLM checkpoints,
     ``Qwen2.5-VL-ViT/`` (understanding ViT) and ``Wan2.2_VAE.pth`` (VAE) in a
     single repo. There is no BAGEL-style top-level ``config.json``; the Lance
     constants are taken from upstream ``config/config_factory.py`` /
     ``inference_lance.sh`` and hardcoded in :data:`LANCE_DEFAULTS`.
  2. Understanding ViT — Qwen2.5-VL vision tower instead of SigLIP.
  3. VAE — Wan2.2 (``Wan2.2_VAE.pth``) instead of the BAGEL autoencoder.

Single-stage covers all six modalities (``t2i``, ``t2v``, ``image_edit``,
``video_edit``, ``x2t_image``, ``x2t_video``); the video paths use the
``Lance_3B_Video`` checkpoint, :class:`LancePositionEmbedding3D` and
:meth:`LanceWanVAE.decode_video`. The two-stage AR+DiT topology is a
follow-up (needs ``LanceConfig`` / ``LanceProcessor`` in the ``vllm``
package).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from PIL import Image
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.bagel.pipeline_bagel import (
    BagelPipeline,
    add_special_tokens,
)
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

from .lance_transformer import (
    LanceBagel,
    LanceIdentityConnector,
    LancePositionEmbedding3D,
    LanceQwen2_5_VLNaViTWrapper,
    LanceZeroVitPosEmbed,
    Qwen2MoTConfig,
    Qwen2MoTForCausalLM,
)
from .wan_vae import LanceWanVAE

logger = init_logger(__name__)


@dataclass(frozen=True)
class LanceDefaults:
    """Lance constants that upstream keeps in ``config/config_factory.py`` /
    ``inference_lance.sh`` rather than in any shipped JSON.

    Verified against ``bytedance-research/Lance``'s released ``Lance_3B/model.safetensors``:
    ``vae2llm.weight = (2048, 48)`` ⇒ ``patch_latent_dim = latent_patch_size**2 * z_channels = 1 * 48``,
    i.e. Lance does *not* unfold the Wan latent into a 2×2 patch the way BAGEL does (Wan2.2 already
    patchifies internally), and ``latent_pos_embed.pos_embed = (4096, 2048)`` ⇒ ``max_latent_size = 64``.
    """

    # latent / patch geometry (pt, ph, pw); image path only uses the spatial 1.
    latent_patch_size_spatial: int = 1
    latent_patch_size_temporal: int = 1
    max_latent_size: int = 64
    # Lance_3B_Video ships ``latent_pos_embed.pos_embed`` of shape
    # ``(31 * 64 * 64, 2048) = (126976, 2048)`` ⇒ max_num_frames in latent
    # space is 31, equivalent to ``(num_frames - 1) // downsample_temporal + 1``
    # for up to 121 RGB frames.
    max_num_video_latent_frames: int = 31
    vit_max_num_patch_per_side: int = 70
    connector_act: str = "gelu_pytorch_tanh"
    timestep_shift: float = 3.5
    num_timesteps: int = 30
    cfg_text_scale: float = 4.0
    # Wan2.2 VAE
    vae_z_channels: int = 48
    vae_downsample_spatial: int = 16
    vae_downsample_temporal: int = 4


LANCE_DEFAULTS = LanceDefaults()

# Subdirectories inside the HF repo (see config.json::checkpoint_directories).
_IMAGE_CKPT_DIR = "Lance_3B"
_VIDEO_CKPT_DIR = "Lance_3B_Video"
_VIT_DIR = "Qwen2.5-VL-ViT"
_VAE_FILE = "Wan2.2_VAE.pth"


def get_lance_post_process_func(od_config: OmniDiffusionConfig):
    """Lance returns PIL.Image.Image directly, same as BAGEL."""

    def post_process_func(x):
        return x

    return post_process_func


def get_lance_pre_process_func(od_config: OmniDiffusionConfig):
    def pre_process_func(x):
        return x

    return pre_process_func


@dataclass
class _LanceVaeCfg:
    z_channels: int = LANCE_DEFAULTS.vae_z_channels
    # BAGEL core computes ``latent_downsample = vae.downsample * latent_patch_size``.
    # For Lance image latents that is ``16 * 2 = 32``.
    downsample: int = LANCE_DEFAULTS.vae_downsample_spatial


@dataclass
class _LanceVitCfg:
    patch_size: int = 14
    hidden_size: int = 1280  # Qwen2.5-VL vision hidden (out_hidden_size=2048)


class LancePipeline(BagelPipeline):
    """Lance pipeline.  Inherits BAGEL's forward/generation; overrides only
    construction (checkpoint layout, Qwen2.5-VL ViT, Wan2.2 VAE)."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        # Intentionally do NOT call BagelPipeline.__init__ — its assumptions
        # about config.json / vit_config.json / SigLIP / the BAGEL AE do not
        # hold for Lance.  We replicate the BAGEL construction sequence with
        # Lance-specific component builders, then reuse every inherited method.
        import torch.nn as nn
        from vllm.transformers_utils.configs.bagel import BagelConfig

        from vllm_omni.diffusion.distributed.utils import get_local_device
        from vllm_omni.diffusion.model_loader.diffusers_loader import (
            DiffusersPipelineLoader,
        )

        nn.Module.__init__(self)
        self.od_config = od_config
        self.device = get_local_device()
        self.scheduler = None
        self.scheduler_kwargs = {}

        model = od_config.model
        if os.path.exists(model):
            repo_root = model
        else:
            repo_root = download_weights_from_hf_specific(model, od_config.revision, ["*"])

        # If --model points directly at ``Lance_3B`` / ``Lance_3B_Video`` (or
        # ``Qwen2.5-VL-ViT``), walk up so ``repo_root`` is the bundled top-level
        # dir that owns the sibling components (VAE, ViT).
        base = os.path.basename(repo_root.rstrip("/"))
        if base in {_IMAGE_CKPT_DIR, _VIDEO_CKPT_DIR, _VIT_DIR}:
            parent = os.path.dirname(repo_root.rstrip("/"))
            if os.path.isdir(parent) and os.path.isfile(os.path.join(parent, _VAE_FILE)):
                repo_root = parent

        is_video = self._select_video_variant(od_config) or base == _VIDEO_CKPT_DIR
        ckpt_dir = _VIDEO_CKPT_DIR if is_video else _IMAGE_CKPT_DIR
        ckpt_path = os.path.join(repo_root, ckpt_dir)
        if not os.path.isdir(ckpt_path):
            # Some users point --model directly at a single checkpoint dir.
            ckpt_path = repo_root
        self._repo_root = repo_root
        self._ckpt_path = ckpt_path
        self._is_video = is_video
        if is_video:
            logger.info(
                "Lance video checkpoint selected (%s) — wired with "
                "LancePositionEmbedding3D and Wan2.2 multi-frame VAE decode "
                "(text-to-video). Image-edit / x2t_video remain follow-ups.",
                ckpt_dir,
            )

        # ---- LLM (Qwen2-MoT; identical weight layout to BAGEL) ----
        llm_cfg_path = os.path.join(ckpt_path, "llm_config.json")
        llm_config = Qwen2MoTConfig.from_json_file(llm_cfg_path)
        llm_config.qk_norm = True
        # The released Lance checkpoint ships a separate ``language_model.lm_head.weight``
        # tensor even though ``llm_config.json`` says ``tie_word_embeddings=True``; keep
        # the head untied so the checkpoint loads with zero missing/unexpected keys.
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = od_config.override_transformer_cls_name or "Qwen2MoTDecoderLayer"
        # Lance is Qwen2.5-VL-MoT and ships ``rope_scaling = {"type": "mrope",
        # "mrope_section": [16, 24, 24]}``.  Keep the mRoPE configuration:
        # :class:`BagelRotaryEmbedding` now auto-dispatches on ``rope_type`` and
        # consumes 3-D ``(t, h, w)`` position ids, while :class:`LanceBagel`
        # broadcasts scalar positions to ``(3, S)`` for text-only blocks and
        # emits true per-token ``(t, h, w)`` for video latent blocks.  Required
        # for the Qwen2.5-VL backbone to produce coherent x2t / t2v output.
        if not isinstance(getattr(llm_config, "rope_scaling", None), dict):
            llm_config.rope_scaling = {"rope_type": "mrope", "mrope_section": [16, 24, 24]}
        else:
            llm_config.rope_scaling.setdefault("rope_type", llm_config.rope_scaling.get("type", "mrope"))
            llm_config.rope_scaling.setdefault("mrope_section", [16, 24, 24])

        self.tokenizer = self._load_tokenizer(ckpt_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)
        # Image / video preprocessor for the understanding paths
        # (x2t_image / x2t_video / image_edit / video_edit).  Lance's HF repo
        # does not bundle a ``preprocessor_config.json``; we use transformers'
        # ``Qwen2VLImageProcessor`` / ``Qwen2VLVideoProcessor`` with their
        # default CLIP-style normalization, ``patch_size=14``, ``merge_size=2``,
        # which matches the bundled ``Qwen2.5-VL-ViT/config.json`` exactly.
        self.image_processor = self._build_image_processor()
        self.video_processor = self._build_video_processor()
        tok_len = len(self.tokenizer)
        required_max_id = max(int(v) for v in self.new_token_ids.values())
        llm_config.vocab_size = max(
            int(getattr(llm_config, "vocab_size", tok_len)),
            int(tok_len),
            int(required_max_id + 1),
        )

        parallel_config = od_config.parallel_config if od_config else None
        quant_config = od_config.quantization_config

        self.language_model = Qwen2MoTForCausalLM(
            llm_config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix="bagel.language_model",
        )
        self.transformer = self.language_model.model

        # ---- Understanding ViT: Qwen2.5-VL vision (bundled) ----
        self.vit_model = self._build_qwen2_5_vl_vit(repo_root)

        # ---- VAE: Wan2.2 (bundled .pth) ----
        self.vae = self._build_wan22_vae(repo_root)

        vae_cfg = _LanceVaeCfg()
        vit_cfg = _LanceVitCfg(
            patch_size=14,
            hidden_size=getattr(getattr(self.vit_model, "config", None), "hidden_size", 1280),
        )

        # Lance uses Qwen2.5-VL's vision tower whose ``merger`` already projects
        # to the LLM hidden size and which carries its own positional encoding;
        # there is no separate BAGEL-style ``connector`` / ``vit_pos_embed`` in
        # the released checkpoint (Lance_3B safetensors only contain
        # ``vit_model.*``, never ``connector.*`` / ``vit_pos_embed.*``).  We keep
        # BAGEL's ``visual_und=True`` so ``vit_model`` is registered as a child
        # of ``self.bagel`` and BAGEL's ``forward_cache_update_vit`` works
        # unchanged, then immediately replace ``connector`` with an identity and
        # ``vit_pos_embed`` with a zero op so the strict load check does not
        # demand those phantom weights and the addition in
        # ``forward_cache_update_vit`` is numerically a no-op.
        und_enabled = self.vit_model is not None
        self.bagel = LanceBagel(
            language_model=self.language_model,
            vit_model=self.vit_model,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix="bagel",
            config=BagelConfig(
                llm_config=llm_config,
                vae_config=vae_cfg,
                vit_config=vit_cfg,
                vit_max_num_patch_per_side=LANCE_DEFAULTS.vit_max_num_patch_per_side,
                connector_act=LANCE_DEFAULTS.connector_act,
                interpolate_pos=False,
                latent_patch_size=LANCE_DEFAULTS.latent_patch_size_spatial,
                max_latent_size=LANCE_DEFAULTS.max_latent_size,
                timestep_shift=LANCE_DEFAULTS.timestep_shift,
                visual_gen=True,
                visual_und=und_enabled,
            ),
        )
        if und_enabled:
            self.bagel.connector = LanceIdentityConnector()
            self.bagel.vit_pos_embed = LanceZeroVitPosEmbed()
            # Hand the Qwen2-VL processors to LanceBagel.prepare_vit_images /
            # prepare_vit_videos so they can re-call them and recover
            # ``image_grid_thw`` / ``video_grid_thw`` (BAGEL's lambda drops the
            # grid).
            self.bagel._lance_image_processor = self.image_processor
            self.bagel._lance_video_processor = self.video_processor

        # Lance_3B_Video ships a 3-D latent positional table
        # (``latent_pos_embed.pos_embed`` shape ``(max_num_frames * max_latent_size**2, hidden_size)``)
        # in place of BAGEL's 2-D ``PositionEmbedding(max_latent_size, hidden)``;
        # swap in :class:`LancePositionEmbedding3D` so the video checkpoint loads
        # cleanly.  ``max_num_frames`` is derived from the table size.
        if is_video:
            self.bagel.latent_pos_embed = LancePositionEmbedding3D(
                max_num_frames=LANCE_DEFAULTS.max_num_video_latent_frames,
                max_num_patch_per_side=LANCE_DEFAULTS.max_latent_size,
                hidden_size=llm_config.hidden_size,
            )

        # Weight sources.  Verified against bytedance-research/Lance:
        #   * Lance_3B/model.safetensors carries the LLM + connectors with
        #     keys ``language_model.* / vae2llm.* / llm2vae.* / time_embedder.* /
        #     latent_pos_embed.*`` — load under the ``bagel.`` namespace.
        #   * Lance_3B_Video/model.safetensors additionally includes
        #     ``vit_model.*`` (390 tensors).
        #   * Qwen2.5-VL-ViT/vit.safetensors (image checkpoint only) carries
        #     the bare ``blocks.* / merger.* / patch_embed.*`` and must land
        #     under ``bagel.vit_model.vision_model.*`` to match the
        #     :class:`LanceQwen2_5_VLNaViTWrapper` hierarchy.
        # Always use the resolved repo_root for weight sources so subfolder paths
        # resolve correctly even when the user passed a per-checkpoint subdir
        # (we already walked ``repo_root`` up to the bundled root above).
        weights_model = repo_root if os.path.isdir(repo_root) else od_config.model
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=weights_model,
                subfolder=ckpt_dir if ckpt_path != repo_root else None,
                revision=od_config.revision,
                prefix="bagel.",
                fall_back_to_pt=False,
            ),
        ]
        if und_enabled and not is_video and os.path.isdir(os.path.join(repo_root, _VIT_DIR)):
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=weights_model,
                    subfolder=_VIT_DIR,
                    revision=od_config.revision,
                    prefix="bagel.vit_model.vision_model.",
                    fall_back_to_pt=False,
                )
            )

        if quant_config is None and not (od_config.enable_layerwise_offload or od_config.parallel_config.use_hsdp):
            self.to(self.device)
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    # ------------------------------------------------------------------ #
    # Lance-specific component builders
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_video_variant(od_config: OmniDiffusionConfig) -> bool:
        """Pick Lance_3B vs Lance_3B_Video.  Defaults to image; a user can
        force video via the model path ending in the video dir or an
        od_config extra flag."""
        model = str(od_config.model)
        if model.rstrip("/").endswith(_VIDEO_CKPT_DIR):
            return True
        extra = getattr(od_config, "extra", None) or {}
        return bool(extra.get("lance_video", False))

    @staticmethod
    def _load_tokenizer(ckpt_path: str):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True, trust_remote_code=True)

    @staticmethod
    def _build_image_processor():
        """Construct a Qwen2.5-VL-compatible image preprocessor.

        Defaults mirror ``Qwen/Qwen2.5-VL-3B-Instruct`` (CLIP normalization,
        14-pixel patches, 2x spatial merge).
        """
        from transformers import Qwen2VLImageProcessor

        return Qwen2VLImageProcessor()

    @staticmethod
    def _build_video_processor():
        """Construct a Qwen2.5-VL-compatible video preprocessor.

        Used by the ``x2t_video`` understanding path.  Returns
        ``pixel_values_videos`` + 3-D ``video_grid_thw = [T_lat, H, W]``.
        """
        from transformers import Qwen2VLVideoProcessor

        return Qwen2VLVideoProcessor()

    def _build_qwen2_5_vl_vit(self, repo_root: str):
        """Build the bundled Qwen2.5-VL vision tower and wrap it NaViT-style.

        Lance's bundled ``Qwen2.5-VL-ViT/config.json`` sets
        ``_attn_implementation = "flash_attention_2"``; we force ``sdpa``
        instead so the tower constructs on hardware (e.g. Blackwell) where
        ``flash-attn`` is not present, falling back to PyTorch's native
        scaled-dot-product attention.  Numerically equivalent for inference.
        """
        vit_dir = os.path.join(repo_root, _VIT_DIR)
        cfg_path = os.path.join(vit_dir, "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            vit_cfg_dict = json.load(f)
        vit_cfg_dict = dict(vit_cfg_dict)
        vit_cfg_dict["_attn_implementation"] = "sdpa"
        try:
            from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
                Qwen2_5_VLVisionConfig,
            )
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VisionTransformerPretrainedModel,
            )

            vit_conf = Qwen2_5_VLVisionConfig(**vit_cfg_dict)
            vision = Qwen2_5_VisionTransformerPretrainedModel(vit_conf)
        except Exception as e:
            logger.warning(
                "Could not instantiate Qwen2.5-VL vision tower (%s). "
                "Understanding (x2t) path will be unavailable; generation (t2i) "
                "does not use the ViT.",
                e,
            )
            return None
        return LanceQwen2_5_VLNaViTWrapper(vision, spatial_merge_size=int(vit_cfg_dict.get("spatial_merge_size", 2)))

    # ------------------------------------------------------------------ #
    # Lance text-to-video forward path
    # ------------------------------------------------------------------ #
    def forward(self, req):  # type: ignore[override]
        """Dispatch on prompt modality.

        - ``modalities == ["video"]`` (text-to-video) → :meth:`_forward_t2v`
          (3-D latents + ``LanceWanVAE.decode_video``).
        - ``modalities == ["text"]`` + ``multi_modal_data.video`` (x2t_video) →
          :meth:`_forward_x2t_video` (multi-frame Qwen2.5-VL ViT prefill).
        - ``modalities == ["image"]`` + ``multi_modal_data.img2img`` (image_edit)
          → :meth:`_forward_image_edit` (Lance-native VAE+ViT prefill + image gen).
        - ``modalities == ["video"]`` + ``multi_modal_data.video`` (video_edit)
          → :meth:`_forward_video_edit` (Lance-native multi-frame VAE+ViT prefill
          + video gen).
        - Everything else falls through to :meth:`BagelPipeline.forward`
          (t2i, x2t_image).
        """
        first_prompt = req.prompts[0] if req.prompts else None
        modalities: list[str] = []
        mm_data: dict = {}
        if isinstance(first_prompt, dict):
            modalities = first_prompt.get("modalities") or []
            mm_data = first_prompt.get("multi_modal_data") or {}
        if "video" in modalities and mm_data.get("video") is not None:
            return self._forward_video_edit(req)
        if "video" in modalities:
            return self._forward_t2v(req)
        if "text" in modalities and mm_data.get("video") is not None:
            return self._forward_x2t_video(req)
        if "image" in modalities and (mm_data.get("img2img") is not None or mm_data.get("image") is not None):
            return self._forward_image_edit(req)
        return super().forward(req)

    @torch.inference_mode()
    def _forward_t2v(self, req):
        """Minimal text-to-video forward: text prefill + 3-D latent denoising +
        Wan2.2 multi-frame decode.  Mirrors :meth:`BagelPipeline.forward`'s t2i
        branch but with 3-D latents (no image input, no x2t)."""
        from copy import deepcopy

        from vllm_omni.diffusion.data import DiffusionOutput

        from .lance_transformer import NaiveCache

        first_prompt = req.prompts[0]
        if isinstance(first_prompt, dict):
            prompt = first_prompt.get("prompt") or ""
            extra_args = first_prompt.get("extra_args") or {}
        else:
            prompt = str(first_prompt)
            extra_args = {}
        # Sampling-side extras override prompt-side.
        sp_extra = getattr(req.sampling_params, "extra_args", {}) or {}
        extra_args = {**extra_args, **sp_extra}

        # Video shape.  T = number of RGB frames (1..121), H/W in pixels.
        T = int(extra_args.get("num_frames", 25))
        H = int(req.sampling_params.height or extra_args.get("video_height", 480))
        W = int(req.sampling_params.width or extra_args.get("video_width", 768))
        max_lat = self.bagel.max_latent_size
        max_hw = max_lat * self.bagel.latent_downsample
        if H > max_hw or W > max_hw:
            raise ValueError(f"Requested video resolution {H}x{W} exceeds Lance limit {max_hw}x{max_hw}")
        downsample_t = int(getattr(self.bagel.config.vae_config, "downsample_temporal", 4))
        max_T_lat = LANCE_DEFAULTS.max_num_video_latent_frames
        if (T - 1) // downsample_t + 1 > max_T_lat:
            raise ValueError(
                f"Requested num_frames={T} exceeds Lance video temporal limit "
                f"{(max_T_lat - 1) * downsample_t + 1} (max_num_video_latent_frames={max_T_lat})"
            )
        video_shape = (T, H, W)
        logger.info("Lance t2v: video_shape=%s", video_shape)

        cfg_text_scale = float(extra_args.get("cfg_text_scale", 4.0))
        timestep_shift = float(extra_args.get("timestep_shift", LANCE_DEFAULTS.timestep_shift))
        num_timesteps = int(req.sampling_params.num_inference_steps or LANCE_DEFAULTS.num_timesteps)

        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        # ---- Build positive-prompt KV cache ----
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }
        cfg_text_context = deepcopy(gen_context)

        gen_input_text, newlens, new_rope = self.bagel.prepare_prompts(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            prompts=[prompt],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        for k, v in gen_input_text.items():
            if torch.is_tensor(v):
                gen_input_text[k] = v.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            gen_context["past_key_values"] = self.bagel.forward_cache_update_text(
                gen_context["past_key_values"], **gen_input_text
            )
        gen_context["kv_lens"] = newlens
        gen_context["ropes"] = new_rope

        # ---- Build CFG text-unconditional KV cache (empty prompt) ----
        if cfg_text_scale > 1.0:
            neg_prompt = str(extra_args.get("negative_prompt") or "")
            neg_input, neg_newlens, neg_rope = self.bagel.prepare_prompts(
                curr_kvlens=cfg_text_context["kv_lens"],
                curr_rope=cfg_text_context["ropes"],
                prompts=[neg_prompt],
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
            )
            for k, v in neg_input.items():
                if torch.is_tensor(v):
                    neg_input[k] = v.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            ):
                cfg_text_context["past_key_values"] = self.bagel.forward_cache_update_text(
                    cfg_text_context["past_key_values"], **neg_input
                )
            cfg_text_context["kv_lens"] = neg_newlens
            cfg_text_context["ropes"] = neg_rope

        # ---- 3-D latent init + CFG side metadata ----
        gen_input_lat = self.bagel.prepare_video_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            video_shapes=[video_shape],
            new_token_ids=self.new_token_ids,
        )
        for k, v in gen_input_lat.items():
            if torch.is_tensor(v):
                gen_input_lat[k] = v.to(self.device)
        cfg_text_lat = self.bagel.prepare_video_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=cfg_text_context["ropes"],
            video_shapes=[video_shape],
        )
        cfg_img_lat = self.bagel.prepare_video_latent_cfg(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            video_shapes=[video_shape],
        )
        for k, v in cfg_text_lat.items():
            if torch.is_tensor(v):
                cfg_text_lat[k] = v.to(self.device)
        for k, v in cfg_img_lat.items():
            if torch.is_tensor(v):
                cfg_img_lat[k] = v.to(self.device)

        # ---- Denoising loop (Bagel.generate_image is rank-agnostic over packed tokens) ----
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            latents, *_ = self.bagel.generate_image(
                past_key_values=gen_context["past_key_values"],
                cfg_text_past_key_values=cfg_text_context["past_key_values"],
                cfg_img_past_key_values=gen_context["past_key_values"],  # no img CFG branch
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=1.0,
                **gen_input_lat,
                cfg_text_packed_position_ids=cfg_text_lat["cfg_packed_position_ids"],
                cfg_text_packed_query_indexes=cfg_text_lat["cfg_packed_query_indexes"],
                cfg_text_key_values_lens=cfg_text_lat["cfg_key_values_lens"],
                cfg_text_packed_key_value_indexes=cfg_text_lat["cfg_packed_key_value_indexes"],
                cfg_img_packed_position_ids=cfg_img_lat["cfg_packed_position_ids"],
                cfg_img_packed_query_indexes=cfg_img_lat["cfg_packed_query_indexes"],
                cfg_img_key_values_lens=cfg_img_lat["cfg_key_values_lens"],
                cfg_img_packed_key_value_indexes=cfg_img_lat["cfg_packed_key_value_indexes"],
            )

        frames_np = self._decode_video_from_latent(self.bagel, self.vae, latents[0], video_shape)
        # Convert numpy frames to PIL.Image list for downstream serialization.
        frames = [Image.fromarray(f) for f in frames_np]
        logger.info("Lance t2v: decoded %d frames at %dx%d", len(frames), frames[0].width, frames[0].height)
        return DiffusionOutput(
            output=frames[0],
            custom_output={"video_frames": frames, "video_shape": video_shape},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    @torch.inference_mode()
    def _forward_image_edit(self, req):
        """Image edit (img2img): reference image + text prompt → modified image.

        Mirrors :meth:`_forward_t2v` for the text/VAE prefill structure but
        also runs ``_lance_native_prepare_vae_images`` for the reference
        image (so the LLM sees both the Wan2.2-encoded latents and the
        Qwen2.5-VL ViT context) and then a Lance image-gen loop emitting
        2-D latents for the new image.
        """

        import numpy as np

        from vllm_omni.diffusion.data import DiffusionOutput

        from .lance_transformer import NaiveCache

        first_prompt = req.prompts[0]
        assert isinstance(first_prompt, dict), "image_edit requires dict-style prompt"
        prompt = first_prompt.get("prompt") or ""
        mm_data = first_prompt.get("multi_modal_data") or {}
        image_input = mm_data.get("img2img") or mm_data.get("image")
        if image_input is None:
            raise ValueError("image_edit requires multi_modal_data.img2img (or .image).")
        if not isinstance(image_input, list):
            image_input = [image_input]
        image_input = [Image.open(im) if isinstance(im, str) else im for im in image_input]

        extra_args = first_prompt.get("extra_args") or {}
        sp_extra = getattr(req.sampling_params, "extra_args", {}) or {}
        extra_args = {**extra_args, **sp_extra}
        cfg_text_scale = float(extra_args.get("cfg_text_scale", 4.0))
        cfg_img_scale = float(extra_args.get("cfg_img_scale", 1.5))
        timestep_shift = float(extra_args.get("timestep_shift", LANCE_DEFAULTS.timestep_shift))
        num_timesteps = int(req.sampling_params.num_inference_steps or LANCE_DEFAULTS.num_timesteps)

        # Resize the reference image to a multiple of latent_downsample and
        # within the max latent grid.
        stride = self.bagel.latent_downsample
        max_hw = int(self.bagel.max_latent_size * stride)

        def _resize_to_stride(img):
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            scale = min(max_hw / max(w, h), 1.0)
            scale = max(scale, min(256, max_hw) / min(w, h))
            new_w = max(stride, int(round(w * scale / stride) * stride))
            new_h = max(stride, int(round(h * scale / stride) * stride))
            new_w = min(new_w, max_hw)
            new_h = min(new_h, max_hw)
            if new_w != w or new_h != h:
                img = img.resize((new_w, new_h), Image.BICUBIC)
            return img

        image_input = [_resize_to_stride(im) for im in image_input]
        resized_w, resized_h = image_input[0].size
        image_shape = (resized_h, resized_w)
        logger.info("Lance image_edit: ref image %dx%d", resized_w, resized_h)

        def vae_transforms(img):
            arr = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
            return arr.permute(2, 0, 1)  # (C, H, W)

        def vit_transforms(img):
            return self.image_processor(images=img, return_tensors="pt").pixel_values[0]

        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        # Match upstream Lance's edit position-id convention: input reference
        # VAE block and output gen latent share rope range ``[0, max_axis]``,
        # while text / ViT prefill are shifted by ``_EDIT_SHIFT`` so they
        # never collide.  Without this, the model sees the reference at
        # different rope coordinates than the gen target and effectively
        # regenerates instead of editing.
        _EDIT_SHIFT = 10000
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }

        # VAE prefill (Lance-native, 3-D mRoPE positions) — rope starts at 0
        # to align with the gen latent below.
        vae_input, newlens_vae, _ = self.bagel._lance_native_prepare_vae_images(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            images=image_input,
            transforms=vae_transforms,
            new_token_ids=self.new_token_ids,
            is_video=False,
        )
        for k, v in vae_input.items():
            if torch.is_tensor(v):
                vae_input[k] = v.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            gen_context["past_key_values"] = self.bagel.forward_cache_update_vae(
                self.vae, gen_context["past_key_values"], **vae_input
            )
        gen_context["kv_lens"] = newlens_vae
        # Shift rope out of the gen-latent range for the following text / ViT
        # prefills; we snap back to 0 when constructing the gen latent.
        gen_context["ropes"] = [_EDIT_SHIFT]

        # ViT prefill (Lance Qwen2.5-VL with packed flat patches).
        vit_input, newlens_vit, new_rope_vit = self.bagel.prepare_vit_images(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            images=image_input,
            transforms=vit_transforms,
            new_token_ids=self.new_token_ids,
        )
        for k, v in vit_input.items():
            if torch.is_tensor(v):
                vit_input[k] = v.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            gen_context["past_key_values"] = self.bagel.forward_cache_update_vit(
                gen_context["past_key_values"], **vit_input
            )
        gen_context["kv_lens"] = newlens_vit
        gen_context["ropes"] = new_rope_vit

        # Text prefill.
        text_input, newlens_text, new_rope_text = self.bagel.prepare_prompts(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            prompts=[prompt],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        for k, v in text_input.items():
            if torch.is_tensor(v):
                text_input[k] = v.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            gen_context["past_key_values"] = self.bagel.forward_cache_update_text(
                gen_context["past_key_values"], **text_input
            )
        gen_context["kv_lens"] = newlens_text
        gen_context["ropes"] = new_rope_text

        # CFG text branch (unconditional text).
        cfg_text_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }
        if cfg_text_scale > 1.0:
            neg_input, neg_newlens, neg_rope = self.bagel.prepare_prompts(
                curr_kvlens=cfg_text_context["kv_lens"],
                curr_rope=cfg_text_context["ropes"],
                prompts=[str(extra_args.get("negative_prompt") or "")],
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
            )
            for k, v in neg_input.items():
                if torch.is_tensor(v):
                    neg_input[k] = v.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            ):
                cfg_text_context["past_key_values"] = self.bagel.forward_cache_update_text(
                    cfg_text_context["past_key_values"], **neg_input
                )
            cfg_text_context["kv_lens"] = neg_newlens
            cfg_text_context["ropes"] = neg_rope

        # ---- 2-D latent init (image output) ----
        # Gen latent uses rope=[0] so its tokens share position coordinates
        # with the reference image's VAE block above (edit alignment).
        gen_input_lat = self.bagel.prepare_vae_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=[0],
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )
        for k, v in gen_input_lat.items():
            if torch.is_tensor(v):
                gen_input_lat[k] = v.to(self.device)
        cfg_text_lat = self.bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=[0],
            image_sizes=[image_shape],
        )
        cfg_img_lat = self.bagel.prepare_vae_latent_cfg(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=[0],
            image_sizes=[image_shape],
        )
        for k, v in cfg_text_lat.items():
            if torch.is_tensor(v):
                cfg_text_lat[k] = v.to(self.device)
        for k, v in cfg_img_lat.items():
            if torch.is_tensor(v):
                cfg_img_lat[k] = v.to(self.device)

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            latents, *_ = self.bagel.generate_image(
                past_key_values=gen_context["past_key_values"],
                cfg_text_past_key_values=cfg_text_context["past_key_values"],
                cfg_img_past_key_values=gen_context["past_key_values"],
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                **gen_input_lat,
                cfg_text_packed_position_ids=cfg_text_lat["cfg_packed_position_ids"],
                cfg_text_packed_query_indexes=cfg_text_lat["cfg_packed_query_indexes"],
                cfg_text_key_values_lens=cfg_text_lat["cfg_key_values_lens"],
                cfg_text_packed_key_value_indexes=cfg_text_lat["cfg_packed_key_value_indexes"],
                cfg_img_packed_position_ids=cfg_img_lat["cfg_packed_position_ids"],
                cfg_img_packed_query_indexes=cfg_img_lat["cfg_packed_query_indexes"],
                cfg_img_key_values_lens=cfg_img_lat["cfg_key_values_lens"],
                cfg_img_packed_key_value_indexes=cfg_img_lat["cfg_packed_key_value_indexes"],
            )

        img = self._decode_image_from_latent(self.bagel, self.vae, latents[0], image_shape)
        return DiffusionOutput(
            output=img,
            custom_output={"image_shape": image_shape},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    @torch.inference_mode()
    def _forward_video_edit(self, req):
        """Video edit: reference video + text prompt → modified video.

        Lance-native VAE prefill (multi-frame Wan2.2 encoding) + Qwen2.5-VL
        ViT prefill + 3-D latent denoising + ``LanceWanVAE.decode_video``.
        Mirrors :meth:`_forward_image_edit` but with the temporal axis
        threaded through both prefill and gen latent.
        """

        from vllm_omni.diffusion.data import DiffusionOutput

        from .lance_transformer import NaiveCache

        first_prompt = req.prompts[0]
        assert isinstance(first_prompt, dict), "video_edit requires dict-style prompt"
        prompt = first_prompt.get("prompt") or ""
        mm_data = first_prompt.get("multi_modal_data") or {}
        video_input = mm_data.get("video")
        if video_input is None:
            raise ValueError("video_edit requires multi_modal_data.video.")
        if isinstance(video_input, str):
            import imageio.v3 as iio

            video_input = iio.imread(video_input)
        # ``video_input`` now: ndarray (T, H, W, 3).  Permute to (C, T, H, W) for VAE.
        import numpy as _np

        if isinstance(video_input, _np.ndarray):
            video_chw = torch.from_numpy(video_input).float() / 127.5 - 1.0  # (T, H, W, 3)
            video_chw = video_chw.permute(3, 0, 1, 2).contiguous()  # (C, T, H, W)
        elif torch.is_tensor(video_input):
            video_chw = video_input.float()
        else:
            raise ValueError(f"Unsupported video_input type {type(video_input)}")
        T = int(video_chw.shape[1])
        H = int(video_chw.shape[2])
        W = int(video_chw.shape[3])

        extra_args = first_prompt.get("extra_args") or {}
        sp_extra = getattr(req.sampling_params, "extra_args", {}) or {}
        extra_args = {**extra_args, **sp_extra}
        cfg_text_scale = float(extra_args.get("cfg_text_scale", 4.0))
        cfg_img_scale = float(extra_args.get("cfg_img_scale", 1.5))
        timestep_shift = float(extra_args.get("timestep_shift", LANCE_DEFAULTS.timestep_shift))
        num_timesteps = int(req.sampling_params.num_inference_steps or LANCE_DEFAULTS.num_timesteps)
        video_shape = (T, H, W)
        logger.info("Lance video_edit: video %dx%dx%d (T,H,W)", T, H, W)

        def vae_transforms(_):
            return video_chw

        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        # Position-id layout for video editing — matches upstream Lance's
        # ``shift_position_ids(pos_shift=1000)`` + ``modality1=modality2``
        # convention:
        #   * input video VAE block + output gen video latent share the same
        #     rope range [0, max(T_lat, H_lat, W_lat)] so the model can align
        #     "edit *this* token" between reference and output;
        #   * text + ViT prefill use a far-shifted rope start (``_EDIT_SHIFT``)
        #     so their positions never collide with the VAE/gen range.
        _EDIT_SHIFT = 10000
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }

        # Multi-frame VAE prefill — rope starts at 0 (aligned with gen latent below).
        vae_input, newlens_vae, _ = self.bagel._lance_native_prepare_vae_images(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            images=[video_chw],
            transforms=vae_transforms,
            new_token_ids=self.new_token_ids,
            is_video=True,
        )
        for k, v in vae_input.items():
            if torch.is_tensor(v):
                vae_input[k] = v.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            gen_context["past_key_values"] = self.bagel.forward_cache_update_vae(
                self.vae, gen_context["past_key_values"], **vae_input
            )
        gen_context["kv_lens"] = newlens_vae
        # Reset rope to the post-VAE shifted range for the text / ViT prefill;
        # we'll snap back to 0 when constructing the gen latent so it shares
        # positions with the VAE-prefilled reference.
        gen_context["ropes"] = [_EDIT_SHIFT]

        # Multi-frame ViT prefill (reuse x2t_video prep).
        vit_input, newlens_vit, new_rope_vit = self.bagel.prepare_vit_videos(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            videos=[video_input],
            new_token_ids=self.new_token_ids,
        )
        for k, v in vit_input.items():
            if torch.is_tensor(v):
                vit_input[k] = v.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            gen_context["past_key_values"] = self.bagel.forward_cache_update_vit(
                gen_context["past_key_values"], **vit_input
            )
        gen_context["kv_lens"] = newlens_vit
        gen_context["ropes"] = new_rope_vit

        # Text prefill.
        text_input, newlens_text, new_rope_text = self.bagel.prepare_prompts(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            prompts=[prompt],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        for k, v in text_input.items():
            if torch.is_tensor(v):
                text_input[k] = v.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            gen_context["past_key_values"] = self.bagel.forward_cache_update_text(
                gen_context["past_key_values"], **text_input
            )
        gen_context["kv_lens"] = newlens_text
        gen_context["ropes"] = new_rope_text

        # CFG text branch.
        cfg_text_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }
        if cfg_text_scale > 1.0:
            neg_input, neg_newlens, neg_rope = self.bagel.prepare_prompts(
                curr_kvlens=cfg_text_context["kv_lens"],
                curr_rope=cfg_text_context["ropes"],
                prompts=[str(extra_args.get("negative_prompt") or "")],
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
            )
            for k, v in neg_input.items():
                if torch.is_tensor(v):
                    neg_input[k] = v.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            ):
                cfg_text_context["past_key_values"] = self.bagel.forward_cache_update_text(
                    cfg_text_context["past_key_values"], **neg_input
                )
            cfg_text_context["kv_lens"] = neg_newlens
            cfg_text_context["ropes"] = neg_rope

        # 3-D latent init (video output).  ``curr_rope=[0]`` so the gen latent
        # tokens share positions with the input VAE block above — that's what
        # tells the model "edit token (t,h,w) in the input is token (t,h,w) in
        # the output".  The kv-cache still contains the text/ViT blocks at the
        # shifted rope range, so attending over them works.
        gen_input_lat = self.bagel.prepare_video_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=[0],
            video_shapes=[video_shape],
            new_token_ids=self.new_token_ids,
        )
        for k, v in gen_input_lat.items():
            if torch.is_tensor(v):
                gen_input_lat[k] = v.to(self.device)
        cfg_text_lat = self.bagel.prepare_video_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=[0],
            video_shapes=[video_shape],
        )
        cfg_img_lat = self.bagel.prepare_video_latent_cfg(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=[0],
            video_shapes=[video_shape],
        )
        for k, v in cfg_text_lat.items():
            if torch.is_tensor(v):
                cfg_text_lat[k] = v.to(self.device)
        for k, v in cfg_img_lat.items():
            if torch.is_tensor(v):
                cfg_img_lat[k] = v.to(self.device)

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            latents, *_ = self.bagel.generate_image(
                past_key_values=gen_context["past_key_values"],
                cfg_text_past_key_values=cfg_text_context["past_key_values"],
                cfg_img_past_key_values=gen_context["past_key_values"],
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                **gen_input_lat,
                cfg_text_packed_position_ids=cfg_text_lat["cfg_packed_position_ids"],
                cfg_text_packed_query_indexes=cfg_text_lat["cfg_packed_query_indexes"],
                cfg_text_key_values_lens=cfg_text_lat["cfg_key_values_lens"],
                cfg_text_packed_key_value_indexes=cfg_text_lat["cfg_packed_key_value_indexes"],
                cfg_img_packed_position_ids=cfg_img_lat["cfg_packed_position_ids"],
                cfg_img_packed_query_indexes=cfg_img_lat["cfg_packed_query_indexes"],
                cfg_img_key_values_lens=cfg_img_lat["cfg_key_values_lens"],
                cfg_img_packed_key_value_indexes=cfg_img_lat["cfg_packed_key_value_indexes"],
            )

        frames_np = self._decode_video_from_latent(self.bagel, self.vae, latents[0], video_shape)
        frames = [Image.fromarray(f) for f in frames_np]
        return DiffusionOutput(
            output=frames[0],
            custom_output={"video_frames": frames, "video_shape": video_shape},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    @torch.inference_mode()
    def _forward_x2t_video(self, req):
        """Video understanding (x2t_video): video → text caption / VQA answer.

        Mirrors :meth:`_forward_t2v` for the prefill / generation
        bookkeeping but routes the video through the Qwen2.5-VL ViT (no VAE
        prefill — Lance's training has Wan2.2 latents only for image_edit /
        video_edit, not for understanding).
        """

        from vllm_omni.diffusion.data import DiffusionOutput

        from .lance_transformer import NaiveCache

        first_prompt = req.prompts[0]
        assert isinstance(first_prompt, dict), "x2t_video requires dict-style prompt"
        prompt = first_prompt.get("prompt") or ""
        video_input = (first_prompt.get("multi_modal_data") or {}).get("video")
        if video_input is None:
            raise ValueError("x2t_video requires multi_modal_data.video to be a video tensor/array/path.")
        # Accept a path → load via imageio; else assume tensor / numpy array (T, H, W, 3).
        if isinstance(video_input, str):
            import imageio.v3 as iio

            video_input = iio.imread(video_input)
        # Wrap single video as a list for the prep function.
        videos = [video_input]

        extra_args = first_prompt.get("extra_args") or {}
        sp_extra = getattr(req.sampling_params, "extra_args", {}) or {}
        extra_args = {**extra_args, **sp_extra}
        max_text_tokens = int(extra_args.get("max_think_tokens", 200))
        do_sample = bool(extra_args.get("do_sample", False))
        text_temperature = float(extra_args.get("text_temperature", 0.3))

        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        # ---- Text-prompt prefill ----
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }
        if prompt:
            gen_input_text, newlens, new_rope = self.bagel.prepare_prompts(
                curr_kvlens=gen_context["kv_lens"],
                curr_rope=gen_context["ropes"],
                prompts=[prompt],
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
            )
            for k, v in gen_input_text.items():
                if torch.is_tensor(v):
                    gen_input_text[k] = v.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            ):
                gen_context["past_key_values"] = self.bagel.forward_cache_update_text(
                    gen_context["past_key_values"], **gen_input_text
                )
            gen_context["kv_lens"] = newlens
            gen_context["ropes"] = new_rope

        # ---- Multi-frame ViT prefill ----
        gen_input_vit, newlens_vit, new_rope_vit = self.bagel.prepare_vit_videos(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            videos=videos,
            new_token_ids=self.new_token_ids,
        )
        for k, v in gen_input_vit.items():
            if torch.is_tensor(v):
                gen_input_vit[k] = v.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            gen_context["past_key_values"] = self.bagel.forward_cache_update_vit(
                gen_context["past_key_values"], **gen_input_vit
            )
        gen_context["kv_lens"] = newlens_vit
        gen_context["ropes"] = new_rope_vit

        # ---- Text generation ----
        start_input = self.bagel.prepare_start_tokens(gen_context["kv_lens"], gen_context["ropes"], self.new_token_ids)
        for k, v in start_input.items():
            if torch.is_tensor(v):
                start_input[k] = v.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            token_ids = self.bagel.generate_text(
                past_key_values=gen_context["past_key_values"],
                max_length=max_text_tokens,
                do_sample=do_sample,
                temperature=text_temperature,
                end_token_id=self.new_token_ids["eos_token_id"],
                **start_input,
            )
        decoded = self.tokenizer.decode(token_ids[:, 0].tolist())
        text_output = decoded.split("<|im_end|>")[0]
        if "<|im_start|>" in text_output:
            text_output = text_output.split("<|im_start|>")[-1]
        logger.info("Lance x2t_video: generated %d tokens", token_ids.shape[0])
        return DiffusionOutput(
            output=text_output,
            custom_output={"text_output": text_output},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    @staticmethod
    def _decode_video_from_latent(
        bagel,
        vae: LanceWanVAE,
        latent: torch.Tensor,
        video_shape: tuple[int, int, int],
    ):
        """Pack a flat Lance latent into ``(B, 48, t, h, w)`` and decode to a
        ``(T_lat, H, W, 3)`` numpy frame list via :meth:`LanceWanVAE.decode_video`.
        """

        T, H, W = video_shape
        downsample_t = int(getattr(bagel.config.vae_config, "downsample_temporal", 4))
        t_lat = (T - 1) // downsample_t + 1
        h_lat = H // bagel.latent_downsample
        w_lat = W // bagel.latent_downsample
        p = bagel.latent_patch_size
        c = bagel.latent_channel
        # ``latent`` is flat ``(t_lat*h_lat*w_lat, c * p * p)`` (Lance image
        # path uses ``p=1`` so this is ``(N, c)``).  Reshape into the 5-D layout
        # the Wan2.2 VAE decoder consumes.
        latent = latent.reshape(t_lat, h_lat, w_lat, p, p, c)
        latent = torch.einsum("thwpqc->cthpwq", latent)
        latent = latent.reshape(1, c, t_lat, h_lat * p, w_lat * p)

        vae_dtype = next(vae.parameters()).dtype
        latent = latent.to(vae_dtype)
        video = vae.decode_video(latent)  # (1, 3, T_out, H_out, W_out)
        # Scale [-1,1] -> [0,255] uint8 frames; the upstream VAE already clamps.
        video = (video[0] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 3, 0) * 255
        return video.to(torch.uint8).cpu().numpy()  # (T_out, H_out, W_out, 3)

    def _build_wan22_vae(self, repo_root: str) -> LanceWanVAE:
        """Wrap the bundled ``Wan2.2_VAE.pth`` with a BAGEL-compatible
        ``.encode(images)`` / ``.decode(latent)`` surface.

        Uses the upstream Wan2.2 VAE module (ported in :mod:`wan_vae`) since
        the released ``.pth`` is keyed for that module — *not* for the diffusers
        ``AutoencoderKLWan`` state dict that ``OmniAutoencoderKLWan`` expects.
        Single images are treated as 1-frame clips; video latents pass through
        the same module via the 5-D ``encode_video``/``decode_video`` path.
        Wan2.2 VAE constants: z=48ch, /16 spatial, /4 temporal.
        """
        vae_path = os.path.join(repo_root, _VAE_FILE)
        return LanceWanVAE(vae_path=vae_path, dtype=torch.bfloat16, device=self.device)
