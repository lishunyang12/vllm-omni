# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lance transformer pieces.

The Lance LLM is BAGEL's Qwen2-MoT transformer verbatim (identical
``*_moe_gen`` / ``q_norm`` / ``vae2llm`` / ``llm2vae`` / ``time_embedder`` /
``latent_pos_embed`` layout), so ``Bagel`` / ``Qwen2MoTForCausalLM`` /
``Qwen2MoTConfig`` / ``NaiveCache`` are re-exported unchanged. Lance-specific
pieces: :class:`LanceBagel` overrides the two ViT entry points to consume
Qwen2.5-VL's packed ``pixel_values`` + ``image_grid_thw`` layout (BAGEL
assumes SigLIP-style ``(C, H, W)`` tensors), and :class:`LancePositionEmbedding3D`
supplies the video 3-D latent position embedding. The x2t / video / edit
paths thread per-axis 3-D mRoPE positions; t2i uses scalar positions
(BAGEL-equivalent).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Lance LLM == BAGEL Qwen2-MoT: reuse verbatim (checkpoint weight names match).
from vllm_omni.diffusion.models.bagel.bagel_transformer import (  # noqa: F401
    Bagel,
    BaseNavitOutputWithPast,
    MLPconnector,
    NaiveCache,
    PackedAttentionMoT,
    PositionEmbedding,
    Qwen2MoTConfig,
    Qwen2MoTDecoderLayer,
    Qwen2MoTForCausalLM,
    Qwen2MoTModel,
    TimestepEmbedder,
    get_1d_sincos_pos_embed_from_grid,
    patchify,
)

__all__ = [
    "Bagel",
    "BaseNavitOutputWithPast",
    "LanceBagel",
    "LanceIdentityConnector",
    "LancePositionEmbedding3D",
    "LanceQwen2_5_VLNaViTWrapper",
    "LanceZeroVitPosEmbed",
    "MLPconnector",
    "NaiveCache",
    "PackedAttentionMoT",
    "PositionEmbedding",
    "Qwen2MoTConfig",
    "Qwen2MoTDecoderLayer",
    "Qwen2MoTForCausalLM",
    "Qwen2MoTModel",
    "TimestepEmbedder",
    "patchify",
]

# mRoPE temporal scaling constants for Lance (Qwen2.5-VL backbone).
# Upstream Lance computes vision rope positions as
# ``t_index = frame_idx * tokens_per_second * second_per_grid_t`` so adjacent
# latent frames sit at well-separated rope coordinates.  Lance bundles a
# ``Qwen2.5-VL-ViT/config.json`` with ``tokens_per_second: 2`` and uses
# ``second_per_grid_ts = 1.0`` in its inference script.
LANCE_TOKENS_PER_SECOND = 2
LANCE_SECONDS_PER_GRID = 1.0


def get_3d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """3-D sin-cos positional embedding (t, h, w), matching the upstream Lance
    ``modeling/lance/modeling_utils.py`` dimension split exactly."""
    assert embed_dim % 2 == 0, "Embedding dimension must be even for 3D embeddings"
    d = embed_dim // 3
    d = d if d % 2 == 0 else d - 1
    dim_t, dim_h = d, d
    dim_w = embed_dim - 2 * d
    assert dim_w % 2 == 0
    emb_t = get_1d_sincos_pos_embed_from_grid(dim_t, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, grid[2])
    return np.concatenate([emb_t, emb_h, emb_w], axis=1)


def get_3d_sincos_pos_embed(embed_dim: int, t: int, h: int, w: int) -> np.ndarray:
    grid_t = np.arange(t, dtype=np.float32)
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    tt, hh, ww = np.meshgrid(grid_t, grid_h, grid_w, indexing="ij")
    grid = np.stack([tt, hh, ww], axis=0)
    return get_3d_sincos_pos_embed_from_grid(embed_dim, grid)


class LancePositionEmbedding3D(nn.Module):
    """Frozen 3-D sin-cos latent position embedding for the video path.

    BAGEL only ships a 2-D ``PositionEmbedding`` (image latents).  Lance's
    ``Lance_3B_Video`` checkpoint adds a temporal axis; this mirrors upstream
    ``modeling/lance/modeling_utils.py::PositionEmbedding3D``.  The image path
    uses ``t=1`` and is numerically equivalent to the 2-D embedding.
    """

    def __init__(self, max_num_frames: int, max_num_patch_per_side: int, hidden_size: int):
        super().__init__()
        self.max_num_frames = max_num_frames
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        n = max_num_frames * max_num_patch_per_side * max_num_patch_per_side
        self.pos_embed = nn.Parameter(torch.zeros(n, hidden_size), requires_grad=False)
        self._init_weights()

    def _init_weights(self) -> None:
        pe = get_3d_sincos_pos_embed(
            self.hidden_size,
            self.max_num_frames,
            self.max_num_patch_per_side,
            self.max_num_patch_per_side,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pe).float())

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pos_embed[position_ids]


class LanceIdentityConnector(nn.Module):
    """No-op connector for Lance.

    BAGEL's ``connector`` projects the ViT hidden size to the LLM hidden size.
    Qwen2.5-VL's vision tower (which Lance uses) already projects to the LLM
    hidden size internally via ``merger`` (``out_hidden_size = hidden_size``),
    and the released Lance safetensors carry no ``connector.*`` weights.  We
    therefore plug in an ``Identity`` connector so ``forward_cache_update_vit``
    keeps its existing call site without a separate code path.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x


class LanceZeroVitPosEmbed(nn.Module):
    """No-op positional embedding for Lance's ViT tokens.

    BAGEL adds an extra 2-D sin-cos ``vit_pos_embed`` on top of the ViT output.
    Qwen2.5-VL's vision tower already carries its own (rotary) positional
    encoding, and the released Lance safetensors carry no
    ``vit_pos_embed.*`` weights.  This module returns a broadcast-friendly
    zero so the addition in ``forward_cache_update_vit`` is a no-op without
    requiring a code-path branch.
    """

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.zeros((), device=position_ids.device)


class LanceQwen2_5_VLNaViTWrapper(nn.Module):
    """Packed (NaViT-style) wrapper around the Qwen2.5-VL vision tower.

    Bridges BAGEL's ``vit(packed_pixel_values, ...) -> [num_tokens, vit_hidden]``
    surface to the HF ``Qwen2_5_VisionTransformerPretrainedModel`` which
    consumes ``(hidden_states, grid_thw)``.  The packed call additionally needs
    a per-image ``image_grid_thw`` so non-square images (and the
    spatial-merge token count) line up — :class:`LanceBagel` stashes the grid
    on the wrapper before invoking the ViT.
    """

    def __init__(self, vision_model: nn.Module, spatial_merge_size: int = 2):
        super().__init__()
        # Accept either the full Qwen2_5_VLForConditionalGeneration.visual or
        # the bare vision transformer.
        self.vision_model = getattr(vision_model, "visual", vision_model)
        self.spatial_merge_size = spatial_merge_size
        # Set by ``LanceBagel.forward_cache_update_vit`` before each call so
        # the wrapper can pass true per-image (T, H, W) to the HF ViT.
        self._pending_grid_thw: torch.Tensor | None = None

    @property
    def config(self):  # parity with SiglipNaViTWrapper.vision_model.config access
        return self.vision_model.config

    def set_pending_grid_thw(self, grid_thw: torch.Tensor) -> None:
        self._pending_grid_thw = grid_thw

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        if self._pending_grid_thw is not None:
            grid_thw_t = self._pending_grid_thw.to(packed_pixel_values.device)
            self._pending_grid_thw = None
        else:
            # Fallback for square images when grid_thw was not pre-stashed.
            cu = cu_seqlens.tolist()
            grid_thw = []
            for i in range(len(cu) - 1):
                n = cu[i + 1] - cu[i]
                side = int(round(float(n) ** 0.5))
                grid_thw.append([1, side, side])
            grid_thw_t = torch.tensor(grid_thw, dtype=torch.long, device=packed_pixel_values.device)

        hidden = packed_pixel_values
        if hasattr(self.vision_model, "dtype"):
            hidden = hidden.to(self.vision_model.dtype)
        out = self.vision_model(hidden_states=hidden, grid_thw=grid_thw_t)
        # Qwen2.5-VL returns ``BaseModelOutputWithPooling(last_hidden_state=...,
        # pooler_output=merged)``; the merged sequence (post spatial-merger,
        # shape ``[sum_tokens_after_merge, out_hidden_size]``) is what the LLM
        # consumes.
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if isinstance(out, tuple):
            return out[0]
        return out


class LanceBagel(Bagel):
    """Bagel subclass with Lance-specific ViT handling.

    The released Lance checkpoint pairs BAGEL's Qwen2-MoT trunk with the
    Qwen2.5-VL vision tower (whose ``merger`` already projects to LLM
    ``hidden_size`` and which carries its own rotary positional encoding).
    Two BAGEL assumptions therefore break for Lance:

    1.  ``prepare_vit_images`` calls ``transforms(image) -> (C, H, W)`` and
        then ``patchify(...)``; Lance's image processor returns
        ``(num_patches_flat, patch_features) + image_grid_thw`` already.
    2.  ``forward_cache_update_vit`` adds a ``connector`` projection plus a
        2-D ``vit_pos_embed`` that has no checkpoint weights and would
        also double-count the ViT's own positional encoding.

    This subclass overrides exactly those two methods.  Everything else —
    LLM trunk, VAE flow, generation loop, ``forward_cache_update_text`` /
    ``forward_cache_update_vae`` — is reused unchanged.
    """

    @staticmethod
    def _qwen_vl_processor_call(processor, image):
        """Invoke the Qwen2-VL image processor and return ``(pixel_values, grid_thw)``."""
        proc_out = processor(images=image, return_tensors="pt")
        pixel_values = proc_out["pixel_values"]
        grid_thw = proc_out["image_grid_thw"]
        return pixel_values, grid_thw

    # ------------------------------------------------------------------ #
    # 3-D mRoPE plumbing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mrope_broadcast(position_ids: torch.Tensor) -> torch.Tensor:
        """Convert scalar ``(S,)`` position ids to mrope ``(3, S)`` by
        broadcasting the scalar value to all three (t, h, w) axes.

        Lance/Qwen2.5-VL's mRoPE expects 3-D positions per token; for text
        tokens (and image-edit context tokens) all three axes share the
        same scalar position id.  Vision-generation paths override this with
        per-token ``(t, h, w)`` indices.
        """
        if position_ids is None:
            return position_ids
        if position_ids.ndim == 1:
            return position_ids.unsqueeze(0).expand(3, -1).contiguous()
        return position_ids

    def prepare_prompts(self, *args, **kwargs):
        gen_input, newlens, new_rope = super().prepare_prompts(*args, **kwargs)
        if "packed_text_position_ids" in gen_input:
            gen_input["packed_text_position_ids"] = self._mrope_broadcast(gen_input["packed_text_position_ids"])
        return gen_input, newlens, new_rope

    def prepare_vae_latent(self, *args, **kwargs):
        gen_input = super().prepare_vae_latent(*args, **kwargs)
        if "packed_position_ids" in gen_input:
            gen_input["packed_position_ids"] = self._mrope_broadcast(gen_input["packed_position_ids"])
        return gen_input

    def prepare_vae_latent_cfg(self, *args, **kwargs):
        gen_input = super().prepare_vae_latent_cfg(*args, **kwargs)
        if "cfg_packed_position_ids" in gen_input:
            gen_input["cfg_packed_position_ids"] = self._mrope_broadcast(gen_input["cfg_packed_position_ids"])
        return gen_input

    def prepare_start_tokens(self, *args, **kwargs):
        gen_input = super().prepare_start_tokens(*args, **kwargs)
        if isinstance(gen_input, dict):
            for k in (
                "packed_query_position_ids",
                "packed_text_position_ids",
                "packed_position_ids",
            ):
                if k in gen_input and torch.is_tensor(gen_input[k]):
                    gen_input[k] = self._mrope_broadcast(gen_input[k])
        return gen_input

    def prepare_vit_videos(self, curr_kvlens, curr_rope, videos, new_token_ids):
        """Multi-frame ViT prefill for the ``x2t_video`` understanding path.

        ``videos`` is a list of per-request video tensors / numpy arrays of
        shape ``(T, H, W, 3)``.  We call the Qwen2-VL **video** processor
        (stored on ``self._lance_video_processor``) which produces
        ``pixel_values_videos`` already in packed
        ``(num_patches_flat_3d, patch_features)`` layout plus a 3-D
        ``video_grid_thw`` (``[T_lat, H_patches, W_patches]``).  We forward the
        same pixel-values tensor to the Qwen2.5-VL ViT (wrapped by
        :class:`LanceQwen2_5_VLNaViTWrapper`), set the wrapper's pending
        grid, and place the post-merger tokens in the LLM sequence with
        proper 3-D mRoPE positions ``(P + t, P + h, P + w)`` over the
        ``T_lat × H_lat × W_lat`` post-merge grid.
        """
        processor = getattr(self, "_lance_video_processor", None)
        if processor is None:
            raise RuntimeError(
                "LanceBagel.prepare_vit_videos requires the pipeline to set "
                "``bagel._lance_video_processor`` (a Qwen2-VL-compatible video processor)."
            )

        packed_vit_token_indexes: list[int] = []
        vit_token_seqlens: list[int] = []
        packed_vit_tokens: list[torch.Tensor] = []
        packed_vit_position_ids: list[torch.Tensor] = []
        packed_text_ids: list[int] = []
        packed_text_indexes: list[int] = []
        packed_seqlens: list[int] = []
        packed_indexes: list[int] = []
        packed_key_value_indexes: list[int] = []
        grid_thw_list: list[torch.Tensor] = []
        per_axis_pos: list[tuple[int, int, int]] = []

        merge_size = int(getattr(self.vit_model, "spatial_merge_size", 2))
        merge_factor = merge_size * merge_size

        _curr = curr = 0
        newlens: list[int] = []
        new_rope: list[int] = []
        for video, curr_kvlen, curr_position_id in zip(videos, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            per_axis_pos.append((curr_position_id, curr_position_id, curr_position_id))

            proc_out = processor(videos=video, return_tensors="pt")
            # Qwen2-VL video processor: pixel_values_videos is the same packed
            # (num_patches_flat, patch_features) layout as the image processor
            # but with a temporal axis ``T_lat`` in video_grid_thw.
            pixel_values = proc_out["pixel_values_videos"]
            grid_thw = proc_out["video_grid_thw"]
            T_lat, H, W = (int(v) for v in grid_thw[0].tolist())
            num_patches_pre_merge = T_lat * H * W
            assert num_patches_pre_merge == pixel_values.shape[0], (
                f"pixel_values rows ({pixel_values.shape[0]}) != T_lat*H*W ({num_patches_pre_merge}); "
                f"video_grid_thw={grid_thw.tolist()}"
            )
            num_vit_tokens = num_patches_pre_merge // merge_factor

            packed_vit_tokens.append(pixel_values)
            vit_token_seqlens.append(num_patches_pre_merge)
            packed_vit_position_ids.append(torch.arange(num_patches_pre_merge, dtype=torch.long))
            grid_thw_list.append(grid_thw[0].to(torch.long))

            packed_vit_token_indexes.extend(range(_curr, _curr + num_vit_tokens))
            packed_indexes.extend(range(curr, curr + num_vit_tokens))
            curr += num_vit_tokens
            _curr += num_vit_tokens
            h_merged = H // merge_size
            w_merged = W // merge_size
            t_scale = LANCE_TOKENS_PER_SECOND * LANCE_SECONDS_PER_GRID
            for ti in range(T_lat):
                for hi in range(h_merged):
                    for wi in range(w_merged):
                        per_axis_pos.append(
                            (
                                curr_position_id + int(ti * t_scale),
                                curr_position_id + hi,
                                curr_position_id + wi,
                            )
                        )

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            end_p = curr_position_id + max(int((T_lat - 1) * t_scale), h_merged - 1, w_merged - 1) + 1
            per_axis_pos.append((end_p, end_p, end_p))

            packed_seqlens.append(num_vit_tokens + 2)
            newlens.append(curr_kvlen + num_vit_tokens + 2)
            new_rope.append(end_p + 1)

        pos_t = torch.tensor([p[0] for p in per_axis_pos], dtype=torch.long)
        pos_h = torch.tensor([p[1] for p in per_axis_pos], dtype=torch.long)
        pos_w = torch.tensor([p[2] for p in per_axis_pos], dtype=torch.long)
        packed_position_ids_3d = torch.stack([pos_t, pos_h, pos_w], dim=0)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": packed_position_ids_3d,
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "_lance_grid_thw": torch.stack(grid_thw_list, dim=0),
        }
        return generation_input, newlens, new_rope

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        # ``transforms`` is set up by ``BagelPipeline.forward`` to be
        # ``processor(images=img, return_tensors='pt').pixel_values[0]``, which
        # discards the ``image_grid_thw``.  For Lance we need both, so we ignore
        # the supplied lambda and re-call the processor here directly.  Lance's
        # pipeline always sets ``self.image_processor`` to ``Qwen2VLImageProcessor``.
        if not hasattr(self, "_lance_image_processor"):
            raise RuntimeError(
                "LanceBagel.prepare_vit_images requires the pipeline to set "
                "``bagel._lance_image_processor`` (a Qwen2-VL-compatible processor)."
            )
        processor = self._lance_image_processor

        packed_vit_token_indexes: list[int] = []
        vit_token_seqlens: list[int] = []
        packed_vit_tokens: list[torch.Tensor] = []
        packed_vit_position_ids: list[torch.Tensor] = []
        packed_text_ids: list[int] = []
        packed_text_indexes: list[int] = []
        packed_seqlens: list[int] = []
        packed_indexes: list[int] = []
        packed_key_value_indexes: list[int] = []
        grid_thw_list: list[torch.Tensor] = []

        merge_size = int(getattr(self.vit_model, "spatial_merge_size", 2))
        merge_factor = merge_size * merge_size

        _curr = curr = 0
        newlens: list[int] = []
        new_rope: list[int] = []
        # 3-D position ids per query token (mRoPE) — text framing tokens use
        # scalar broadcast, image patches use per-token ``(P + t, P + h_merged,
        # P + w_merged)`` so the Qwen2.5-VL backbone sees the spatial layout
        # along the (h, w) axes of mRoPE.
        per_axis_pos: list[tuple[int, int, int]] = []
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            per_axis_pos.append((curr_position_id, curr_position_id, curr_position_id))

            pixel_values, grid_thw = self._qwen_vl_processor_call(processor, image)
            # pixel_values: (num_patches_flat, patch_features). grid_thw: (1, 3) with (T, H, W) in patch units.
            T, H, W = (int(v) for v in grid_thw[0].tolist())
            num_patches_pre_merge = T * H * W
            assert num_patches_pre_merge == pixel_values.shape[0], (
                f"pixel_values rows ({pixel_values.shape[0]}) != T*H*W ({num_patches_pre_merge}); "
                f"image_grid_thw={grid_thw.tolist()}"
            )
            num_img_tokens = num_patches_pre_merge // merge_factor

            packed_vit_tokens.append(pixel_values)
            vit_token_seqlens.append(num_patches_pre_merge)
            packed_vit_position_ids.append(torch.arange(num_patches_pre_merge, dtype=torch.long))
            grid_thw_list.append(grid_thw[0].to(torch.long))

            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens
            # 3-D mRoPE positions for image patches: scan in row-major over the
            # post-merge grid (H/merge × W/merge).  Temporal axis stays at the
            # block's scalar position (single still image).
            h_merged = H // merge_size
            w_merged = W // merge_size
            for hi in range(h_merged):
                for wi in range(w_merged):
                    per_axis_pos.append((curr_position_id, curr_position_id + hi, curr_position_id + wi))

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            end_p = curr_position_id + max(h_merged - 1, w_merged - 1) + 1
            per_axis_pos.append((end_p, end_p, end_p))

            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(end_p + 1)

        pos_t = torch.tensor([p[0] for p in per_axis_pos], dtype=torch.long)
        pos_h = torch.tensor([p[1] for p in per_axis_pos], dtype=torch.long)
        pos_w = torch.tensor([p[2] for p in per_axis_pos], dtype=torch.long)
        packed_position_ids_3d = torch.stack([pos_t, pos_h, pos_w], dim=0)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": packed_position_ids_3d,
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            # Lance-extra: per-image (T, H, W) so the ViT wrapper can pass the
            # true grid to the HF tower (non-square images would otherwise be
            # reconstructed incorrectly as ``sqrt(n)`` squares).
            "_lance_grid_thw": torch.stack(grid_thw_list, dim=0),
        }
        return generation_input, newlens, new_rope

    def forward_cache_update_vit(self, *args, **kwargs):
        # Pluck the Lance-specific extra and stash on the wrapper for the
        # upcoming ViT call.  Everything else delegates to the parent so we
        # don't have to copy 60 lines of LLM-call boilerplate.
        grid_thw = kwargs.pop("_lance_grid_thw", None)
        if grid_thw is not None and hasattr(self.vit_model, "set_pending_grid_thw"):
            self.vit_model.set_pending_grid_thw(grid_thw)
        return super().forward_cache_update_vit(*args, **kwargs)

    def prepare_video_latent(self, curr_kvlens, curr_rope, video_shapes, new_token_ids):
        """3-D analogue of :meth:`prepare_vae_latent`.

        ``video_shapes`` is a list of ``(T, H, W)`` per request (RGB pixel
        space).  We package one packed-init-noise tensor over ``T_lat × H_lat ×
        W_lat`` latent tokens per video, plus 1-D indices into the 3-D position
        embedding table maintained by :class:`LancePositionEmbedding3D`
        (`bagel.latent_pos_embed`).  Latent geometry:

        - spatial: ``H_lat = H // latent_downsample`` (``=16`` for Lance)
        - temporal: ``T_lat = (T - 1) // downsample_temporal + 1`` (``=4`` for Wan2.2)
        - channels: ``latent_channel = 48``

        Position ids are flattened ``t * max_per_side² + h * max_per_side + w``
        so they index directly into the ``(max_num_frames * max_per_side²,
        hidden_size)`` table.
        """
        downsample_t = int(getattr(self.config.vae_config, "downsample_temporal", 4))
        max_per_side = int(self.max_latent_size)

        packed_text_ids, packed_text_indexes = [], []
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = [], [], []
        packed_seqlens, packed_indexes = [], []
        packed_key_value_indexes = []
        # 3-D position ids per token (mRoPE).  ``per_axis_pos[i]`` is the list
        # of (t, h, w) for the i-th query token; we stack into ``(3, S)`` at the
        # end.  Text framing tokens broadcast scalar to all axes; video latent
        # tokens carry per-token ``(P + t, P + h, P + w)``.
        per_axis_pos: list[tuple[int, int, int]] = []

        query_curr = curr = 0
        for (T, H, W), curr_kvlen, curr_position_id in zip(video_shapes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1
            per_axis_pos.append((curr_position_id, curr_position_id, curr_position_id))

            h = H // self.latent_downsample
            w = W // self.latent_downsample
            t = (T - 1) // downsample_t + 1
            num_video_tokens = t * h * w

            # 1-D position ids into the 3-D table (frame-major, then row, then col).
            tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
            tt_flat = tt.flatten().tolist()
            hh_flat = hh.flatten().tolist()
            ww_flat = ww.flatten().tolist()
            vae_position_ids = (tt * (max_per_side * max_per_side) + hh * max_per_side + ww).flatten()
            packed_vae_position_ids.append(vae_position_ids)

            packed_init_noises.append(torch.randn(num_video_tokens, self.latent_channel * self.latent_patch_size**2))

            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_video_tokens))
            packed_indexes.extend(range(curr, curr + num_video_tokens))
            curr += num_video_tokens
            query_curr += num_video_tokens
            # Per-token 3-D mRoPE positions, matching upstream Lance's
            # ``get_rope_index``: the temporal axis is amplified by
            # ``tokens_per_second * second_per_grid_t`` (see Qwen2.5-VL docs)
            # so neighbouring latent frames are well-separated in rope space.
            # ``second_per_grid_t = 1.0`` matches upstream's default;
            # ``tokens_per_second = 2`` is Lance's vit config.
            t_scale = LANCE_TOKENS_PER_SECOND * LANCE_SECONDS_PER_GRID
            for ti, hi, wi in zip(tt_flat, hh_flat, ww_flat):
                per_axis_pos.append(
                    (
                        curr_position_id + int(ti * t_scale),
                        curr_position_id + hi,
                        curr_position_id + wi,
                    )
                )

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1
            # end_of_image sits "past" the latent block; bump along all axes by
            # max of the latent block size so subsequent text resumes cleanly.
            end_p = curr_position_id + max(t - 1, h - 1, w - 1) + 1
            per_axis_pos.append((end_p, end_p, end_p))

            packed_seqlens.append(num_video_tokens + 2)

        # Stack into (3, total_query_tokens) for mRoPE.
        pos_t = torch.tensor([p[0] for p in per_axis_pos], dtype=torch.long)
        pos_h = torch.tensor([p[1] for p in per_axis_pos], dtype=torch.long)
        pos_w = torch.tensor([p[2] for p in per_axis_pos], dtype=torch.long)
        packed_position_ids_3d = torch.stack([pos_t, pos_h, pos_w], dim=0)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": packed_position_ids_3d,
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }
        return generation_input

    def prepare_video_latent_cfg(self, curr_kvlens, curr_rope, video_shapes):
        """3-D analogue of :meth:`prepare_vae_latent_cfg` (CFG side).

        Mirrors :meth:`prepare_video_latent`'s mRoPE 3-D position layout
        (text frame ⇒ scalar; video latent block ⇒ per-token ``(P+t, P+h, P+w)``).
        """
        downsample_t = int(getattr(self.config.vae_config, "downsample_temporal", 4))

        packed_indexes, packed_key_value_indexes = [], []
        per_axis_pos: list[tuple[int, int, int]] = []
        query_curr = curr = 0
        for (T, H, W), curr_kvlen, curr_position_id in zip(video_shapes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1
            per_axis_pos.append((curr_position_id, curr_position_id, curr_position_id))

            h = H // self.latent_downsample
            w = W // self.latent_downsample
            t = (T - 1) // downsample_t + 1
            num_video_tokens = t * h * w
            packed_indexes.extend(range(curr, curr + num_video_tokens))
            curr += num_video_tokens
            query_curr += num_video_tokens
            tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
            for ti, hi, wi in zip(tt.flatten().tolist(), hh.flatten().tolist(), ww.flatten().tolist()):
                per_axis_pos.append((curr_position_id + ti, curr_position_id + hi, curr_position_id + wi))

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1
            end_p = curr_position_id + max(t - 1, h - 1, w - 1) + 1
            per_axis_pos.append((end_p, end_p, end_p))

        pos_t = torch.tensor([p[0] for p in per_axis_pos], dtype=torch.long)
        pos_h = torch.tensor([p[1] for p in per_axis_pos], dtype=torch.long)
        pos_w = torch.tensor([p[2] for p in per_axis_pos], dtype=torch.long)
        cfg_packed_position_ids_3d = torch.stack([pos_t, pos_h, pos_w], dim=0)

        return {
            "cfg_packed_position_ids": cfg_packed_position_ids_3d,
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

    # ------------------------------------------------------------------ #
    # Image-edit / video-edit VAE prefill
    # ------------------------------------------------------------------ #
    def _lance_native_prepare_vae_images(
        self,
        curr_kvlens,
        curr_rope,
        images,
        transforms,
        new_token_ids,
        timestep=0,
        is_video: bool = False,
    ):
        """Lance-native ``prepare_vae_images`` with 3-D mRoPE positions.

        Mirrors :meth:`Bagel.prepare_vae_images` but emits per-token mRoPE
        ``(curr + t, curr + h, curr + w)`` positions for VAE latent tokens
        (BAGEL emits scalar ``curr_position_id`` for all of them, which
        triggers a CUDA index-out-of-bounds gather on Lance's Qwen2.5-VL
        backbone).  For still images ``t = 0``; for video clips ``t``
        ranges over ``T_lat = (T - 1) // downsample_temporal + 1``.
        """
        downsample_t = int(getattr(self.config.vae_config, "downsample_temporal", 4))
        max_per_side = int(self.max_latent_size)

        packed_text_ids: list[int] = []
        packed_text_indexes: list[int] = []
        packed_vae_position_ids: list[torch.Tensor] = []
        packed_vae_token_indexes: list[int] = []
        packed_indexes: list[int] = []
        packed_key_value_indexes: list[int] = []
        packed_seqlens: list[int] = []
        patchified_vae_latent_shapes: list[tuple] = []
        vae_image_tensors: list[torch.Tensor] = []
        per_axis_pos: list[tuple[int, int, int]] = []

        query_curr = curr = 0
        newlens: list[int] = []
        new_rope: list[int] = []
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1
            per_axis_pos.append((curr_position_id, curr_position_id, curr_position_id))

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            # ``image_tensor`` is ``(C, H, W)`` for image input or ``(C, T, H, W)``
            # for video input.  We derive the *latent* grid from the RGB spatial
            # dims and our Wan2.2 VAE downsample factor.
            if image_tensor.dim() == 3:
                H, W = image_tensor.shape[1:]
                T = 1
            elif image_tensor.dim() == 4:
                _, T, H, W = image_tensor.shape
            else:
                raise ValueError(f"vae transforms must return 3-D or 4-D tensor; got {image_tensor.shape}")
            h_lat = H // self.latent_downsample
            w_lat = W // self.latent_downsample
            t_lat = ((T - 1) // downsample_t + 1) if is_video else 1
            patchified_vae_latent_shapes.append((t_lat, h_lat, w_lat) if is_video else (h_lat, w_lat))

            # 1-D index into the (3-D when video) ``latent_pos_embed`` table.
            tt, hh, ww = torch.meshgrid(torch.arange(t_lat), torch.arange(h_lat), torch.arange(w_lat), indexing="ij")
            vae_pos_ids = (tt * (max_per_side * max_per_side) + hh * max_per_side + ww).flatten()
            packed_vae_position_ids.append(vae_pos_ids)

            num_image_tokens = t_lat * h_lat * w_lat
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens
            for ti, hi, wi in zip(tt.flatten().tolist(), hh.flatten().tolist(), ww.flatten().tolist()):
                per_axis_pos.append((curr_position_id + ti, curr_position_id + hi, curr_position_id + wi))

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1
            end_p = curr_position_id + max(t_lat - 1, h_lat - 1, w_lat - 1) + 1
            per_axis_pos.append((end_p, end_p, end_p))

            packed_seqlens.append(num_image_tokens + 2)
            newlens.append(curr_kvlen + num_image_tokens + 2)
            new_rope.append(end_p + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            if image_tensor.dim() == 3:
                padded_images[i, :, : image_tensor.shape[1], : image_tensor.shape[2]] = image_tensor
            else:
                padded_images[i, :, : image_tensor.shape[1], : image_tensor.shape[2], : image_tensor.shape[3]] = (
                    image_tensor
                )

        pos_t = torch.tensor([p[0] for p in per_axis_pos], dtype=torch.long)
        pos_h = torch.tensor([p[1] for p in per_axis_pos], dtype=torch.long)
        pos_w = torch.tensor([p[2] for p in per_axis_pos], dtype=torch.long)
        packed_position_ids_3d = torch.stack([pos_t, pos_h, pos_w], dim=0)

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": packed_position_ids_3d,
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
        return generation_input, newlens, new_rope

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        """No-op VAE prefill for the x2t / x2t_video path.

        Lance's understanding paths route image / video context through the
        Qwen2.5-VL ViT only.  BAGEL's pipeline runs ``prepare_vae_images``
        unconditionally for any image input, so this default override
        short-circuits it.  :meth:`_forward_image_edit` /
        :meth:`_forward_video_edit` call :meth:`_lance_native_prepare_vae_images`
        directly when they actually need Lance-style VAE prefill.
        """
        generation_input = {
            "padded_images": torch.empty(0, 3, 0, 0),
            "patchified_vae_latent_shapes": [],
            "packed_vae_position_ids": torch.empty(0, dtype=torch.long),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.empty(0, dtype=torch.long),
            "packed_text_ids": torch.empty(0, dtype=torch.long),
            "packed_text_indexes": torch.empty(0, dtype=torch.long),
            "packed_position_ids": torch.empty(0, dtype=torch.long),
            "packed_seqlens": torch.empty(0, dtype=torch.int),
            "packed_indexes": torch.empty(0, dtype=torch.long),
            "packed_key_value_indexes": torch.empty(0, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
        return generation_input, list(curr_kvlens), list(curr_rope)

    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values,
        padded_images=None,
        patchified_vae_latent_shapes=None,
        packed_vae_position_ids=None,
        packed_timesteps=None,
        packed_vae_token_indexes=None,
        packed_text_ids=None,
        packed_text_indexes=None,
        packed_position_ids=None,
        packed_seqlens=None,
        packed_indexes=None,
        key_values_lens=None,
        packed_key_value_indexes=None,
    ):
        """Lance-native VAE prefill that *actually* scatters the encoded
        latents into the LLM query sequence.

        :meth:`Bagel.forward_cache_update_vae` in vllm-omni computes
        ``packed_latent = vae2llm(...) + time_embed + pos_embed`` and then
        passes only ``packed_text_ids`` to the LLM — the VAE embeddings
        never enter the query sequence (the LLM builds it from
        ``embed_tokens(packed_text_ids)`` which is just the 2 framing
        tokens).  The mismatch between ``query_lens = [num_vae + 2]`` and
        the resulting 2-token sequence is what crashes the gather inside
        attention.

        We scatter both pieces explicitly: text framing tokens at
        ``packed_text_indexes`` and the VAE latent embeddings at
        ``packed_vae_token_indexes``, producing a full-length
        ``(sum(packed_seqlens), hidden)`` sequence the LLM can attend
        over.  Empty prep data (legacy x2t / x2t_video no-op path) is
        short-circuited.
        """
        if (
            packed_text_ids is None
            or packed_text_ids.numel() == 0
            or padded_images is None
            or padded_images.numel() == 0
        ):
            return past_key_values

        padded_latent = vae_model.encode(padded_images)
        p = self.latent_patch_size
        packed_latent_list = []
        for latent, shape in zip(padded_latent, patchified_vae_latent_shapes):
            if isinstance(shape, tuple) and len(shape) == 3:
                t_lat, h_lat, w_lat = shape
                latent = latent[:, :t_lat, : h_lat * p, : w_lat * p].reshape(
                    self.latent_channel, t_lat, h_lat, p, w_lat, p
                )
                latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * self.latent_channel)
            else:
                h_lat, w_lat = shape
                latent = latent[:, : h_lat * p, : w_lat * p].reshape(self.latent_channel, h_lat, p, w_lat, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent_list.append(latent)
        packed_latent = torch.cat(packed_latent_list, dim=0)

        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed

        # ---- Scatter text + VAE embeds into the full query sequence ---- #
        total_len = int(packed_seqlens.sum().item())
        packed_text_embed = self.language_model.model.embed_tokens(packed_text_ids)
        packed_query_sequence = packed_latent.new_zeros((total_len, packed_latent.shape[-1]))
        packed_query_sequence[packed_text_indexes] = packed_text_embed.to(packed_query_sequence.dtype)
        packed_query_sequence[packed_vae_token_indexes] = packed_latent.to(packed_query_sequence.dtype)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        output = self.language_model.forward(
            packed_query_sequence=packed_query_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        return output.past_key_values
