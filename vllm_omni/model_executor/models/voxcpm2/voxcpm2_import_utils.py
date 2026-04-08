# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic import utilities for VoxCPM2 native package.

Supports three discovery modes:
1. VLLM_OMNI_VOXCPM_CODE_PATH env var (explicit source tree)
2. Sibling directory: ../VoxCPM/src
3. pip-installed voxcpm package (>= 2.0)
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _iter_voxcpm2_src_candidates() -> list[Path]:
    """Yield candidate source directories for VoxCPM2."""
    candidates: list[Path] = []
    env_path = os.environ.get("VLLM_OMNI_VOXCPM_CODE_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root.parent / "VoxCPM" / "src")

    # Deduplicate
    seen: set[str] = set()
    unique: list[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _prepend_src(candidate: Path) -> None:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _import_voxcpm2_attrs(module_name: str, *attr_names: str) -> tuple[Any, ...]:
    """Import attributes from voxcpm package, trying source tree first."""
    last_exc: ImportError | None = None

    # Try source tree candidates
    for candidate in _iter_voxcpm2_src_candidates():
        if not candidate.exists():
            continue
        _prepend_src(candidate)
        try:
            mod = importlib.import_module(module_name)
            return tuple(getattr(mod, name) for name in attr_names)
        except (ImportError, AttributeError) as exc:
            last_exc = ImportError(str(exc))
            continue

    # Fallback: pip-installed package
    try:
        mod = importlib.import_module(module_name)
        return tuple(getattr(mod, name) for name in attr_names)
    except (ImportError, AttributeError) as exc:
        last_exc = ImportError(str(exc))

    raise ImportError(
        f"Could not import {attr_names} from {module_name}. "
        f"Install voxcpm>=2.0: pip install voxcpm. "
        f"Or set VLLM_OMNI_VOXCPM_CODE_PATH to the VoxCPM source tree. "
        f"Last error: {last_exc}"
    )


def import_voxcpm2_core():
    """Import VoxCPM core class for model loading."""
    (VoxCPM,) = _import_voxcpm2_attrs("voxcpm.core", "VoxCPM")
    return VoxCPM


def import_voxcpm2_model():
    """Import VoxCPM2Model for subclassing."""
    (VoxCPM2Model,) = _import_voxcpm2_attrs("voxcpm.model.voxcpm2", "VoxCPM2Model")
    return VoxCPM2Model


def import_audio_vae_v2():
    """Import AudioVAE V2 for separate VAE stage loading."""
    AudioVAE, AudioVAEConfig = _import_voxcpm2_attrs(
        "voxcpm.modules.audiovae", "AudioVAE", "AudioVAEConfig"
    )
    return AudioVAE, AudioVAEConfig


def make_voxcpm2_model_for_omni(base_cls):
    """Subclass VoxCPM2Model to add latents_only support.

    Overrides _generate_with_prompt_cache to skip audio_vae.decode()
    when latents_only=True, returning raw latent features instead.
    Same pattern as VoxCPM v1's VoxCPMModelForOmni.
    """
    import torch
    from typing import Generator, Tuple, Union, List

    class VoxCPM2ModelForOmni(base_cls):
        @torch.inference_mode()
        def _generate_with_prompt_cache(
            self,
            target_text,
            prompt_cache,
            min_len=2,
            max_len=2000,
            inference_timesteps=10,
            cfg_value=2.0,
            retry_badcase=False,
            retry_badcase_max_times=3,
            retry_badcase_ratio_threshold=6.0,
            streaming=False,
            streaming_prefix_len=4,
            latents_only=False,
        ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
            """Same as parent but with latents_only option to skip VAE decode."""
            if not latents_only:
                # Delegate to parent for normal (decoded audio) path
                yield from super()._generate_with_prompt_cache(
                    target_text=target_text,
                    prompt_cache=prompt_cache,
                    min_len=min_len,
                    max_len=max_len,
                    inference_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=retry_badcase_max_times,
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                    streaming=streaming,
                    streaming_prefix_len=streaming_prefix_len,
                )
                return

            # --- latents_only path: replicate parent logic but skip audio_vae.decode ---
            # Build inputs (same as parent)
            if prompt_cache is None:
                prompt_audio_feat = torch.empty(
                    (0, self.patch_size, self.audio_vae.latent_dim),
                    dtype=torch.float32,
                )
                text = target_text
            else:
                prompt_audio_feat = prompt_cache["audio_feat"]
                prompt_text = prompt_cache.get("prompt_text", "")
                text = prompt_text + target_text

            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat([
                text_token,
                torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
            ], dim=-1)

            target_text_token = torch.LongTensor(self.text_tokenizer(target_text))

            audio_length = prompt_audio_feat.size(0)
            text_length = text_token.shape[0]
            text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
            audio_pad_feat = torch.zeros(
                (text_token.shape[0], self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32, device=text_token.device,
            )
            text_token = torch.cat([text_token, text_pad_token])
            audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat], dim=0)
            text_mask = torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).int().to(text_token.device)
            audio_mask = torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).int().to(text_token.device)

            # Reference audio handling
            ref_audio_feat = None
            ref_text_token = None
            ref_text_mask = None
            ref_audio_mask = None
            if prompt_cache and "ref_audio_feat" in prompt_cache:
                ref_audio_feat = prompt_cache["ref_audio_feat"]
                ref_length = ref_audio_feat.size(0)
                ref_text_token = torch.zeros(ref_length, dtype=torch.int32, device=text_token.device)
                ref_text_mask = torch.zeros(ref_length, dtype=torch.int32, device=text_token.device)
                ref_audio_mask = torch.ones(ref_length, dtype=torch.int32, device=text_token.device)
                ref_audio_pad = torch.zeros(
                    (text_length, self.patch_size, self.audio_vae.latent_dim),
                    dtype=torch.float32, device=text_token.device,
                )
                ref_text_token = torch.cat([torch.zeros(text_length, dtype=torch.int32, device=text_token.device), ref_text_token])
                ref_audio_feat = torch.cat([ref_audio_pad, ref_audio_feat], dim=0)
                ref_text_mask = torch.cat([torch.zeros(text_length, dtype=torch.int32, device=text_token.device), ref_text_mask])
                ref_audio_mask = torch.cat([torch.zeros(text_length, dtype=torch.int32, device=text_token.device), ref_audio_mask])

            from voxcpm.model.utils import get_dtype
            text_token = text_token.unsqueeze(0).to(self.device)
            text_mask = text_mask.unsqueeze(0).to(self.device)
            audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
            audio_mask = audio_mask.unsqueeze(0).to(self.device)

            inference_result = self._inference(
                text_token, text_mask, audio_feat, audio_mask,
                min_len=min_len, max_len=max_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
            )

            if streaming:
                for latent_pred, pred_audio_feat, _ctx in inference_result:
                    # Skip VAE decode, yield raw latent
                    yield (None, target_text_token, latent_pred)
            else:
                target_text_length = len(self.text_tokenizer(target_text))
                retry_times = 0
                while retry_times < retry_badcase_max_times:
                    latent_pred, pred_audio_feat, context_len = next(inference_result)
                    if retry_badcase and pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        retry_times += 1
                        continue
                    break
                # Skip VAE decode, yield raw latent
                yield (None, target_text_token, pred_audio_feat)

    return VoxCPM2ModelForOmni
