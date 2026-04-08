# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage wrappers for VoxCPM2 latent generator and AudioVAE V2.

Wraps the native VoxCPM2 API (generate / generate_streaming) for
vllm-omni's two-stage pipeline.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange


class VoxCPM2LatentGenerator:
    """Wraps VoxCPM2 model for latent generation (Stage 0)."""

    def __init__(self, model: Any):
        self._model = model
        self.sample_rate = int(getattr(model, "sample_rate", 48000))

    def generate_latents(
        self,
        *,
        text: str,
        reference_audio: str | None = None,
        prompt_audio: str | None = None,
        prompt_text: str | None = None,
        control_instruction: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
        min_length: int = 2,
        max_length: int = 4096,
    ) -> torch.Tensor:
        """Generate full latent sequence (sync mode)."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # VoxCPM2's generate() returns audio numpy array.
        # We call it and convert the result to a tensor.
        gen_kwargs: dict[str, Any] = {
            "text": text.strip(),
            "cfg_value": cfg_value,
            "num_inference_steps": inference_timesteps,
        }
        if reference_audio is not None:
            gen_kwargs["reference_audio"] = reference_audio
        if prompt_audio is not None:
            gen_kwargs["prompt_audio"] = prompt_audio
        if prompt_text is not None:
            gen_kwargs["prompt_text"] = prompt_text
        if control_instruction is not None:
            gen_kwargs["control_instruction"] = control_instruction

        result = self._model.generate(**gen_kwargs)
        if isinstance(result, torch.Tensor):
            return result.detach().cpu().to(torch.float32)
        return torch.as_tensor(result, dtype=torch.float32)

    def iter_latent_chunks_streaming(
        self,
        *,
        text: str,
        reference_audio: str | None = None,
        prompt_audio: str | None = None,
        prompt_text: str | None = None,
        control_instruction: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
        min_length: int = 2,
        max_length: int = 4096,
        streaming_prefix_len: int = 3,
    ) -> Generator[tuple[torch.Tensor, bool], None, None]:
        """Yield (latent_chunk, is_last) for async_chunk streaming."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        inner = getattr(self._model, "model", self._model)

        # Build prompt cache for voice cloning
        prompt_cache = None
        if prompt_audio and prompt_text:
            prompt_cache = inner.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=prompt_audio,
            )

        gen_kw: dict[str, Any] = {
            "target_text": text.strip(),
            "prompt_cache": prompt_cache,
            "min_len": min_length,
            "max_len": max_length,
            "inference_timesteps": inference_timesteps,
            "cfg_value": cfg_value,
            "streaming_prefix_len": streaming_prefix_len,
            "latents_only": True,
        }

        # Try streaming entry point
        stream_fn = getattr(inner, "generate_with_prompt_cache_streaming", None)
        if stream_fn is not None:
            gen = stream_fn(**gen_kw)
        else:
            gen = inner._generate_with_prompt_cache(streaming=True, **gen_kw)

        # Yield chunks with look-ahead for is_last detection
        iterator = iter(gen)
        previous = next(iterator, None)
        while previous is not None:
            current = next(iterator, None)
            _, _target_tok, chunk_latent = previous
            if not isinstance(chunk_latent, torch.Tensor):
                chunk_latent = torch.as_tensor(chunk_latent)
            yield chunk_latent, current is None
            previous = current


class VoxCPM2AudioVAE:
    """Wraps AudioVAE V2 for audio decoding (Stage 1). Outputs 48kHz."""

    def __init__(self, audio_vae: nn.Module, *, patch_size: int = 4):
        self.audio_vae = audio_vae
        self.sample_rate = int(getattr(audio_vae, "sample_rate", 48000))
        self.latent_dim = int(getattr(audio_vae, "latent_dim", 64))
        self.patch_size = int(patch_size)
        self._chunk_size = int(getattr(audio_vae, "chunk_size", 1))
        self._stream_audio_patch_samples = max(1, self.patch_size * self._chunk_size)

    def _prepare_latents(self, latent_audio_feat: Any) -> torch.Tensor:
        """Reshape latents to [1, latent_dim, T] for VAE decoder."""
        latents = latent_audio_feat
        if not isinstance(latents, torch.Tensor):
            latents = torch.tensor(latents, dtype=torch.float32)
        latents = latents.detach().to(torch.float32)

        if latents.ndim == 3:
            if latents.shape[-1] == self.latent_dim:
                latents = rearrange(latents, "t p d -> 1 d (t p)")
            elif latents.shape[1] == self.latent_dim:
                latents = latents.contiguous()
            else:
                raise ValueError(f"Unsupported latent shape: {tuple(latents.shape)}")
        elif latents.ndim == 2:
            if latents.shape[0] == self.latent_dim:
                latents = latents.unsqueeze(0)
            elif latents.shape[1] == self.latent_dim:
                latents = rearrange(latents, "t d -> 1 d t")
            else:
                raise ValueError(f"Unsupported latent shape: {tuple(latents.shape)}")
        else:
            raise ValueError(f"Unsupported latent ndim: {latents.ndim}")

        return latents

    @torch.no_grad()
    def decode(self, latent_audio_feat: Any, *, trim_streaming_patch: bool = False) -> torch.Tensor:
        """Decode latents to 48kHz audio waveform."""
        latents = self._prepare_latents(latent_audio_feat)
        device = next(self.audio_vae.parameters()).device
        raw = self.audio_vae.decode(latents.to(device=device, dtype=torch.float32))

        if isinstance(raw, dict):
            audio = raw.get("audio")
            if audio is None:
                audio = next(v for v in raw.values() if isinstance(v, torch.Tensor))
        else:
            audio = raw

        if audio.dim() == 3:
            stream = audio.squeeze(1)
        elif audio.dim() == 2:
            stream = audio
        else:
            stream = audio.reshape(audio.shape[0], -1)

        if trim_streaming_patch:
            stream = stream[..., -self._stream_audio_patch_samples:]

        return stream.reshape(-1).detach().cpu().to(torch.float32)
