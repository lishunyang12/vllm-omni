# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native model loading for VoxCPM2.

VoxCPM2 uses `VoxCPM.from_pretrained()` instead of v1's `VoxCPMModel.from_local()`.
AudioVAE V2 outputs at 48kHz (vs v1's 24kHz).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from vllm.logger import init_logger

from .voxcpm2_import_utils import import_audio_vae_v2, import_voxcpm2_core
from .voxcpm2_stage_wrappers import VoxCPM2AudioVAE, VoxCPM2LatentGenerator

logger = init_logger(__name__)


def _load_voxcpm2_model(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
):
    """Load VoxCPM2 model via native from_pretrained API."""
    VoxCPM = import_voxcpm2_core()

    load_kwargs = {
        "load_denoiser": False,
    }
    if device is not None:
        load_kwargs["device"] = str(device)
    if dtype is not None:
        load_kwargs["dtype"] = dtype

    model = VoxCPM.from_pretrained(model_path, **load_kwargs)
    return model


def load_voxcpm2_latent_generator(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> VoxCPM2LatentGenerator:
    """Load latent generator (Stage 0)."""
    model = _load_voxcpm2_model(model_path, device=device, dtype=dtype)
    return VoxCPM2LatentGenerator(model)


def load_voxcpm2_audio_vae(
    model_path: str,
    *,
    device: torch.device,
) -> VoxCPM2AudioVAE:
    """Load AudioVAE V2 decoder (Stage 1).

    AudioVAE V2 outputs 48kHz audio (vs v1's 24kHz).
    """
    AudioVAE, AudioVAEConfig = import_audio_vae_v2()

    # Try loading config from model directory
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        config_dict = json.loads(config_path.read_text())
        audio_vae_config = config_dict.get("audio_vae_config")
        patch_size = int(config_dict.get("patch_size", 4))
    else:
        audio_vae_config = None
        patch_size = 4  # VoxCPM2 default

    if audio_vae_config is not None:
        audio_vae = AudioVAE(config=AudioVAEConfig(**audio_vae_config))
    else:
        audio_vae = AudioVAE()

    # Load weights
    vae_path = Path(model_path) / "audiovae.pth"
    if vae_path.exists():
        state_dict = torch.load(vae_path, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        audio_vae.load_state_dict(state_dict, strict=True)

    audio_vae = audio_vae.to(device=device, dtype=torch.float32).eval()
    return VoxCPM2AudioVAE(audio_vae, patch_size=patch_size)
