# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ollin Boer Bohan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Decoder-only TAEH3 runtime adapted from madebyollin/taehv."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from torch import nn

TAEH3_UPSTREAM_COMMIT = "e589fddc076e77f5ba8cd6baabe4ba3260b261cd"
TAEH3_CHECKPOINT_URL = f"https://raw.githubusercontent.com/madebyollin/taehv/{TAEH3_UPSTREAM_COMMIT}/taeh3.pth"
TAEH3_CHECKPOINT_SHA256 = "af92965c2d7986a89a757e7cccd26f9eeeff0c3f0d5495eb168aeb2d6d9be9ba"


def _conv(in_channels: int, out_channels: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value / 3) * 3


class _MemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _conv(in_channels * 2, out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat((value, past), dim=1)) + self.skip(value))


class _TGrow(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = value.shape
        return self.conv(value).reshape(-1, channels, height, width)


def _apply_decoder_parallel(decoder: nn.Sequential, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 5:
        raise ValueError(f"TAEH3 expects NTCHW latent input, got {tuple(value.shape)}")
    batch, frames, channels, height, width = value.shape
    value = value.reshape(batch * frames, channels, height, width)
    for block in decoder:
        if isinstance(block, _MemBlock):
            frames = value.shape[0] // batch
            shaped = value.reshape(batch, frames, value.shape[1], value.shape[2], value.shape[3])
            past = F.pad(shaped, (0, 0, 0, 0, 0, 0, 1, 0))[:, :frames]
            value = block(value, past.reshape_as(value))
        else:
            value = block(value)
    frames = value.shape[0] // batch
    return value.reshape(batch, frames, value.shape[1], value.shape[2], value.shape[3])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_taeh3_checkpoint(source: str | os.PathLike[str]) -> Path:
    raw_source = str(source)
    parsed = urlparse(raw_source)
    if parsed.scheme in {"http", "https"}:
        if raw_source != TAEH3_CHECKPOINT_URL:
            raise ValueError(f"unsupported remote TAEH3 checkpoint: {raw_source}")
        checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"taeh3-{TAEH3_CHECKPOINT_SHA256[:12]}.pth"
        if not checkpoint.is_file():
            temporary = checkpoint.with_name(f"{checkpoint.name}.{os.getpid()}.tmp")
            try:
                torch.hub.download_url_to_file(
                    raw_source,
                    temporary,
                    hash_prefix=TAEH3_CHECKPOINT_SHA256,
                    progress=True,
                )
                os.replace(temporary, checkpoint)
            finally:
                temporary.unlink(missing_ok=True)
    else:
        checkpoint = Path(source).expanduser()

    checkpoint = checkpoint.resolve(strict=True)
    actual_sha256 = _sha256_file(checkpoint)
    if actual_sha256 != TAEH3_CHECKPOINT_SHA256:
        raise RuntimeError(f"TAEH3 checkpoint SHA-256 {actual_sha256} != {TAEH3_CHECKPOINT_SHA256}: {checkpoint}")
    return checkpoint


class TAEH3Decoder(nn.Module):
    """Tiny decoder for normalized MiniMax H3 video latents."""

    latent_channels = 24
    patch_size = 2
    temporal_upscale = 4
    frames_to_trim = temporal_upscale - 1

    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            _Clamp(),
            _conv(24, 256),
            nn.ReLU(inplace=True),
            _MemBlock(256, 256),
            _MemBlock(256, 256),
            _MemBlock(256, 256),
            nn.Upsample(scale_factor=2),
            _TGrow(256, 1),
            _conv(256, 128, bias=False),
            _MemBlock(128, 128),
            _MemBlock(128, 128),
            _MemBlock(128, 128),
            nn.Upsample(scale_factor=2),
            _TGrow(128, 2),
            _conv(128, 64, bias=False),
            _MemBlock(64, 64),
            _MemBlock(64, 64),
            _MemBlock(64, 64),
            nn.Upsample(scale_factor=2),
            _TGrow(64, 2),
            _conv(64, 64, bias=False),
            nn.ReLU(inplace=True),
            _conv(64, 3 * self.patch_size**2),
        )

    @classmethod
    def from_checkpoint(
        cls,
        source: str | os.PathLike[str],
        *,
        device: torch.device,
    ) -> TAEH3Decoder:
        checkpoint = resolve_taeh3_checkpoint(source)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            raise TypeError(f"TAEH3 checkpoint must contain a state dict, got {type(state_dict).__name__}")
        decoder_state = {
            name.removeprefix("decoder."): tensor for name, tensor in state_dict.items() if name.startswith("decoder.")
        }
        decoder = cls()
        decoder.decoder.load_state_dict(decoder_state, strict=True)
        decoder.to(device=device, dtype=torch.float16)
        decoder.eval()
        decoder.requires_grad_(False)
        return decoder

    @torch.no_grad()
    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"TAEH3 expects NCTHW latent with {self.latent_channels} channels, got {tuple(latent.shape)}"
            )
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        value = latent.permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype).contiguous()
        value = _apply_decoder_parallel(self.decoder, value)

        chunk_frames = 5 * self.temporal_upscale
        pad_frames = (-value.shape[1]) % chunk_frames
        if pad_frames:
            value = F.pad(value, (0, 0, 0, 0, 0, 0, 0, pad_frames))
        if value.shape[1] < chunk_frames:
            raise ValueError(f"TAEH3 decoded timeline is too short: {value.shape[1]} frames")
        value = value.unflatten(1, (-1, chunk_frames))
        value = value[:, :, self.frames_to_trim :].flatten(1, 2)
        value = value[:, : -3 * self.temporal_upscale]

        batch, frames, channels, height, width = value.shape
        value = F.pixel_shuffle(
            value.reshape(batch * frames, channels, height, width),
            self.patch_size,
        )
        value = value.reshape(batch, frames, 3, value.shape[-2], value.shape[-1])
        return value.clamp_(0, 1).permute(0, 2, 1, 3, 4).contiguous()


__all__ = [
    "TAEH3_CHECKPOINT_SHA256",
    "TAEH3_CHECKPOINT_URL",
    "TAEH3_UPSTREAM_COMMIT",
    "TAEH3Decoder",
    "resolve_taeh3_checkpoint",
]
