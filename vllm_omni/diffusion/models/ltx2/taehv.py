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

"""Decoder-only wide TAEHV runtime adapted from madebyollin/taehv."""

from __future__ import annotations

import hashlib
import os
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from torch import nn

TAEHV_UPSTREAM_COMMIT = "32ac0146b11007cda5a57b60a3b35653361fb8a4"
TAEHV_CHECKPOINT_URL = f"https://raw.githubusercontent.com/madebyollin/taehv/{TAEHV_UPSTREAM_COMMIT}/taeltx2_3_wide.pth"
TAEHV_CHECKPOINT_SHA256 = "007788e6b9cb7f77e8589ae30ba7456b119d38b0d017e1d349c1c1d11e3d6339"


def _conv(in_channels: int, out_channels: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value / 3) * 3


class _WideMemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = max(1, out_channels // 64)
        if out_channels % groups:
            raise ValueError(f"TAEHV channels {out_channels} must be divisible by {groups}")
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels, groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels, groups=groups),
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


def _apply_decoder_sequential(decoder: nn.Sequential, value: torch.Tensor) -> torch.Tensor:
    """Run the reviewed O(1)-timeline-memory path used by the released pipeline."""
    if value.ndim != 5:
        raise ValueError(f"TAEHV expects NTCHW latent input, got {tuple(value.shape)}")
    work_queue = deque((frame, 0) for frame in value.unbind(1))
    memory: list[torch.Tensor | None] = [None] * len(decoder)
    output: list[torch.Tensor] = []

    while work_queue:
        frame, index = work_queue.popleft()
        if index == len(decoder):
            output.append(frame.unsqueeze(1))
            continue
        block = decoder[index]
        if isinstance(block, _WideMemBlock):
            past = memory[index]
            next_frame = block(frame, torch.zeros_like(frame) if past is None else past)
            memory[index] = frame
            work_queue.appendleft((next_frame, index + 1))
        elif isinstance(block, _TGrow):
            grown = block(frame)
            chunks = grown.view(grown.shape[0] // block.stride, -1, *grown.shape[-2:]).chunk(
                block.stride,
                dim=1,
            )
            for next_frame in reversed(chunks):
                work_queue.appendleft((next_frame, index + 1))
        else:
            work_queue.appendleft((block(frame), index + 1))

    return torch.cat(output, dim=1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_taehv_checkpoint(source: str | os.PathLike[str]) -> Path:
    raw_source = str(source)
    parsed = urlparse(raw_source)
    if parsed.scheme in {"http", "https"}:
        if raw_source != TAEHV_CHECKPOINT_URL:
            raise ValueError(f"unsupported remote TAEHV checkpoint: {raw_source}")
        checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"taeltx2_3_wide-{TAEHV_CHECKPOINT_SHA256[:12]}.pth"
        if not checkpoint.is_file():
            temporary = checkpoint.with_name(f"{checkpoint.name}.{os.getpid()}.tmp")
            try:
                torch.hub.download_url_to_file(
                    raw_source,
                    temporary,
                    hash_prefix=TAEHV_CHECKPOINT_SHA256,
                    progress=True,
                )
                os.replace(temporary, checkpoint)
            finally:
                temporary.unlink(missing_ok=True)
    else:
        checkpoint = Path(source).expanduser()

    checkpoint = checkpoint.resolve(strict=True)
    actual_sha256 = _sha256_file(checkpoint)
    if actual_sha256 != TAEHV_CHECKPOINT_SHA256:
        raise RuntimeError(f"TAEHV checkpoint SHA-256 {actual_sha256} != {TAEHV_CHECKPOINT_SHA256}: {checkpoint}")
    return checkpoint


class LTXWideTAEHVDecoder(nn.Module):
    """Wide tiny decoder for LTX-2.3/2.5 128-channel video latents."""

    latent_channels = 128
    patch_size = 4
    temporal_upscale = 8
    frames_to_trim = temporal_upscale - 1

    def __init__(self) -> None:
        super().__init__()
        channels = (1024, 512, 256, 64)
        self.decoder = nn.Sequential(
            _Clamp(),
            _conv(self.latent_channels, channels[0]),
            nn.ReLU(inplace=True),
            _WideMemBlock(channels[0], channels[0]),
            _WideMemBlock(channels[0], channels[0]),
            _WideMemBlock(channels[0], channels[0]),
            nn.Upsample(scale_factor=2),
            _TGrow(channels[0], 2),
            _conv(channels[0], channels[1], bias=False),
            _WideMemBlock(channels[1], channels[1]),
            _WideMemBlock(channels[1], channels[1]),
            _WideMemBlock(channels[1], channels[1]),
            nn.Upsample(scale_factor=2),
            _TGrow(channels[1], 2),
            _conv(channels[1], channels[2], bias=False),
            _WideMemBlock(channels[2], channels[2]),
            _WideMemBlock(channels[2], channels[2]),
            _WideMemBlock(channels[2], channels[2]),
            nn.Upsample(scale_factor=2),
            _TGrow(channels[2], 2),
            _conv(channels[2], channels[3], bias=False),
            nn.ReLU(inplace=True),
            _conv(channels[3], 3 * self.patch_size**2),
        )

    @classmethod
    def from_checkpoint(
        cls,
        source: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LTXWideTAEHVDecoder:
        checkpoint = resolve_taehv_checkpoint(source)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            raise TypeError(f"TAEHV checkpoint must contain a state dict, got {type(state_dict).__name__}")
        decoder = cls()
        decoder_state = {
            name.removeprefix("decoder."): tensor for name, tensor in state_dict.items() if name.startswith("decoder.")
        }
        expected_state = decoder.decoder.state_dict()
        for index, block in enumerate(decoder.decoder):
            if not isinstance(block, _TGrow):
                continue
            key = f"{index}.conv.weight"
            if decoder_state[key].shape[0] > expected_state[key].shape[0]:
                decoder_state[key] = decoder_state[key][-expected_state[key].shape[0] :]
        decoder.decoder.load_state_dict(decoder_state, strict=True)
        decoder.to(device=device, dtype=dtype)
        decoder.eval()
        decoder.requires_grad_(False)
        return decoder

    @torch.no_grad()
    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"TAEHV expects NCTHW latent with {self.latent_channels} channels, got {tuple(latent.shape)}"
            )
        parameter = next(self.parameters())
        value = latent.permute(0, 2, 1, 3, 4).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        value = _apply_decoder_sequential(self.decoder, value)
        value = value[:, self.frames_to_trim :]

        batch, frames, channels, height, width = value.shape
        value = F.pixel_shuffle(
            value.reshape(batch * frames, channels, height, width),
            self.patch_size,
        )
        return value.reshape(batch, frames, 3, value.shape[-2], value.shape[-1]).clamp_(0, 1)


__all__ = [
    "LTXWideTAEHVDecoder",
    "TAEHV_CHECKPOINT_SHA256",
    "TAEHV_CHECKPOINT_URL",
    "TAEHV_UPSTREAM_COMMIT",
    "resolve_taehv_checkpoint",
]
