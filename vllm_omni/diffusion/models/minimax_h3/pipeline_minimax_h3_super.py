# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fixed MiniMax H3 draft stage used by Super Acceleration."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError

from .lora import MINIMAX_H3_SUPER_TURBO_SCALE
from .pipeline_minimax_h3 import MINIMAX_H3_AUDIO_SAMPLE_RATE, MINIMAX_H3_FPS, MiniMaxH3Pipeline
from .taeh3 import TAEH3_CHECKPOINT_URL, TAEH3Decoder

H3_SUPER_DRAFT_HEIGHT = 512
H3_SUPER_DRAFT_WIDTH = 896
H3_SUPER_DEFAULT_DURATION = 5.0
H3_SUPER_SUPPORTED_DURATIONS = (5.0, 10.0)
H3_SUPER_REFINER_HEIGHT = 544
H3_SUPER_REFINER_WIDTH = 960
H3_SUPER_REFINER_FRAME_COUNTS = (121, 241)


def _resolve_refiner_frame_count(decoded_frames: int) -> int:
    for frame_count in reversed(H3_SUPER_REFINER_FRAME_COUNTS):
        if decoded_frames >= frame_count:
            return frame_count
    raise RuntimeError(f"MiniMax H3 Super draft must decode at least 121 frames, got {decoded_frames}")


def _prepare_h3_super_handoff(
    video: torch.Tensor,
    audio: torch.Tensor,
    *,
    target_height: int = H3_SUPER_REFINER_HEIGHT,
    target_width: int = H3_SUPER_REFINER_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the 2K-refiner BF16-video/FP32-PCM Stage 1 handoff on device."""
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise RuntimeError(f"MiniMax H3 Super video must be [1,3,T,H,W], got {tuple(video.shape)}")
    if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] not in (1, 2):
        raise RuntimeError(f"MiniMax H3 Super audio must be [1,C,S], got {tuple(audio.shape)}")

    frame_count = _resolve_refiner_frame_count(int(video.shape[2]))
    frames = video[:, :, :frame_count]
    frames_4d = frames.permute(0, 2, 1, 3, 4).flatten(0, 1)
    resized_4d = F.interpolate(
        frames_4d,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    prepared_video = (
        resized_4d.unflatten(0, (1, frame_count))
        .permute(0, 2, 1, 3, 4)
        .clamp(0.0, 1.0)
        .mul(2.0)
        .sub(1.0)
        .to(torch.bfloat16)
        .contiguous()
    )

    sample_count = int(frame_count / MINIMAX_H3_FPS * MINIMAX_H3_AUDIO_SAMPLE_RATE)
    prepared_audio = audio.float()
    if prepared_audio.shape[1] == 1:
        prepared_audio = prepared_audio.expand(-1, 2, -1)
    prepared_audio = prepared_audio[:, :, :sample_count]
    if prepared_audio.shape[-1] < sample_count:
        prepared_audio = F.pad(prepared_audio, (0, sample_count - prepared_audio.shape[-1]))
    prepared_audio = prepared_audio.clamp(-1.0, 1.0).contiguous()
    return prepared_video, prepared_audio


def _minimax_h3_super_post_process(output):
    """Keep compact inter-stage tensors intact instead of expanding to FP32 NumPy."""
    if not isinstance(output, tuple) or len(output) != 2:
        return output
    video, audio = output
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().contiguous()
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().contiguous()
    return {
        "video": video,
        "audio": audio,
        "audio_sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE,
        "fps": MINIMAX_H3_FPS,
    }


def get_minimax_h3_super_post_process_func(od_config):
    del od_config
    return _minimax_h3_super_post_process


class MiniMaxH3SuperDraftPipeline(MiniMaxH3Pipeline):
    """H3 Turbo stage with the released Super Acceleration draft contract."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        additional_config = getattr(self.od_config, "additional_config", {}) or {}
        checkpoint_source = additional_config.get("taeh3_checkpoint", TAEH3_CHECKPOINT_URL)
        self.taeh3_decoder = TAEH3Decoder.from_checkpoint(
            checkpoint_source,
            device=self.device,
        )

    @staticmethod
    def _resolve_super_duration(extra_args: dict) -> float:
        target = extra_args.get("target")
        target = target if isinstance(target, dict) else {}
        duration = target.get(
            "duration_seconds",
            extra_args.get("duration_seconds", extra_args.get("duration", H3_SUPER_DEFAULT_DURATION)),
        )
        try:
            duration = float(duration)
        except (TypeError, ValueError) as exc:
            raise OmniClientError("MiniMax H3 Super duration must be 5 or 10 seconds") from exc
        if not any(math.isclose(duration, supported) for supported in H3_SUPER_SUPPORTED_DURATIONS):
            raise OmniClientError("MiniMax H3 Super duration must be 5 or 10 seconds")
        return duration

    def decode(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video = self.taeh3_decoder.decode_video(video_latent)
        video = video[..., :height, :width].contiguous()
        with self._component_on_device(self.audio_vae):
            audio = self.audio_vae.decode_latent(audio_latent)
        return _prepare_h3_super_handoff(video, audio)

    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        sampling = request.sampling_params
        if not self._has_active_v01_turbo_lora(sampling):
            raise OmniClientError("MiniMax H3 Super requires the published 4-step Turbo v0.1 LoRA")
        if not math.isclose(float(sampling.lora_scale), MINIMAX_H3_SUPER_TURBO_SCALE):
            raise OmniClientError(f"MiniMax H3 Super requires lora_scale={MINIMAX_H3_SUPER_TURBO_SCALE:g}")

        extra_args = dict(sampling.extra_args or {})
        duration = self._resolve_super_duration(extra_args)
        extra_args.update(
            {
                "task": "fl2va",
                "duration": duration,
                "flow_shift": 12.0,
                "audio_flow_shift": 3.0,
            }
        )
        sampling.extra_args = extra_args
        sampling.height = H3_SUPER_DRAFT_HEIGHT
        sampling.width = H3_SUPER_DRAFT_WIDTH
        sampling.fps = 24
        sampling.frame_rate = 24.0
        sampling.num_inference_steps = 5
        return super().forward(request)


__all__ = [
    "H3_SUPER_DEFAULT_DURATION",
    "H3_SUPER_DRAFT_HEIGHT",
    "H3_SUPER_DRAFT_WIDTH",
    "H3_SUPER_REFINER_FRAME_COUNTS",
    "H3_SUPER_REFINER_HEIGHT",
    "H3_SUPER_REFINER_WIDTH",
    "H3_SUPER_SUPPORTED_DURATIONS",
    "MiniMaxH3SuperDraftPipeline",
    "get_minimax_h3_super_post_process_func",
]
