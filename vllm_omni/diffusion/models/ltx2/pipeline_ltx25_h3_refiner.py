# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""LTX-2.5 refinement stage for MiniMax H3 Super Acceleration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .ltx2_components import LTX25_TWO_STAGE_COMPONENT_PROFILE
from .ltx2_conditioning import LTXI2VConditioningMixin
from .ltx2_recipes import LTX25_H3_REFINER_RECIPE
from .ltx2_runtime import LTXRuntime
from .taehv import TAEHV_CHECKPOINT_URL, LTXWideTAEHVDecoder

H3_REFINER_DRAFT_HEIGHT = 384
H3_REFINER_DRAFT_WIDTH = 672
H3_REFINER_OUTPUT_HEIGHT = 768
H3_REFINER_OUTPUT_WIDTH = 1344
H3_REFINER_FPS = 24.0
H3_REFINER_AUDIO_SAMPLE_RATE = 32_000
H3_REFINER_AUDIO_VAE_SAMPLE_RATE = 16_000
H3_REFINER_FRAME_COUNTS = (121, 241)
H3_REFINER_VAE_TILE_FRAMES = 128
H3_REFINER_VAE_TILE_OVERLAP_FRAMES = 24
H3_REFINER_VAE_TILE_HEIGHT = 768
H3_REFINER_VAE_TILE_WIDTH = 768
H3_REFINER_VAE_TILE_OVERLAP_SPATIAL = 64


def _prompt_additional_information(prompt: Any) -> dict[str, Any]:
    if not isinstance(prompt, dict):
        return {}
    value = prompt.get("additional_information")
    return value if isinstance(value, dict) else {}


def _as_video_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    video = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    if video.ndim == 4:
        if video.shape[-1] in (1, 3, 4):
            video = video.permute(3, 0, 1, 2).unsqueeze(0)
        elif video.shape[0] in (1, 3, 4):
            video = video.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported H3 video shape: {tuple(video.shape)}")
    elif video.ndim == 5 and video.shape[-1] in (1, 3, 4):
        video = video.permute(0, 4, 1, 2, 3)
    if video.ndim != 5 or video.shape[1] not in (1, 3, 4):
        raise ValueError(f"H3 video must be [T,H,W,C] or [B,C,T,H,W], got {tuple(video.shape)}")
    if video.shape[0] != 1:
        raise ValueError("MiniMax H3 Super supports exactly one video per request")
    if video.shape[1] == 1:
        video = video.expand(-1, 3, -1, -1, -1)
    elif video.shape[1] == 4:
        video = video[:, :3]
    return video.contiguous()


def _as_audio_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    audio = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    if audio.ndim == 1:
        audio = audio[None, None]
    elif audio.ndim == 2:
        audio = audio[None]
    if audio.ndim != 3:
        raise ValueError(f"H3 audio must be [samples], [channels,samples], or [B,channels,samples], got {audio.shape}")
    if audio.shape[0] != 1:
        raise ValueError("MiniMax H3 Super supports exactly one audio sample per request")
    if audio.shape[1] == 1:
        audio = audio.expand(-1, 2, -1)
    elif audio.shape[1] != 2:
        raise ValueError(f"H3 audio must be mono or stereo, got {audio.shape[1]} channels")
    if not audio.is_floating_point():
        info = torch.iinfo(audio.dtype)
        audio = audio.float() / float(max(abs(info.min), info.max))
    return audio.float().clamp(-1, 1).contiguous()


def _resolve_refiner_frame_count(draft_frames: int) -> int:
    if draft_frames >= 241:
        return 241
    if draft_frames >= 121:
        return 121
    raise ValueError(f"H3 Super draft must contain at least 121 frames, got {draft_frames}")


def _configure_h3_refiner_vae(vae: Any) -> None:
    """Use the released one-tile geometry for the 121-frame H3 draft."""
    vae.tile_sample_min_num_frames = H3_REFINER_VAE_TILE_FRAMES
    vae.tile_sample_stride_num_frames = H3_REFINER_VAE_TILE_FRAMES - H3_REFINER_VAE_TILE_OVERLAP_FRAMES
    vae.tile_sample_min_height = H3_REFINER_VAE_TILE_HEIGHT
    vae.tile_sample_stride_height = H3_REFINER_VAE_TILE_HEIGHT - H3_REFINER_VAE_TILE_OVERLAP_SPATIAL
    vae.tile_sample_min_width = H3_REFINER_VAE_TILE_WIDTH
    vae.tile_sample_stride_width = H3_REFINER_VAE_TILE_WIDTH - H3_REFINER_VAE_TILE_OVERLAP_SPATIAL
    # Diffusers currently gates temporal tiled encode on this shared flag.
    # 121 frames stay in one tile; the 241-frame arm is split safely.
    vae.use_framewise_decoding = True


def _encode_h3_refiner_media(video: torch.Tensor, audio: torch.Tensor) -> bytes:
    """Encode the final worker-resident tensors before crossing process boundaries."""
    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

    if video.ndim != 5 or video.shape[0] != 1 or video.shape[-1] != 3 or video.dtype != torch.uint8:
        raise ValueError(f"H3 refiner video must be uint8 BTHWC, got {tuple(video.shape)} {video.dtype}")
    if audio.ndim != 3 or audio.shape[0] != 1:
        raise ValueError(f"H3 refiner audio must be BCH, got {tuple(audio.shape)}")
    video_frames = video[0].detach().cpu().numpy()
    audio_waveform = audio[0].detach().float().cpu().numpy()
    return mux_video_audio_bytes(
        video_frames,
        audio_waveform,
        fps=H3_REFINER_FPS,
        audio_sample_rate=H3_REFINER_AUDIO_SAMPLE_RATE,
        crf="18",
        video_codec_options={"preset": "ultrafast", "threads": "0"},
    )


def get_ltx25_h3_refiner_post_process_func(od_config: Any):
    del od_config

    def post_process_func(output: Any):
        if isinstance(output, Mapping) and isinstance(output.get("video_mp4"), bytes):
            return {
                "payload": {"video": output["video_mp4"]},
                "metadata": {
                    "video": {"fps": H3_REFINER_FPS, "media_type": "video/mp4"},
                    "audio": {"sample_rate": H3_REFINER_AUDIO_SAMPLE_RATE},
                },
            }
        if not (isinstance(output, tuple) and len(output) == 2):
            return output
        video, audio = output
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().float().cpu()
        return {
            "video": video,
            "audio": audio,
            "audio_sample_rate": H3_REFINER_AUDIO_SAMPLE_RATE,
            "fps": H3_REFINER_FPS,
        }

    return post_process_func


class LTX25H3RefinerPipeline(LTXI2VConditioningMixin, LTXRuntime):
    """Encode an H3 draft and run only LTX-2.5's three-step refiner."""

    pipeline_kind = "h3_refiner"
    component_profile = LTX25_TWO_STAGE_COMPONENT_PROFILE
    pipeline_recipe = LTX25_H3_REFINER_RECIPE
    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)
    _resident_modules: ClassVar[list[str]] = list(component_profile.resident_modules)
    supports_request_batch = False
    support_image_input = True
    unified_text_image_entry = False
    dummy_run_num_frames = 121

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _configure_h3_refiner_vae(self.vae)
        self._h3_source_audio: torch.Tensor | None = None
        additional_config = getattr(self.od_config, "additional_config", {}) or {}
        checkpoint_source = additional_config.get("taehv_checkpoint", TAEHV_CHECKPOINT_URL)
        self.encode_output_video = bool(additional_config.get("encode_output_video", True))
        self.taehv_decoder = LTXWideTAEHVDecoder.from_checkpoint(
            checkpoint_source,
            device=self.device,
            dtype=self.vae.dtype,
        )

    def _prepare_h3_video(
        self,
        value: Any,
        *,
        frame_count: int,
    ) -> tuple[torch.Tensor, PIL.Image.Image]:
        video = _as_video_tensor(value)
        video = video[:, :, :frame_count]
        first_frame = video[0, :, 0]
        input_is_preprocessed = video.dtype == torch.bfloat16 and tuple(video.shape[-2:]) == (
            H3_REFINER_DRAFT_HEIGHT,
            H3_REFINER_DRAFT_WIDTH,
        )
        if input_is_preprocessed:
            first_frame = (first_frame.float() + 1.0) * 127.5
        elif first_frame.is_floating_point():
            first_frame = first_frame.float()
            if float(first_frame.amax()) <= 1.5:
                first_frame = first_frame * 255.0
        first_frame = first_frame.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        image = PIL.Image.fromarray(first_frame, mode="RGB")

        if input_is_preprocessed:
            return video.to(device=self.device, dtype=self.vae.dtype), image

        input_was_integer = not video.is_floating_point()
        video = video.to(device=self.device, dtype=self.vae.dtype)
        if input_was_integer:
            video = video / 127.5 - 1.0
        elif float(video.detach().amax()) <= 1.5 and float(video.detach().amin()) >= -0.01:
            video = video * 2.0 - 1.0
        video = video.permute(0, 2, 1, 3, 4).flatten(0, 1)
        video = F.interpolate(
            video,
            size=(H3_REFINER_DRAFT_HEIGHT, H3_REFINER_DRAFT_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        video = video.unflatten(0, (1, frame_count)).permute(0, 2, 1, 3, 4).contiguous()
        return video, image

    def _encode_h3_video(self, video: torch.Tensor) -> torch.Tensor:
        encoded = self.vae.encode(video)
        lowres_latents = retrieve_latents(encoded, None, "argmax").to(self.vae.dtype)
        return self._spatial_upsample_phase(lowres_latents)

    @staticmethod
    def _conform_source_audio(audio: torch.Tensor, *, frame_count: int) -> torch.Tensor:
        sample_count = int(frame_count / H3_REFINER_FPS * H3_REFINER_AUDIO_SAMPLE_RATE)
        audio = audio[..., :sample_count]
        if audio.shape[-1] < sample_count:
            audio = F.pad(audio, (0, sample_count - audio.shape[-1]))
        return audio.contiguous()

    def _encode_h3_audio(self, audio: torch.Tensor, *, frame_count: int) -> torch.Tensor:
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError("MiniMax H3 Super refinement requires torchaudio") from exc

        waveform = torchaudio.functional.resample(
            audio.to(device=self.device, dtype=torch.float32),
            H3_REFINER_AUDIO_SAMPLE_RATE,
            H3_REFINER_AUDIO_VAE_SAMPLE_RATE,
        )
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=H3_REFINER_AUDIO_VAE_SAMPLE_RATE,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            f_min=0.0,
            f_max=8000.0,
            n_mels=64,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        ).to(self.device)(waveform)
        mel = mel.clamp_min(1e-5).log().transpose(-1, -2).to(self.audio_vae.dtype)
        encoded = self.audio_vae.encode(mel)
        latents = retrieve_latents(encoded, None, "argmax").to(self.audio_vae.dtype)
        target_frames = round(frame_count / H3_REFINER_FPS * 25.0)
        latents = latents[:, :, :target_frames]
        if latents.shape[2] < target_frames:
            latents = F.pad(latents, (0, 0, 0, target_frames - latents.shape[2]))
        return latents.contiguous()

    def _extract_h3_payload(
        self,
        req: DiffusionRequestBatch,
    ) -> tuple[Any, Any, Any | None, int]:
        if len(req.prompts) != 1:
            raise ValueError("MiniMax H3 Super refinement supports one request at a time")
        if req.is_dummy_run():
            frame_count = self.pipeline_recipe.num_frames
            video = torch.zeros(
                (1, 3, frame_count, H3_REFINER_DRAFT_HEIGHT, H3_REFINER_DRAFT_WIDTH),
                dtype=torch.uint8,
            )
            audio = torch.zeros(
                (1, 2, int(frame_count / H3_REFINER_FPS * H3_REFINER_AUDIO_SAMPLE_RATE)),
                dtype=torch.float32,
            )
            return video, audio, None, frame_count

        additional = _prompt_additional_information(req.prompts[0])
        video = additional.get("h3_video")
        audio = additional.get("h3_audio")
        if video is None or audio is None:
            raise ValueError("LTX25H3RefinerPipeline requires h3_video and h3_audio from the upstream H3 stage")
        draft_frames = int(_as_video_tensor(video).shape[2])
        return video, audio, additional.get("h3_first_frame"), _resolve_refiner_frame_count(draft_frames)

    def _normalize_refiner_sampling(self, req: DiffusionRequestBatch, *, frame_count: int) -> None:
        for sampling in req.sampling_params_list:
            sampling.height = H3_REFINER_OUTPUT_HEIGHT
            sampling.width = H3_REFINER_OUTPUT_WIDTH
            sampling.num_frames = frame_count
            sampling.fps = int(H3_REFINER_FPS)
            sampling.frame_rate = H3_REFINER_FPS
            sampling.num_inference_steps = len(self.pipeline_recipe.phases[0].sigmas or ()) - 1
            sampling.guidance_scale = None
            sampling.guidance_scale_provided = False
            sampling.sigmas = None
            sampling.lora_request = None
            sampling.lora_scale = 1.0
            sampling.extra_args = {}

    def _forward_request(self, req: DiffusionRequestBatch, **kwargs):
        if kwargs.get("latents") is not None or kwargs.get("audio_latents") is not None:
            raise ValueError("LTX25H3RefinerPipeline owns the H3 video/audio latent encoding")

        raw_video, raw_audio, first_frame, frame_count = self._extract_h3_payload(req)
        self._normalize_refiner_sampling(req, frame_count=frame_count)
        video, draft_first_frame = self._prepare_h3_video(raw_video, frame_count=frame_count)
        source_audio = self._conform_source_audio(_as_audio_tensor(raw_audio), frame_count=frame_count)
        video_latents = self._encode_h3_video(video)
        audio_latents = self._encode_h3_audio(source_audio, frame_count=frame_count)
        self._h3_source_audio = source_audio
        try:
            kwargs.update(
                {
                    "image": first_frame if first_frame is not None else draft_first_frame,
                    "image_crf": 18,
                    "height": H3_REFINER_OUTPUT_HEIGHT,
                    "width": H3_REFINER_OUTPUT_WIDTH,
                    "num_frames": frame_count,
                    "frame_rate": H3_REFINER_FPS,
                    "num_inference_steps": len(self.pipeline_recipe.phases[0].sigmas or ()) - 1,
                    "latents": video_latents,
                    "audio_latents": audio_latents,
                }
            )
            return super()._forward_request(req, **kwargs)
        finally:
            self._h3_source_audio = None

    def _decode_output(
        self,
        *,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        output_type: str,
        connector_prompt_embeds: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
        device: torch.device,
        decode_timestep: float | list[float],
        decode_noise_scale: float | list[float] | None,
        prompt_batch_size: int,
    ) -> DiffusionOutput:
        if output_type == "latent":
            return self._make_output((latents, audio_latents))
        if self._h3_source_audio is None:
            raise RuntimeError("LTX H3 refiner requires the original H3 audio during decode")

        del generator, device, decode_timestep, decode_noise_scale, prompt_batch_size
        latents = latents.to(connector_prompt_embeds.dtype)

        dist_initialized = torch.distributed.is_initialized()
        is_output_rank = not dist_initialized or torch.distributed.get_rank() == 0
        should_decode_video = not self.distributed_video_decode or is_output_rank
        if should_decode_video:
            video = self.taehv_decoder.decode_video(latents)
        else:
            video = torch.empty(0, device=latents.device, dtype=latents.dtype)

        if self.distributed_video_decode and not is_output_rank:
            return self._make_output(
                (
                    torch.empty(0, device=video.device, dtype=video.dtype),
                    torch.empty(0, device=audio_latents.device, dtype=audio_latents.dtype),
                )
            )

        if video.numel() > 0:
            video = video.permute(0, 1, 3, 4, 2).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).contiguous()
        if self.encode_output_video:
            return self._make_output({"video_mp4": _encode_h3_refiner_media(video, self._h3_source_audio)})
        return self._make_output((video, self._h3_source_audio))


__all__ = [
    "H3_REFINER_AUDIO_SAMPLE_RATE",
    "H3_REFINER_DRAFT_HEIGHT",
    "H3_REFINER_DRAFT_WIDTH",
    "H3_REFINER_FPS",
    "H3_REFINER_FRAME_COUNTS",
    "H3_REFINER_OUTPUT_HEIGHT",
    "H3_REFINER_OUTPUT_WIDTH",
    "LTX25H3RefinerPipeline",
    "get_ltx25_h3_refiner_post_process_func",
]
