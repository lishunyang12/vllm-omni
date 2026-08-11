# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate raw LTX video and audio outputs with the official or Omni runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

DIFFUSERS_LTX25_REVISION = "7564fb016dabda0c943416190fc92398c50b1b20"
MODES = ("distilled-one-stage", "distilled-two-stage", "full")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("diffusers", "official", "omni"), required=True)
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="distilled-one-stage",
        help="Official LTX-2.5 model-card recipe",
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--model-class-name")
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gemma-root", type=Path)
    parser.add_argument("--image", type=Path, help="Override the request's I2V conditioning image")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    parser.add_argument("--disable-cudnn-sdpa", action="store_true")
    parser.add_argument(
        "--omni-attention-backend",
        choices=("TORCH_SDPA", "CUDNN_ATTN", "TRTLLM_ATTN"),
        default="CUDNN_ATTN",
        help="Attention backend for Omni accuracy A/B runs (default: CUDNN_ATTN).",
    )
    return parser.parse_args()


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.accelerator.synchronize()


def _reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.accelerator.reset_peak_memory_stats()


def _peak_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    _synchronize_cuda()
    return float(torch.accelerator.max_memory_allocated() / (1024**2))


def _latency_metadata(start: float, load_end: float, generation_end: float) -> dict[str, float]:
    return {
        "load": (load_end - start) * 1000.0,
        "generation": (generation_end - load_end) * 1000.0,
        "e2e": (generation_end - start) * 1000.0,
    }


def _save_outputs(
    output_dir: Path,
    *,
    video: torch.Tensor,
    audio: torch.Tensor,
    audio_sample_rate: int,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    video = video.detach().float().cpu().clamp(0.0, 1.0)
    audio = audio.detach().float().cpu()
    np.save(output_dir / "video.npy", video.numpy())
    np.save(output_dir / "audio.npy", audio.numpy())

    frame_indices = sorted({0, video.shape[0] // 2, video.shape[0] - 1})
    for index in frame_indices:
        frame = video[index].mul(255.0).round().to(torch.uint8).numpy()
        Image.fromarray(frame).save(output_dir / f"frame_{index:04d}.png")

    metadata.update(
        {
            "video_shape": list(video.shape),
            "audio_shape": list(audio.shape),
            "audio_sample_rate": int(audio_sample_rate),
            "frame_indices": frame_indices,
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _insert_official_paths(official_root: Path) -> None:
    for relative_path in ("packages/ltx-core/src", "packages/ltx-pipelines/src"):
        path = str((official_root / relative_path).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def _configure_official_sdpa(pipeline: Any) -> None:
    """Use PyTorch SDPA for official connector and denoiser attention."""
    from ltx_core.loader.attention_ops import set_attention_module_op
    from ltx_core.model.transformer.attention import PytorchAttention

    class AllValidSDPA(PytorchAttention):
        def __call__(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            heads: int,
            mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if mask is not None and torch.count_nonzero(mask).item():
                raise ValueError("The LTX accuracy guard cannot discard a non-empty attention mask")
            return super().__call__(query, key, value, heads, mask=None)

    attention = AllValidSDPA()
    module_op = set_attention_module_op(
        attention=attention,
        masked_attention=attention,
    )
    owners_and_attributes = (
        (pipeline.stage, "_transformer_builder"),
        (pipeline.prompt_encoder, "_embeddings_processor_builder"),
    )
    for owner, attribute in owners_and_attributes:
        builder = getattr(owner, attribute)
        setattr(
            owner,
            attribute,
            builder.with_module_ops((*builder.module_ops, module_op)),
        )


@torch.inference_mode()
def _run_official(args: argparse.Namespace, request: dict[str, Any]) -> None:
    if args.mode != "distilled-one-stage":
        raise ValueError("The legacy official backend only supports --mode distilled-one-stage")
    if args.official_root is None or args.checkpoint is None or args.gemma_root is None:
        raise ValueError("Official backend requires --official-root, --checkpoint, and --gemma-root")
    start = time.perf_counter()
    _reset_peak_gpu_memory()
    _insert_official_paths(args.official_root)
    # vLLM disables cuDNN SDPA during import; mirror the worker's Gemma dispatch.
    torch.backends.cuda.enable_cudnn_sdp(False)

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.types import OffloadMode

    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=str(args.checkpoint),
        gemma_root=str(args.gemma_root),
        loras=(),
        offload_mode=OffloadMode.CPU if args.enable_layerwise_offload else OffloadMode.NONE,
    )
    _configure_official_sdpa(pipeline)
    _synchronize_cuda()
    load_end = time.perf_counter()
    image_path = request.get("image")
    images = (
        []
        if image_path is None
        else [
            ImageConditioningInput(
                path=str(image_path),
                frame_idx=0,
                strength=1.0,
                # Both runtimes must receive the same source pixels. The
                # official CLI's optional H.264 preprocessing is not part of
                # the seeded model trajectory under test.
                crf=0,
            )
        ]
    )
    video, audio = pipeline(
        prompt=request["prompt"],
        negative_prompt=request["negative_prompt"],
        seed=request["seed"],
        height=request["height"],
        width=request["width"],
        num_frames=request["num_frames"],
        frame_rate=request["fps"],
        num_inference_steps=request["num_inference_steps"],
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=request["video_cfg_scale"],
            stg_scale=request["video_stg_scale"],
            rescale_scale=request["video_rescale_scale"],
            modality_scale=request["video_modality_scale"],
            skip_step=0,
            stg_blocks=request["video_stg_blocks"],
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=request["audio_cfg_scale"],
            stg_scale=request["audio_stg_scale"],
            rescale_scale=request["audio_rescale_scale"],
            modality_scale=request["audio_modality_scale"],
            skip_step=0,
            stg_blocks=request["audio_stg_blocks"],
        ),
        images=images,
        max_batch_size=4,
    )
    video_tensor = torch.cat([chunk.detach().cpu() for chunk in video], dim=0)
    _synchronize_cuda()
    generation_end = time.perf_counter()
    _save_outputs(
        args.output_dir,
        video=_canonical_video(video_tensor),
        audio=audio.waveform,
        audio_sample_rate=audio.sampling_rate,
        metadata={
            "backend": "official",
            "attention_backend": "torch_sdpa",
            "official_revision": os.environ.get("VLLM_TEST_LTX_OFFICIAL_REVISION"),
            "checkpoint": str(args.checkpoint),
            "latency_ms": _latency_metadata(start, load_end, generation_end),
            "peak_gpu_memory_mb": _peak_gpu_memory_mb(),
        },
    )


def _load_image(path: str | Path) -> Image.Image:
    with Image.open(str(path)) as source:
        image = source.convert("RGB")
        image.load()
    return image


@torch.inference_mode()
def _run_diffusers(args: argparse.Namespace, request: dict[str, Any]) -> None:
    if args.model is None:
        raise ValueError("Diffusers backend requires --model")
    if request.get("image") is not None and args.mode != "distilled-one-stage":
        raise ValueError("The official LTX-2.5 model card only defines I2V for the one-stage recipe")

    if args.disable_cudnn_sdpa:
        torch.backends.cuda.enable_cudnn_sdp(False)

    start = time.perf_counter()
    _reset_peak_gpu_memory()

    import diffusers
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        LTX2ImageToVideoPipeline,
        LTX2LatentUpsamplePipeline,
        LTX2Pipeline,
        LTX2VideoTransformer3DModel,
    )
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    pipeline_class = LTX2ImageToVideoPipeline if request.get("image") is not None else LTX2Pipeline
    upsample_pipeline = None
    if args.mode == "full":
        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            args.model,
            subfolder="transformer_full",
            dtype=torch.bfloat16,
        )
        pipeline = pipeline_class.from_pretrained(
            args.model,
            transformer=transformer,
            dtype=torch.bfloat16,
        )
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            use_dynamic_shifting=True,
            shift_terminal=0.1,
        )
    else:
        pipeline = pipeline_class.from_pretrained(args.model, dtype=torch.bfloat16)

    pipeline.enable_model_cpu_offload()
    if args.mode == "distilled-two-stage":
        pipeline.vae.enable_tiling()
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            args.model,
            subfolder="latent_upsampler",
            dtype=torch.bfloat16,
        ).to("cuda")
        upsample_pipeline = LTX2LatentUpsamplePipeline(
            vae=pipeline.vae,
            latent_upsampler=latent_upsampler,
        )
    _synchronize_cuda()
    load_end = time.perf_counter()

    generator = torch.Generator("cuda").manual_seed(request["seed"])
    common: dict[str, Any] = {
        "prompt": request["prompt"],
        "negative_prompt": request["negative_prompt"],
        "frame_rate": float(request["fps"]),
        "generator": generator,
        "return_dict": False,
    }
    if args.mode != "full":
        common.update(
            {
                "guidance_scale": request["video_cfg_scale"],
                "audio_guidance_scale": request["audio_cfg_scale"],
                "stg_scale": request["video_stg_scale"],
                "audio_stg_scale": request["audio_stg_scale"],
                "modality_scale": request["video_modality_scale"],
                "audio_modality_scale": request["audio_modality_scale"],
                "guidance_rescale": request["video_rescale_scale"],
                "audio_guidance_rescale": request["audio_rescale_scale"],
            }
        )

    image_path = request.get("image")
    if image_path is not None:
        common["image"] = _load_image(image_path)

    stage_latency_ms: dict[str, float] = {}
    if args.mode == "distilled-two-stage":
        assert upsample_pipeline is not None
        stage_start = time.perf_counter()
        stage_1_latents, audio_latents = pipeline(
            height=request["height"],
            width=request["width"],
            num_frames=request["num_frames"],
            sigmas=request["sigmas"],
            output_type="latent",
            **common,
        )
        _synchronize_cuda()
        upsample_start = time.perf_counter()
        stage_latency_ms["stage_1"] = (upsample_start - stage_start) * 1000.0
        upsampled_latents = upsample_pipeline(
            latents=stage_1_latents,
            output_type="latent",
            return_dict=False,
        )[0]
        _synchronize_cuda()
        stage_2_start = time.perf_counter()
        stage_latency_ms["latent_upsample"] = (stage_2_start - upsample_start) * 1000.0
        stage_2_sigmas = request.get("stage_2_sigmas", [0.909375, 0.725, 0.421875])
        video, audio = pipeline(
            num_frames=request["num_frames"],
            sigmas=stage_2_sigmas,
            latents=upsampled_latents,
            audio_latents=audio_latents,
            noise_scale=stage_2_sigmas[0],
            output_type="np",
            **common,
        )
        _synchronize_cuda()
        stage_latency_ms["stage_2"] = (time.perf_counter() - stage_2_start) * 1000.0
    else:
        call_kwargs: dict[str, Any] = {
            "height": request["height"],
            "width": request["width"],
            "num_frames": request["num_frames"],
            "output_type": "np",
        }
        if args.mode == "distilled-one-stage":
            call_kwargs["sigmas"] = request["sigmas"]
        video, audio = pipeline(**call_kwargs, **common)

    video_tensor = _canonical_video(video)
    audio_tensor = torch.as_tensor(np.asarray(audio) if not isinstance(audio, torch.Tensor) else audio)
    _synchronize_cuda()
    generation_end = time.perf_counter()
    _save_outputs(
        args.output_dir,
        video=video_tensor,
        audio=audio_tensor,
        audio_sample_rate=int(pipeline.vocoder.config.output_sampling_rate),
        metadata={
            "backend": "diffusers",
            "mode": args.mode,
            "task": "i2v" if image_path is not None else "t2v",
            "model": args.model,
            "diffusers_version": diffusers.__version__,
            "diffusers_ltx25_api_revision": DIFFUSERS_LTX25_REVISION,
            "cudnn_sdpa": torch.backends.cuda.cudnn_sdp_enabled(),
            "latency_ms": _latency_metadata(start, load_end, generation_end),
            "stage_latency_ms": stage_latency_ms,
            "peak_gpu_memory_mb": _peak_gpu_memory_mb(),
        },
    )


def _unwrap_omni_output(output: Any) -> tuple[Any, Any, int]:
    from vllm_omni.outputs import OmniRequestOutput

    audio = None
    audio_sample_rate = None
    frames = output[0] if isinstance(output, list) and output else output
    if isinstance(frames, OmniRequestOutput):
        multimodal_output = frames.multimodal_output or {}
        audio = multimodal_output.get("audio")
        audio_sample_rate = multimodal_output.get("audio_sample_rate")
        if frames.is_pipeline_output and isinstance(frames.request_output, OmniRequestOutput):
            frames = frames.request_output
            multimodal_output = frames.multimodal_output or {}
            audio = multimodal_output.get("audio", audio)
            audio_sample_rate = multimodal_output.get("audio_sample_rate", audio_sample_rate)
        if isinstance(frames, OmniRequestOutput):
            if not frames.images:
                raise ValueError("No video frames found in OmniRequestOutput")
            frames = frames.images

    if isinstance(frames, list) and len(frames) == 1:
        frames = frames[0]
    if isinstance(frames, tuple) and len(frames) == 2:
        frames, audio = frames
    elif isinstance(frames, dict):
        audio = frames.get("audio", audio)
        audio_sample_rate = frames.get("audio_sample_rate", audio_sample_rate)
        frames = frames.get("frames", frames.get("video"))

    if frames is None or audio is None or audio_sample_rate is None:
        raise ValueError("Omni output did not contain video, audio, and audio_sample_rate")
    return frames, audio, int(audio_sample_rate)


def _omni_performance(output: Any) -> tuple[dict[str, float], float]:
    current = output[0] if isinstance(output, list) and output else output
    stage_durations = getattr(current, "stage_durations", {})
    peak_memory_mb = getattr(current, "peak_memory_mb", 0.0)
    nested = getattr(current, "request_output", None)
    if nested is not None:
        stage_durations = stage_durations or getattr(nested, "stage_durations", {})
        peak_memory_mb = peak_memory_mb or getattr(nested, "peak_memory_mb", 0.0)
    return (
        {str(key): float(value) for key, value in stage_durations.items()},
        float(peak_memory_mb or 0.0),
    )


def _canonical_video(video: Any) -> torch.Tensor:
    if isinstance(video, list):
        if len(video) == 1:
            return _canonical_video(video[0])
        tensors = [torch.as_tensor(np.asarray(item) if not isinstance(item, torch.Tensor) else item) for item in video]
        if all(item.ndim == 3 for item in tensors):
            video = torch.stack(tensors)
        elif all(item.ndim == 4 for item in tensors):
            video = torch.cat(tensors)
        else:
            raise ValueError(f"Cannot combine video items with dimensions {[item.ndim for item in tensors]}")
    tensor = torch.as_tensor(np.asarray(video) if not isinstance(video, torch.Tensor) else video).detach().cpu()
    if tensor.ndim == 5:
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 4D video tensor, got {tuple(tensor.shape)}")
    if tensor.shape[-1] in (3, 4):
        tensor = tensor[..., :3]
    elif tensor.shape[1] in (3, 4):
        tensor = tensor[:, :3].permute(0, 2, 3, 1)
    elif tensor.shape[0] in (3, 4):
        tensor = tensor[:3].permute(1, 2, 3, 0)
    else:
        raise ValueError(f"Cannot infer video channel dimension from {tuple(tensor.shape)}")
    tensor = tensor.float()
    if tensor.numel() and tensor.max() > 1:
        tensor = tensor / 255.0
    elif tensor.numel() and tensor.min() < 0:
        tensor = tensor.clamp(-1.0, 1.0).add(1.0).mul(0.5)
    return tensor.clamp(0.0, 1.0)


@torch.inference_mode()
def _run_omni(args: argparse.Namespace, request: dict[str, Any]) -> None:
    if args.model is None:
        raise ValueError("Omni backend requires --model")
    if args.mode == "full" and request.get("image") is not None:
        raise ValueError("Omni LTX-2.5 Full/SFT currently supports T2V only; remove the image input")

    requested_model_class_name = args.model_class_name or (
        "LTX2FullPipeline"
        if args.mode == "full"
        else ("LTX2DistilledPipeline" if args.mode == "distilled-two-stage" else "LTX2Pipeline")
    )

    attention_config = {"default": {"backend": args.omni_attention_backend}}
    start = time.perf_counter()

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import get_extra_body_params, get_model_class_name
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(request["seed"])
    omni = Omni(
        model=args.model,
        model_class_name=requested_model_class_name,
        enforce_eager=True,
        enable_layerwise_offload=args.enable_layerwise_offload,
        diffusion_attention_config=attention_config,
        parallel_config=DiffusionParallelConfig(),
    )
    try:
        _synchronize_cuda()
        load_end = time.perf_counter()
        model_class_name = get_model_class_name(omni)
        spatial_scale = 2 if args.mode == "distilled-two-stage" else 1
        # LTX-2.5 recipes own and pin their sigma schedules. Forwarding the
        # identical request list is rejected by the recipe's request guard.
        sampling_params = OmniDiffusionSamplingParams(
            height=request["height"] * spatial_scale,
            width=request["width"] * spatial_scale,
            generator=generator,
            guidance_scale=None,
            num_inference_steps=30 if args.mode == "full" else request["num_inference_steps"],
            num_frames=request["num_frames"],
            fps=request["fps"],
            sigmas=None,
            frame_rate=float(request["fps"]),
            output_type="np",
        )
        guidance = {
            key: value for key, value in request.items() if key.startswith("video_") or key.startswith("audio_")
        }
        if args.mode != "full":
            apply_declared_extra_args(sampling_params, get_extra_body_params(model_class_name), guidance)
        prompt: dict[str, Any] = {
            "prompt": request["prompt"],
            "negative_prompt": request["negative_prompt"],
        }
        image_path = request.get("image")
        if image_path is not None:
            with Image.open(str(image_path)) as source_image:
                image = source_image.convert("RGB")
                image.load()
            prompt["multi_modal_data"] = {"image": image}
        output = omni.generate(prompt, sampling_params)
        stage_durations, peak_memory_mb = _omni_performance(output)
        video, audio, audio_sample_rate = _unwrap_omni_output(output)
        audio_tensor = torch.as_tensor(np.asarray(audio) if not isinstance(audio, torch.Tensor) else audio)
        video_tensor = _canonical_video(video)
        _synchronize_cuda()
        generation_end = time.perf_counter()
        peak_memory_mb = peak_memory_mb or _peak_gpu_memory_mb()
        _save_outputs(
            args.output_dir,
            video=video_tensor,
            audio=audio_tensor,
            audio_sample_rate=audio_sample_rate,
            metadata={
                "backend": "omni",
                "attention_backend": args.omni_attention_backend.lower(),
                "model": args.model,
                "model_class_name": model_class_name,
                "mode": args.mode,
                "task": "i2v" if image_path is not None else "t2v",
                "latency_ms": _latency_metadata(start, load_end, generation_end),
                "stage_latency_ms": stage_durations,
                "peak_gpu_memory_mb": peak_memory_mb,
            },
        )
    finally:
        omni.shutdown()


def main() -> None:
    args = _parse_args()
    request = json.loads(args.request.read_text())
    if args.image is not None:
        request["image"] = str(args.image.resolve())
    if args.backend == "official":
        _run_official(args, request)
    elif args.backend == "diffusers":
        _run_diffusers(args, request)
    else:
        _run_omni(args, request)


if __name__ == "__main__":
    main()
