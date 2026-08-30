# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate MiniMax-H3 request-mode and DLO lifecycle behavior.

The DLO modes accept any positive data-parallel size.  With AllGather enabled,
the script submits one complete DP wave and verifies recovery after an invalid
wave.  It also checks that every mode shuts down all worker processes cleanly.

Examples:

    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
    VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=600 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python examples/offline_inference/minimax_h3/dlo_lifecycle.py \
        --model /path/to/MiniMax-H3/FL2VA --mode dlo --dp-size 4

    # Rank-local mmap storage, with no DLO collective or DP wave scheduling.
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
    CUDA_VISIBLE_DEVICES=0,1 \
    python examples/offline_inference/minimax_h3/dlo_lifecycle.py \
        --model /path/to/MiniMax-H3/FL2VA --mode dlo-no-allgather --dp-size 2

    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
    CUDA_VISIBLE_DEVICES=0,1 \
    python examples/offline_inference/minimax_h3/dlo_lifecycle.py \
        --model /path/to/MiniMax-H3/FL2VA --mode request --tp-size 2

    # One offline request with SP8-backed DLO AllGather.
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python examples/offline_inference/minimax_h3/dlo_lifecycle.py \
        --model /path/to/MiniMax-H3/FL2VA --mode dlo \
        --dp-size 1 --sp-size 8 --video-output output.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.entrypoints.async_omni import AsyncOmni

DEFAULT_PROMPTS = (
    "At night, three cats march into a bedroom playing tiny brass instruments, "
    "then abruptly file out, with synchronized room ambience.",
    "A paper boat crosses a rain-filled street while distant traffic and water sounds remain synchronized.",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to MiniMax-H3/FL2VA")
    parser.add_argument(
        "--mode",
        choices=("dlo", "dlo-no-allgather", "request"),
        required=True,
        help="DLO (with or without AllGather) or ordinary non-DLO request mode",
    )
    parser.add_argument(
        "--dp-size",
        type=positive_int,
        default=2,
        help="DLO data-parallel size (for example 1, 2, 4, or 8; default: 2)",
    )
    parser.add_argument(
        "--tp-size",
        type=positive_int,
        default=2,
        help="Tensor-parallel size used only by request mode (default: 2)",
    )
    parser.add_argument(
        "--sp-size",
        type=positive_int,
        default=1,
        help="Ulysses SP and text-encoder/VAE parallel size for DLO modes (default: 1)",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--repetitions", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument(
        "--resident-layers",
        type=nonnegative_int,
        default=0,
        help="Leading DiT layers kept on each GPU during denoise (default: 0)",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--batch-wait-ms", type=float, default=500.0)
    parser.add_argument("--init-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 file used as the first valid request prompt",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video-output", type=Path)
    return parser.parse_args()


def engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    is_dlo = args.mode in {"dlo", "dlo-no-allgather"}
    if not is_dlo and args.sp_size != 1:
        raise ValueError("--sp-size is supported only by DLO modes")
    common: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "num_gpus": args.dp_size * args.sp_size if is_dlo else args.tp_size,
        "ulysses_degree": args.sp_size if is_dlo else 1,
        "ring_degree": 1,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "diffusion_attention_backend": "CUDNN_ATTN",
        "request_batch_max_wait_ms": args.batch_wait_ms,
        "enforce_eager": True,
        "stage_init_timeout": args.init_timeout,
        "init_timeout": args.init_timeout,
    }
    if is_dlo:
        common.update(
            tensor_parallel_size=1,
            data_parallel_size=args.dp_size,
            text_encoder_tp_size=args.sp_size,
            vae_patch_parallel_size=args.sp_size,
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=args.mode == "dlo",
            dlo_resident_layers=args.resident_layers,
        )
    else:
        common.update(
            tensor_parallel_size=args.tp_size,
            data_parallel_size=1,
            text_encoder_tp_size=args.tp_size,
            vae_patch_parallel_size=args.tp_size,
            enable_distributed_layerwise_offload=False,
        )
    return common


def save_video_output(output: Any, path: Path) -> None:
    frames = np.asarray(output.images[0])
    if np.issubdtype(frames.dtype, np.integer):
        frames_u8 = np.clip(frames, 0, 255).astype(np.uint8)
    else:
        frames_u8 = (np.clip(frames, 0.0, 1.0) * 255).round().astype(np.uint8)

    payload = output.multimodal_output or {}
    audio = np.asarray(payload.get("audio"), dtype=np.float32)
    sample_rate = int(payload.get("audio_sample_rate", 32000))
    video_bytes = mux_video_audio_bytes(
        frames_u8,
        np.squeeze(audio),
        fps=24.0,
        audio_sample_rate=sample_rate,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(video_bytes)


def load_primary_prompt(path: Path | None) -> str:
    if path is None:
        return DEFAULT_PROMPTS[0]
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompt


def sampling_params(
    engine: AsyncOmni,
    args: argparse.Namespace,
    seed: int,
) -> list[Any]:
    params = copy.deepcopy(engine.default_sampling_params_list)
    diffusion = params[0]
    diffusion.width = args.width
    diffusion.height = args.height
    diffusion.fps = 24
    diffusion.num_inference_steps = args.steps
    diffusion.seed = seed
    diffusion.extra_args = {
        "task": "t2va",
        "duration": args.duration,
        "aspect_ratio": "16:9",
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }
    return params


async def generate_one(
    engine: AsyncOmni,
    args: argparse.Namespace,
    *,
    request_id: str,
    prompt: str,
    seed: int,
) -> Any:
    final_output = None
    async for output in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=sampling_params(engine, args, seed),
    ):
        if output.finished:
            final_output = output
    if final_output is None:
        raise RuntimeError(f"{request_id} finished without an output")
    return final_output


def output_summary(output: Any, args: argparse.Namespace) -> dict[str, Any]:
    if not output.images:
        raise RuntimeError(f"{output.request_id} returned no video")
    frames = np.asarray(output.images[0])
    audio = np.asarray(output.multimodal_output.get("audio"))
    if frames.ndim != 4 or tuple(frames.shape[1:]) != (
        args.height,
        args.width,
        3,
    ):
        raise RuntimeError(f"{output.request_id} returned invalid video shape {tuple(frames.shape)}")
    if audio.ndim != 3 or tuple(audio.shape[:2]) != (1, 2):
        raise RuntimeError(f"{output.request_id} returned invalid audio shape {tuple(audio.shape)}")
    if args.duration == 5.0 and args.height == 768 and args.width == 1344:
        if tuple(frames.shape) != (124, 768, 1344, 3) or tuple(audio.shape) != (1, 2, 165600):
            raise RuntimeError(
                f"{output.request_id} default shape mismatch: video={tuple(frames.shape)}, audio={tuple(audio.shape)}"
            )
    return {
        "request_id": output.request_id,
        "frames_shape": list(frames.shape),
        "audio_shape": list(audio.shape),
        "peak_memory_mb": output.peak_memory_mb,
        "stage_durations": output.stage_durations,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    primary_prompt = load_primary_prompt(args.prompt_file)
    engine = AsyncOmni(**engine_kwargs(args))
    summary: dict[str, Any] = {
        "mode": args.mode,
        "engine_kwargs": engine_kwargs(args),
    }
    try:
        uses_dp_wave = args.mode == "dlo" and args.dp_size > 1
        if uses_dp_wave:
            started = time.perf_counter()
            asymmetric = await asyncio.gather(
                generate_one(
                    engine,
                    args,
                    request_id="invalid-empty",
                    prompt="",
                    seed=1001,
                ),
                *(
                    generate_one(
                        engine,
                        args,
                        request_id=f"invalid-peer-{index}",
                        prompt=DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)],
                        seed=1001 + index,
                    )
                    for index in range(1, args.dp_size)
                ),
                return_exceptions=True,
            )
            summary["asymmetric_wave_s"] = time.perf_counter() - started
            summary["asymmetric_errors"] = [
                f"{type(result).__name__}: {result}" for result in asymmetric if isinstance(result, BaseException)
            ]
            if len(summary["asymmetric_errors"]) != args.dp_size:
                raise RuntimeError("Every request in the asymmetric DP wave must fail before dispatch")

        request_count = args.dp_size if uses_dp_wave else 1
        valid_waves = []
        for repetition in range(args.repetitions):
            started = time.perf_counter()
            outputs = await asyncio.gather(
                *(
                    generate_one(
                        engine,
                        args,
                        request_id=(f"recovery-{index}" if args.repetitions == 1 else f"recovery-{repetition}-{index}"),
                        prompt=(primary_prompt if index == 0 else DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)]),
                        seed=args.seed + index,
                    )
                    for index in range(request_count)
                )
            )
            valid_waves.append(
                {
                    "valid_wave_s": time.perf_counter() - started,
                    "outputs": [output_summary(output, args) for output in outputs],
                }
            )
        summary["valid_wave_s"] = valid_waves[-1]["valid_wave_s"]
        summary["outputs"] = valid_waves[-1]["outputs"]
        if args.repetitions > 1:
            summary["valid_waves"] = valid_waves
            summary["valid_wave_mean_s"] = sum(wave["valid_wave_s"] for wave in valid_waves) / len(valid_waves)
        if args.video_output is not None:
            if len(outputs) != 1:
                raise ValueError("--video-output requires a single-request run")
            save_video_output(outputs[0], args.video_output)
            summary["video_output"] = str(args.video_output)
    finally:
        started = time.perf_counter()
        engine.close()
        summary["shutdown_s"] = time.perf_counter() - started

    children = multiprocessing.active_children()
    summary["active_children"] = [{"name": child.name, "pid": child.pid} for child in children]
    if children:
        raise RuntimeError(f"Worker processes remain after shutdown: {children}")
    return summary


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run(args))
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(f"E2E_RESULT {rendered}", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
