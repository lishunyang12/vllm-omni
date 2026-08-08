# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate MiniMax-H3 offload placement, repeatability, and lifecycle.

The DLO modes accept any positive data-parallel size.  With AllGather enabled,
the script submits one complete DP wave and verifies recovery after an invalid
wave.  It also checks that every mode shuts down all worker processes cleanly.

Examples:

    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
    VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=600 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python examples/offline_inference/minimax_h3/dlo_lifecycle.py \
        --model /path/to/MiniMax-H3/FL2VA --mode dlo --dp-size 4 --runs 2

    # Rank-local DLO, with no collective or DP wave scheduling.
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
    CUDA_VISIBLE_DEVICES=0,1 \
    python examples/offline_inference/minimax_h3/dlo_lifecycle.py \
        --model /path/to/MiniMax-H3/FL2VA --mode dlo-no-allgather --dp-size 2 \
        --components dit,text_encoder,vae --quantization fp8 --runs 2

    # One GPU: offload only the encoder and repeat the same request twice.
    CUDA_VISIBLE_DEVICES=0 \
    python examples/offline_inference/minimax_h3/dlo_lifecycle.py \
        --model /path/to/MiniMax-H3/FL2VA --mode layerwise-single \
        --components text_encoder --runs 2

Set VLLM_WORKER_MULTIPROC_METHOD=spawn and
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 for long production-shape runs. Multi-rank
AllGather validation also uses VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=600.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to MiniMax-H3/FL2VA")
    parser.add_argument(
        "--mode",
        choices=(
            "resident-single",
            "layerwise-single",
            "dlo",
            "dlo-no-allgather",
            "request",
        ),
        required=True,
        help=("Single-GPU resident/layerwise, DLO with or without AllGather, or ordinary non-DLO request mode"),
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
        "--components",
        help="Comma-separated dit,text_encoder,vae selection; omitted means all",
    )
    parser.add_argument("--quantization", choices=("fp8",))
    parser.add_argument("--resident-layers", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--batch-wait-ms", type=float, default=500.0)
    parser.add_argument("--init-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--profiler-dir",
        type=Path,
        help="Profile only the final run with torch.profiler and write artifacts here",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Also capture the expensive torch profiler memory timeline and snapshot",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    is_dlo = args.mode in {"dlo", "dlo-no-allgather"}
    common: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "usp": 1,
        "ring": 1,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "diffusion_attention_backend": "CUDNN_ATTN",
        "request_batch_max_wait_ms": args.batch_wait_ms,
        "enforce_eager": True,
        "stage_init_timeout": args.init_timeout,
        "init_timeout": args.init_timeout,
    }
    if args.mode == "resident-single":
        common.update(
            num_gpus=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            text_encoder_tp_size=1,
            vae_patch_parallel_size=1,
        )
    elif args.mode == "layerwise-single":
        common.update(
            num_gpus=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            text_encoder_tp_size=1,
            vae_patch_parallel_size=1,
            enable_layerwise_offload=True,
        )
    elif is_dlo:
        if args.mode == "dlo" and args.resident_layers:
            raise ValueError("DLO+AllGather does not support --resident-layers")
        common.update(
            num_gpus=args.dp_size,
            tensor_parallel_size=1,
            data_parallel_size=args.dp_size,
            text_encoder_tp_size=1,
            vae_patch_parallel_size=1,
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=args.mode == "dlo",
            dlo_resident_layers=args.resident_layers,
        )
    else:
        common.update(
            num_gpus=args.tp_size,
            tensor_parallel_size=args.tp_size,
            data_parallel_size=1,
            text_encoder_tp_size=args.tp_size,
            vae_patch_parallel_size=args.tp_size,
            enable_distributed_layerwise_offload=False,
        )

    if args.components is not None:
        if args.mode in ("resident-single", "request"):
            raise ValueError("--components requires a layerwise or DLO mode")
        common["layerwise_offload_components"] = args.components
    if args.quantization is not None:
        if args.mode == "dlo":
            raise ValueError("Online FP8 requires resident, ordinary layerwise, or DLO without AllGather")
        common["quantization"] = args.quantization
    if args.profiler_dir is not None:
        common["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profiler_dir),
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_memory": args.profile_memory,
            "torch_profiler_dump_cuda_time_total": True,
        }
    return common


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


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def output_summary(output: Any, args: argparse.Namespace) -> dict[str, Any]:
    if not output.images:
        raise RuntimeError(f"{output.request_id} returned no video")
    frames = np.asarray(output.images[0])
    multimodal = output.multimodal_output or {}
    audio = np.asarray(multimodal.get("audio"))
    if frames.ndim != 4 or tuple(frames.shape[1:]) != (
        args.height,
        args.width,
        3,
    ):
        raise RuntimeError(f"{output.request_id} returned invalid video shape {tuple(frames.shape)}")
    if audio.ndim != 3 or tuple(audio.shape[:2]) != (1, 2):
        raise RuntimeError(f"{output.request_id} returned invalid audio shape {tuple(audio.shape)}")
    if int(multimodal.get("fps", 0)) != 24 or int(multimodal.get("audio_sample_rate", 0)) != 32000:
        raise RuntimeError(
            f"{output.request_id} returned invalid media rates: "
            f"fps={multimodal.get('fps')}, audio_sample_rate={multimodal.get('audio_sample_rate')}"
        )
    if args.duration == 5.0 and args.height == 768 and args.width == 1344:
        if tuple(frames.shape) != (124, 768, 1344, 3) or tuple(audio.shape) != (1, 2, 165600):
            raise RuntimeError(
                f"{output.request_id} default shape mismatch: video={tuple(frames.shape)}, audio={tuple(audio.shape)}"
            )
    return {
        "request_id": output.request_id,
        "frames_shape": list(frames.shape),
        "audio_shape": list(audio.shape),
        "frames_sha256": _array_sha256(frames),
        "audio_sha256": _array_sha256(audio),
        "peak_memory_mb": output.peak_memory_mb,
        "stage_durations": output.stage_durations,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    resolved_engine_kwargs = engine_kwargs(args)
    started = time.perf_counter()
    engine = AsyncOmni(**resolved_engine_kwargs)
    engine_init_s = time.perf_counter() - started
    selected_components = args.components
    if selected_components is None:
        selected_components = "none" if args.mode in ("resident-single", "request") else "dit,text_encoder,vae"
    summary: dict[str, Any] = {
        "mode": args.mode,
        "components": selected_components,
        "engine_kwargs": resolved_engine_kwargs,
        "engine_init_s": engine_init_s,
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
        runs = []
        for run_index in range(args.runs):
            profile_this_run = args.profiler_dir is not None and run_index == args.runs - 1
            if profile_this_run:
                await engine.start_profile(profile_prefix="minimax_h3_encoder")
            started = time.perf_counter()
            try:
                outputs = await asyncio.gather(
                    *(
                        generate_one(
                            engine,
                            args,
                            request_id=f"run-{run_index + 1}-{index}",
                            prompt=DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)],
                            seed=2000 + index,
                        )
                        for index in range(request_count)
                    )
                )
            finally:
                if profile_this_run:
                    summary["profiler_results"] = await engine.stop_profile()
            runs.append(
                {
                    "run": run_index + 1,
                    "wall_time_s": time.perf_counter() - started,
                    "outputs": [output_summary(output, args) for output in outputs],
                }
            )
        summary["runs"] = runs
        summary["outputs"] = runs[-1]["outputs"]
        summary["valid_wave_s"] = runs[-1]["wall_time_s"]
        if len(runs) > 1:
            for summary_name, hash_name in (
                ("video_output_deterministic", "frames_sha256"),
                ("audio_output_deterministic", "audio_sha256"),
            ):
                summary[summary_name] = all(
                    len({run["outputs"][index][hash_name] for run in runs}) == 1 for index in range(request_count)
                )
            summary["steady_output_deterministic"] = (
                summary["video_output_deterministic"] and summary["audio_output_deterministic"]
            )
        else:
            summary["video_output_deterministic"] = None
            summary["audio_output_deterministic"] = None
            summary["steady_output_deterministic"] = None
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
    if args.steps < 1 or args.runs < 1:
        raise ValueError("--steps and --runs must be at least 1")
    if args.resident_layers < 0:
        raise ValueError("--resident-layers must be non-negative")
    summary = asyncio.run(run(args))
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(f"E2E_RESULT {rendered}", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
