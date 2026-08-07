# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the MiniMax-H3 BF16/FP8 latency and memory comparison.

Example:

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python benchmarks/diffusion/benchmark_minimax_h3_fp8.py \
        --model /path/to/MiniMax-H3/FL2VA \
        --output-dir ./minimax-h3-fp8-benchmark

The default workload matches the full benchmark reported in the MiniMax-H3
online-FP8 pull request: T2VA, 1344x768, 5 seconds, 50 inference steps, and one
warmup followed by one measured request. BF16 and FP8 run in separate child
processes so CUDA and distributed state cannot leak between configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROMPT = (
    "At night, three cats march into a bedroom playing tiny brass instruments, "
    "then abruptly file out, with synchronized room ambience."
)
WORKER_MODE_ENV = "VLLM_OMNI_MINIMAX_H3_BENCH_MODE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Local path to the MiniMax-H3 FL2VA partition.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("minimax-h3-fp8-benchmark"),
        help="Directory for MP4 files and JSON summaries.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2,
        help="Requests per mode. Run 1 is warmup; run 2 is measured by default.",
    )
    return parser.parse_args()


def _array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _peak_memory_mb(result: Any) -> float:
    value = getattr(result, "peak_memory_mb", 0.0)
    if not value:
        inner = getattr(result, "request_output", None)
        value = getattr(inner, "peak_memory_mb", 0.0)
    return float(value or 0.0)


def _validate_mp4(path: Path) -> dict[str, int]:
    import av

    with av.open(str(path)) as container:
        decoded_video_frames = sum(1 for _ in container.decode(video=0))
    with av.open(str(path)) as container:
        decoded_audio_frames = sum(1 for _ in container.decode(audio=0))
    if decoded_video_frames == 0 or decoded_audio_frames == 0:
        raise RuntimeError(
            f"Expected decodable video and audio in {path}, got "
            f"video_frames={decoded_video_frames}, audio_frames={decoded_audio_frames}"
        )
    return {
        "decoded_video_frames": decoded_video_frames,
        "decoded_audio_frames": decoded_audio_frames,
    }


def _run_worker(args: argparse.Namespace, mode: str) -> None:
    import numpy as np
    import torch

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    visible_device_count = torch.accelerator.device_count()
    if visible_device_count != 4:
        raise RuntimeError(
            "This benchmark requires exactly four visible GPUs; set CUDA_VISIBLE_DEVICES to four B300 devices."
        )

    mode_dir = args.output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    quantization = "fp8" if mode == "fp8" else None
    hardware = []
    for device_index in range(visible_device_count):
        properties = torch.cuda.get_device_properties(device_index)
        hardware.append(
            {
                "logical_index": device_index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_gib": properties.total_memory / 2**30,
            }
        )

    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=4,
            ulysses_degree=1,
            ring_degree=1,
            text_encoder_tp_size=4,
            vae_patch_parallel_size=4,
            vae_parallel_mode="tile",
        ),
        trust_remote_code=True,
        enable_cpu_offload=False,
        enforce_eager=False,
        diffusion_attention_backend="CUDNN_ATTN",
        enable_diffusion_pipeline_profiler=True,
        quantization=quantization,
    )

    records: list[dict[str, Any]] = []
    try:
        for run_index in range(args.num_runs):
            started = time.perf_counter()
            outputs = engine.generate(
                PROMPT,
                OmniDiffusionSamplingParams(
                    height=768,
                    width=1344,
                    fps=24,
                    num_inference_steps=args.num_inference_steps,
                    seed=1101,
                    output_type="np",
                    extra_args={
                        "task": "t2va",
                        "duration": args.duration_seconds,
                        "flow_shift": 12.0,
                        "audio_flow_shift": 3.0,
                    },
                ),
                use_tqdm=False,
            )
            wall_time = time.perf_counter() - started
            if len(outputs) != 1:
                raise RuntimeError(f"Expected one output, found {len(outputs)}")

            result = outputs[0]
            frames = np.asarray(result.images[0])
            multimodal = result.multimodal_output
            if multimodal is None:
                raise RuntimeError("MiniMax-H3 returned no audio metadata")
            audio = np.asarray(multimodal["audio"])
            fps = int(multimodal["fps"])
            sample_rate = int(multimodal["audio_sample_rate"])
            if frames.ndim != 4 or tuple(frames.shape[1:]) != (768, 1344, 3):
                raise RuntimeError(f"Unexpected video shape: {frames.shape}")
            if frames.dtype != np.uint8:
                if not np.issubdtype(frames.dtype, np.floating):
                    raise RuntimeError(f"Expected floating-point or uint8 video frames, got {frames.dtype}")
                frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
            if audio.ndim not in (2, 3) or 2 not in audio.shape:
                raise RuntimeError(f"Unexpected audio shape: {audio.shape}")
            if fps != 24 or sample_rate != 32000:
                raise RuntimeError(f"Unexpected media rates: fps={fps}, audio_sample_rate={sample_rate}")

            output_path = mode_dir / f"t2va_{mode}_run{run_index + 1}.mp4"
            output_path.write_bytes(
                mux_video_audio_bytes(
                    frames,
                    np.squeeze(audio).astype(np.float32),
                    fps=fps,
                    audio_sample_rate=sample_rate,
                )
            )
            media_validation = _validate_mp4(output_path)
            record = {
                "run": run_index + 1,
                "warmup": run_index == 0,
                "wall_time_s": wall_time,
                "stage_durations": dict(getattr(result, "stage_durations", {}) or {}),
                "worker_peak_memory_mb": _peak_memory_mb(result),
                "frames_shape": list(frames.shape),
                "audio_shape": list(audio.shape),
                "fps": fps,
                "audio_sample_rate": sample_rate,
                "frames_sha256": _array_sha256(frames),
                "audio_sha256": _array_sha256(audio),
                "mp4": str(output_path),
                **media_validation,
            }
            records.append(record)
            print("RUN_RESULT " + json.dumps(record, sort_keys=True), flush=True)
    finally:
        engine.close()

    summary = {
        "mode": mode,
        "model": args.model,
        "hardware": hardware,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "parallel_config": "tp4_ulysses1_ring1_text_encoder_tp4_vae_tile4",
        "attention_backend": "CUDNN_ATTN",
        "regional_compile": True,
        "prompt": PROMPT,
        "seed": 1101,
        "height": 768,
        "width": 1344,
        "duration_seconds": args.duration_seconds,
        "num_inference_steps": args.num_inference_steps,
        "runs": records,
    }
    summary_path = mode_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


def _measured_run(summary: dict[str, Any]) -> dict[str, Any]:
    runs = summary["runs"]
    return runs[1] if len(runs) > 1 else runs[0]


def _write_comparison(args: argparse.Namespace) -> None:
    summaries = {}
    for mode in ("bf16", "fp8"):
        summary_path = args.output_dir / mode / "summary.json"
        summaries[mode] = json.loads(summary_path.read_text(encoding="utf-8"))

    bf16 = _measured_run(summaries["bf16"])
    fp8 = _measured_run(summaries["fp8"])
    comparison = {
        "measured_run": 2 if args.num_runs > 1 else 1,
        "bf16": bf16,
        "fp8": fp8,
        "wall_time_speedup": bf16["wall_time_s"] / fp8["wall_time_s"],
        "peak_memory_reduction_mb": (bf16["worker_peak_memory_mb"] - fp8["worker_peak_memory_mb"]),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n| Mode | Wall time | Denoise | Decode | Peak/GPU |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for mode, record in (("BF16/FP32", bf16), ("Online FP8", fp8)):
        stages = record["stage_durations"]
        print(
            f"| {mode} | {record['wall_time_s']:.2f} s | "
            f"{stages['MiniMaxH3Pipeline.diffuse']:.2f} s | "
            f"{stages['MiniMaxH3Pipeline.decode']:.2f} s | "
            f"{record['worker_peak_memory_mb']:.0f} MB |"
        )
    print(f"\nSpeedup: {comparison['wall_time_speedup']:.2f}x")
    print(f"Peak memory reduction: {comparison['peak_memory_reduction_mb']:.0f} MB/GPU")


def main() -> None:
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("--num-runs must be at least 1")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be at least 1")
    args.output_dir = args.output_dir.resolve()

    worker_mode = os.environ.get(WORKER_MODE_ENV)
    if worker_mode is not None:
        if worker_mode not in {"bf16", "fp8"}:
            raise ValueError(f"Unsupported {WORKER_MODE_ENV}={worker_mode!r}")
        _run_worker(args, worker_mode)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    child_args = [
        sys.executable,
        str(script_path),
        "--model",
        args.model,
        "--output-dir",
        str(args.output_dir),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--duration-seconds",
        str(args.duration_seconds),
        "--num-runs",
        str(args.num_runs),
    ]
    for mode in ("bf16", "fp8"):
        environment = os.environ.copy()
        environment[WORKER_MODE_ENV] = mode
        environment["PYTHONUNBUFFERED"] = "1"
        print(f"\nStarting {mode} benchmark...", flush=True)
        subprocess.run(child_args, check=True, env=environment)
    _write_comparison(args)


if __name__ == "__main__":
    main()
