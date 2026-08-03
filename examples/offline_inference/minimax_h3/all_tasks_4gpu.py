# SPDX-License-Identifier: Apache-2.0
"""Run every MiniMax-H3 task currently supported by vLLM-Omni.

The shell wrapper launches this file once per checkpoint partition. Keeping
FL2VA and Ref2VA in separate processes avoids reinitializing distributed
process groups after an ``Omni`` engine has been closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

TASK_IDS = (
    "t2va",
    "fl2va_first_frame",
    "ref2va_image_audio",
    "ref2va_two_videos",
)

DEFAULT_PROMPTS = {
    "t2va": (
        "At night, three cats march into a bedroom playing tiny brass "
        "instruments, then abruptly file out, with synchronized room ambience."
    ),
    "fl2va_first_frame": (
        "Continue naturally from the supplied first frame. The cats march "
        "forward while playing, with coherent motion and synchronized sound."
    ),
    "ref2va_image_audio": (
        "Use Picture 1 as the visual subject and Audio 1 as the sound reference. "
        "Create coherent natural motion synchronized to the complete audio."
    ),
    "ref2va_two_videos": (
        "Combine the subjects and motion of Video 1 with the continuation in "
        "Video 2, preserving coherent timing and synchronized sound."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("fl2va", "ref2va"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=1101)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_metadata() -> list[dict[str, object]]:
    hardware = []
    for index in range(torch.accelerator.device_count()):
        properties = torch.cuda.get_device_properties(index)
        hardware.append(
            {
                "logical_index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_gib": round(properties.total_memory / 2**30, 2),
            }
        )
    if len(hardware) != 4:
        raise RuntimeError(f"Expected four visible GPUs, found {len(hardware)}")
    return hardware


def make_engine(model_dir: Path, *, enforce_eager: bool) -> Omni:
    return Omni(
        model=str(model_dir),
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
        enforce_eager=enforce_eager,
        diffusion_attention_backend="CUDNN_ATTN",
        enable_diffusion_pipeline_profiler=True,
    )


def sampling_params(
    args: argparse.Namespace,
    *,
    task: str,
    seed: int,
) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        fps=24,
        num_inference_steps=args.num_inference_steps,
        seed=seed,
        output_type="np",
        extra_args={
            "task": task,
            "duration": args.duration,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        },
    )


def prompt_for(task_id: str) -> str:
    environment_key = f"MINIMAX_H3_{task_id.upper()}_PROMPT"
    return os.environ.get(environment_key, DEFAULT_PROMPTS[task_id])


def save_first_frame(frames: np.ndarray, output_path: Path) -> None:
    from PIL import Image

    first_frame = np.asarray(frames[0])
    if np.issubdtype(first_frame.dtype, np.floating):
        first_frame = np.clip(first_frame, 0.0, 1.0) * 255.0
    Image.fromarray(first_frame.astype(np.uint8)).save(output_path)


def extract_reference_audio(video_path: Path, audio_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(audio_path),
        ],
        check=True,
    )


def run_task(
    engine: Omni,
    args: argparse.Namespace,
    *,
    task_id: str,
    task: str,
    prompt: str | dict[str, Any],
    prompt_text: str,
    seed: int,
    output_path: Path,
) -> tuple[dict[str, object], np.ndarray]:
    started = time.perf_counter()
    outputs = engine.generate(
        prompt,
        sampling_params(args, task=task, seed=seed),
        use_tqdm=False,
    )
    wall_time = time.perf_counter() - started
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one output for {task_id}, found {len(outputs)}")

    result = outputs[0]
    frames = np.asarray(result.images[0])
    multimodal = result.multimodal_output
    if multimodal is None:
        raise RuntimeError(f"{task_id} returned no audio metadata")
    audio = np.asarray(multimodal["audio"])
    fps = int(multimodal["fps"])
    sample_rate = int(multimodal["audio_sample_rate"])

    if frames.ndim != 4 or tuple(frames.shape[1:]) != (
        args.height,
        args.width,
        3,
    ):
        raise RuntimeError(f"Unexpected {task_id} video shape: {frames.shape}")
    if audio.ndim not in (2, 3) or 2 not in audio.shape:
        raise RuntimeError(f"Unexpected {task_id} audio shape: {audio.shape}")
    if fps != 24 or sample_rate != 32000:
        raise RuntimeError(f"Unexpected {task_id} media rates: fps={fps}, audio={sample_rate}")

    output_path.write_bytes(
        _encode_video_bytes(
            frames,
            fps=fps,
            audio=audio,
            audio_sample_rate=sample_rate,
        )
    )
    record: dict[str, object] = {
        "task_id": task_id,
        "task": task,
        "partition": args.partition,
        "prompt": prompt_text,
        "seed": seed,
        "wall_time_s": round(wall_time, 4),
        "stage_durations": dict(getattr(result, "stage_durations", {}) or {}),
        "worker_peak_memory_mb": float(getattr(result, "peak_memory_mb", 0.0) or 0.0),
        "frames_shape": list(frames.shape),
        "audio_shape": list(audio.shape),
        "fps": fps,
        "audio_sample_rate": sample_rate,
        "frames_sha256": array_sha256(frames),
        "audio_sha256": array_sha256(audio),
        "mp4_sha256": file_sha256(output_path),
        "output": str(output_path),
        "completed_at": utc_now(),
    }
    print("TASK_RESULT " + json.dumps(record, sort_keys=True), flush=True)
    return record, frames


def update_summary(
    args: argparse.Namespace,
    *,
    hardware: list[dict[str, object]],
    records: list[dict[str, object]],
) -> None:
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "started_at": utc_now(),
            "expected_tasks": list(TASK_IDS),
            "tasks": [],
        }

    previous = {record["task_id"]: record for record in summary["tasks"]}
    previous.update({record["task_id"]: record for record in records})
    summary.update(
        {
            "updated_at": utc_now(),
            "status": ("completed" if all(task_id in previous for task_id in TASK_IDS) else "in_progress"),
            "model_root": str(args.model_root),
            "hardware": hardware,
            "torch_version": torch.__version__,
            "parallel_config": ("tp4_ulysses1_ring1_text_encoder_tp4_vae_tile4"),
            "attention_backend": "CUDNN_ATTN",
            "precision": "checkpoint BF16/FP32",
            "regional_compile": not args.enforce_eager,
            "height": args.height,
            "width": args.width,
            "duration_seconds": args.duration,
            "num_inference_steps": args.num_inference_steps,
            "tasks": [previous[task_id] for task_id in TASK_IDS if task_id in previous],
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_fl2va_partition(
    engine: Omni,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    t2va_path = args.output_dir / "01_t2va.mp4"
    first_frame_path = args.output_dir / "t2va_first_frame.png"
    reference_audio_path = args.output_dir / "t2va_reference_audio.wav"
    fl2va_path = args.output_dir / "02_fl2va_first_frame.mp4"

    t2va_prompt = prompt_for("t2va")
    t2va, frames = run_task(
        engine,
        args,
        task_id="t2va",
        task="t2va",
        prompt=t2va_prompt,
        prompt_text=t2va_prompt,
        seed=args.seed_base,
        output_path=t2va_path,
    )
    save_first_frame(frames, first_frame_path)
    extract_reference_audio(t2va_path, reference_audio_path)

    fl2va_prompt = prompt_for("fl2va_first_frame")
    fl2va, _ = run_task(
        engine,
        args,
        task_id="fl2va_first_frame",
        task="fl2va",
        prompt={
            "prompt": fl2va_prompt,
            "multi_modal_data": {"image": str(first_frame_path)},
        },
        prompt_text=fl2va_prompt,
        seed=args.seed_base + 1000,
        output_path=fl2va_path,
    )
    return [t2va, fl2va]


def require_generated_assets(output_dir: Path) -> dict[str, Path]:
    assets = {
        "first_frame": output_dir / "t2va_first_frame.png",
        "reference_audio": output_dir / "t2va_reference_audio.wav",
        "t2va_video": output_dir / "01_t2va.mp4",
        "fl2va_video": output_dir / "02_fl2va_first_frame.mp4",
    }
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Ref2VA phase requires the FL2VA phase outputs; missing: " + ", ".join(missing))
    return assets


def run_ref2va_partition(
    engine: Omni,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    assets = require_generated_assets(args.output_dir)
    image_audio_path = args.output_dir / "03_ref2va_image_audio.mp4"
    two_videos_path = args.output_dir / "04_ref2va_two_videos.mp4"

    image_audio_prompt = prompt_for("ref2va_image_audio")
    image_audio, _ = run_task(
        engine,
        args,
        task_id="ref2va_image_audio",
        task="ref2va",
        prompt={
            "prompt": image_audio_prompt,
            "multi_modal_data": {
                "image": str(assets["first_frame"]),
                "audio": str(assets["reference_audio"]),
            },
        },
        prompt_text=image_audio_prompt,
        seed=args.seed_base + 2000,
        output_path=image_audio_path,
    )

    two_videos_prompt = prompt_for("ref2va_two_videos")
    two_videos, _ = run_task(
        engine,
        args,
        task_id="ref2va_two_videos",
        task="ref2va",
        prompt={
            "prompt": two_videos_prompt,
            "multi_modal_data": {
                "video": [
                    str(assets["t2va_video"]),
                    str(assets["fl2va_video"]),
                ]
            },
        },
        prompt_text=two_videos_prompt,
        seed=args.seed_base + 3000,
        output_path=two_videos_path,
    )
    return [image_audio, two_videos]


def main() -> None:
    args = parse_args()
    args.model_root = args.model_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    partition_name = "FL2VA" if args.partition == "fl2va" else "Ref2VA"
    model_dir = args.model_root / partition_name
    if not (model_dir / "model_index.json").is_file():
        raise FileNotFoundError(
            f"Missing {partition_name} checkpoint at {model_dir}. "
            "Download both MiniMax-H3 partitions before running all tasks."
        )

    hardware = hardware_metadata()
    engine = make_engine(model_dir, enforce_eager=args.enforce_eager)
    try:
        if args.partition == "fl2va":
            records = run_fl2va_partition(engine, args)
        else:
            records = run_ref2va_partition(engine, args)
    finally:
        engine.close()

    update_summary(args, hardware=hardware, records=records)


if __name__ == "__main__":
    main()
