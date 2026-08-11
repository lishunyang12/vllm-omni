# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run the reproducible LTX-2.5 offline acceleration and quality matrix.

This is an experiment orchestrator, not another inference implementation. Each
child process executes ``run_ltx_reference.py`` after injecting one documented
set of ``Omni`` constructor overrides. Raw outputs are compared by the existing
``compare_ltx_outputs.py`` utility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import runpy
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_RUNNER = SCRIPT_DIR / "run_ltx_reference.py"
COMPARISON_RUNNER = SCRIPT_DIR / "compare_ltx_outputs.py"
DEFAULT_REQUEST = SCRIPT_DIR / "ltx25_distilled_request.json.example"


@dataclass(frozen=True)
class MatrixCase:
    name: str
    description: str
    gpu_count: int
    omni_kwargs: dict[str, Any] = field(default_factory=dict)
    parallel_kwargs: dict[str, Any] = field(default_factory=dict)
    quality_profile: str = "distributed"
    compare_baseline: str = "eager"


CASES = (
    MatrixCase(
        "eager",
        "Single-GPU eager accuracy and latency baseline",
        1,
        {"enforce_eager": True},
        quality_profile="baseline",
    ),
    MatrixCase(
        "regional_compile",
        "Single-GPU regional torch.compile",
        1,
        {"enforce_eager": False, "diffusion_compile_granularity": "regional"},
        quality_profile="strict",
    ),
    MatrixCase(
        "tp2",
        "Two-GPU tensor parallelism",
        2,
        {"enforce_eager": True},
        {"tensor_parallel_size": 2},
    ),
    MatrixCase(
        "tp4",
        "Four-GPU tensor parallelism",
        4,
        {"enforce_eager": True},
        {"tensor_parallel_size": 4},
    ),
    MatrixCase(
        "ulysses2",
        "Two-GPU Ulysses sequence parallelism",
        2,
        {"enforce_eager": True},
        {"ulysses_degree": 2},
    ),
    MatrixCase(
        "ulysses4",
        "Four-GPU Ulysses sequence parallelism",
        4,
        {"enforce_eager": True},
        {"ulysses_degree": 4},
    ),
    MatrixCase(
        "ring2",
        "Two-GPU Ring sequence parallelism",
        2,
        {"enforce_eager": True},
        {"ring_degree": 2},
    ),
    MatrixCase(
        "ring4",
        "Four-GPU Ring sequence parallelism",
        4,
        {"enforce_eager": True},
        {"ring_degree": 4},
    ),
    MatrixCase(
        "hsdp2",
        "Two-GPU standalone HSDP weight sharding",
        2,
        {"enforce_eager": True},
        {"use_hsdp": True, "hsdp_shard_size": 2},
    ),
    MatrixCase(
        "vae_tiling",
        "Single-GPU VAE tiled decode",
        1,
        {"enforce_eager": True, "vae_use_tiling": True},
        quality_profile="vae",
    ),
    MatrixCase(
        "vae_patch2",
        "Two-GPU Ulysses plus distributed VAE patch decode",
        2,
        {"enforce_eager": True, "vae_use_tiling": True},
        {"ulysses_degree": 2, "vae_patch_parallel_size": 2},
        quality_profile="vae",
        compare_baseline="ulysses2",
    ),
    MatrixCase(
        "model_offload",
        "Single-GPU sequential model CPU offload",
        1,
        {"enforce_eager": True, "enable_cpu_offload": True},
        quality_profile="strict",
    ),
    MatrixCase(
        "layerwise_offload",
        "Single-GPU asynchronous blockwise CPU offload",
        1,
        {"enforce_eager": True, "enable_layerwise_offload": True},
        quality_profile="strict",
    ),
    MatrixCase(
        "fp8",
        "Single-GPU online FP8 W8A8 quantization",
        1,
        {"enforce_eager": True, "quantization": "fp8"},
        quality_profile="fp8",
    ),
    MatrixCase(
        "cache_dit",
        "Single-GPU conservative Cache-DiT for the eight-step distilled recipe",
        1,
        {
            "enforce_eager": True,
            "cache_backend": "cache_dit",
            "enable_cache_dit_summary": True,
            "cache_config": {
                "Fn_compute_blocks": 1,
                "Bn_compute_blocks": 0,
                "max_warmup_steps": 4,
                "residual_diff_threshold": 0.12,
                "max_continuous_cached_steps": 2,
                "enable_taylorseer": False,
                "scm_steps_mask_policy": None,
            },
        },
        quality_profile="cache",
    ),
    MatrixCase(
        "compile_ulysses4_vae_patch4",
        "Experimental regional compile, Ulysses4, and VAE patch4 stack",
        4,
        {
            "enforce_eager": False,
            "diffusion_compile_granularity": "regional",
            "vae_use_tiling": True,
        },
        {"ulysses_degree": 4, "vae_patch_parallel_size": 4},
        quality_profile="vae",
        compare_baseline="ulysses4",
    ),
)

SKIPPED_ACCELERATIONS = (
    {
        "name": "cfg_parallel",
        "reason": (
            "The official distilled request is positive-only (CFG scale 1); "
            "there is no second CFG branch to parallelize."
        ),
    },
    {
        "name": "tea_cache",
        "reason": "LTX-2.5 has no validated TeaCache residual extractor/coefficient profile in vLLM-Omni.",
    },
    {
        "name": "gguf",
        "reason": "No official LTX-2.5 GGUF checkpoint and no validated GGUF component adapter are available.",
    },
)

QUALITY_GATES = {
    "strict": {"ssim_min": 0.999, "psnr_min": 50.0, "lpips_max": 0.005, "audio_cosine_min": 0.999},
    "distributed": {"ssim_min": 0.995, "psnr_min": 40.0, "lpips_max": 0.01, "audio_cosine_min": 0.995},
    "vae": {"ssim_min": 0.99, "psnr_min": 35.0, "lpips_max": 0.02, "audio_cosine_min": 0.995},
    "fp8": {"ssim_min": 0.90, "psnr_min": 22.0, "lpips_max": 0.10, "audio_cosine_min": 0.90},
    "cache": {"ssim_min": 0.95, "psnr_min": 28.0, "lpips_max": 0.05, "audio_cosine_min": 0.95},
}

SUMMARY_FIELDS = (
    "case",
    "status",
    "gpu_count",
    "successful_repeats",
    "failed_repeats",
    "generation_ms_median",
    "e2e_ms_median",
    "peak_gpu_memory_mb_median",
    "speedup_vs_eager_generation",
    "speedup_vs_eager_e2e",
    "comparison_baseline",
    "speedup_vs_comparison_generation",
    "speedup_vs_comparison_e2e",
    "ssim_vs_official_median",
    "psnr_vs_official_median",
    "lpips_vs_official_median",
    "audio_cosine_vs_official_median",
    "ssim_vs_eager_median",
    "ssim_vs_comparison_median",
    "psnr_vs_comparison_median",
    "lpips_vs_comparison_median",
    "audio_cosine_vs_comparison_median",
    "psnr_vs_eager_median",
    "lpips_vs_eager_median",
    "audio_cosine_vs_eager_median",
    "quality_gate",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3", help="Ordered physical GPU IDs, for example 4,6,5,7")
    parser.add_argument("--cases", default="all", help="Comma-separated case names, or all")
    parser.add_argument("--mode", choices=("distilled-one-stage", "distilled-two-stage"), default="distilled-one-stage")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--metrics-device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-warmup-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def _case_map() -> dict[str, MatrixCase]:
    return {case.name: case for case in CASES}


def _selected_cases(value: str) -> list[MatrixCase]:
    available = _case_map()
    if value == "all":
        return list(CASES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"Unknown cases: {', '.join(unknown)}")
    required = {"eager", *names}
    while True:
        dependencies = {available[name].compare_baseline for name in required}
        expanded = required | dependencies
        if expanded == required:
            break
        required = expanded
    return [case for case in CASES if case.name in required]


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain one or more unique comma-separated GPU IDs")
    return gpus


def _request_snapshot(request_path: Path, output_root: Path, *, dry_run: bool, resume: bool) -> tuple[Path, str]:
    request = json.loads(request_path.read_text())
    required = {"prompt", "seed", "height", "width", "num_frames", "fps", "num_inference_steps"}
    missing = sorted(required - request.keys())
    if missing:
        raise ValueError(f"Request is missing fixed benchmark fields: {', '.join(missing)}")
    serialized = json.dumps(request, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    snapshot = output_root / "request.snapshot.json"
    if dry_run:
        return request_path.resolve(), digest
    output_root.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and snapshot.read_text() != serialized:
        raise ValueError(f"Existing request snapshot differs: {snapshot}")
    if snapshot.exists() and not resume:
        raise FileExistsError(f"Output root already has a request snapshot; pass --resume: {output_root}")
    snapshot.write_text(serialized)
    return snapshot, digest


def _command_text(command: list[str], env: dict[str, str]) -> str:
    prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])}"
    return f"{prefix} {shlex.join(command)}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _run_command(command: list[str], *, env: dict[str, str], cwd: Path, log: Path) -> tuple[int, float]:
    start = time.perf_counter()
    with log.open("w") as stream:
        process = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    return process.returncode, (time.perf_counter() - start) * 1000.0


def _nested_float(value: dict[str, Any] | None, *keys: str) -> float | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _metric_values(metrics: dict[str, Any] | None) -> dict[str, float | None]:
    return {
        "ssim": _nested_float(metrics, "video", "ssim", "summary", "mean"),
        "psnr": _nested_float(metrics, "video", "psnr_db", "summary", "mean"),
        "lpips": _nested_float(metrics, "video", "lpips", "summary", "mean"),
        "audio_cosine": _nested_float(metrics, "audio", "cosine_similarity"),
    }


def _comparison_is_identity(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    case_name: str | None = None,
    compare_baseline: str | None = None,
) -> bool:
    return (case_name is not None and case_name == compare_baseline) or (
        reference_dir.resolve() == candidate_dir.resolve()
    )


def _identity_quality_metrics(
    reference_dir: Path,
    candidate_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    video_shape = metadata.get("video_shape", ())
    frame_count = int(video_shape[0]) if isinstance(video_shape, (list, tuple)) and video_shape else 0
    ones = [1.0] * frame_count
    psnr = [120.0] * frame_count
    zeros = [0.0] * frame_count
    latency = metadata.get("latency_ms", {})
    performance = {
        "latency_ms": latency,
        "stage_latency_ms": metadata.get("stage_latency_ms", {}),
        "peak_gpu_memory_mb": float(metadata.get("peak_gpu_memory_mb", 0.0) or 0.0),
    }
    return {
        "reference_dir": str(reference_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "identity": True,
        "comparison_skipped": "reference and candidate are the same artifact set",
        "video": {
            "frame_count": frame_count,
            "ssim": {"summary": {"mean": 1.0, "p95": 1.0, "max": 1.0}, "per_frame": ones},
            "psnr_db": {"summary": {"mean": 120.0, "p95": 120.0, "max": 120.0}, "per_frame": psnr},
            "lpips": {
                "network": "alex",
                "summary": {"mean": 0.0, "p95": 0.0, "max": 0.0},
                "per_frame": zeros,
            },
        },
        "audio": {
            "sample_rate": int(metadata.get("audio_sample_rate", 0) or 0),
            "bitwise_equal": True,
            "relative_l2": 0.0,
            "cosine_similarity": 1.0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
        },
        "performance": {
            "reference": performance,
            "candidate": performance,
            "speedup": {phase: 1.0 for phase in ("load", "generation", "e2e") if latency.get(phase)},
        },
    }


def _quality_gate(profile: str, metrics: dict[str, Any] | None) -> str:
    if profile == "baseline":
        return "baseline"
    values = _metric_values(metrics)
    if any(value is None for value in values.values()):
        return "not_evaluated"
    gate = QUALITY_GATES[profile]
    passed = (
        values["ssim"] >= gate["ssim_min"]
        and values["psnr"] >= gate["psnr_min"]
        and values["lpips"] <= gate["lpips_max"]
        and values["audio_cosine"] >= gate["audio_cosine_min"]
    )
    return "pass" if passed else "fail"


def _median(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(statistics.median(present)) if present else None


def _summarize_case(case: MatrixCase, repeats: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [repeat for repeat in repeats if repeat.get("status") == "success"]
    official_values = [_metric_values(repeat.get("metrics_vs_official")) for repeat in successful]
    eager_values = [_metric_values(repeat.get("metrics_vs_eager")) for repeat in successful]
    metadata = [repeat.get("metadata", {}) for repeat in successful]
    comparison_values = [_metric_values(repeat.get("metrics_vs_comparison")) for repeat in successful]
    gates = [repeat.get("quality_gate") for repeat in successful]
    if not successful:
        status = "failed"
    elif len(successful) < len(repeats):
        status = "partial"
    else:
        status = "success"
    return {
        "case": case.name,
        "status": status,
        "description": case.description,
        "gpu_count": case.gpu_count,
        "successful_repeats": len(successful),
        "failed_repeats": len(repeats) - len(successful),
        "generation_ms_median": _median([_nested_float(item, "latency_ms", "generation") for item in metadata]),
        "e2e_ms_median": _median([_nested_float(item, "latency_ms", "e2e") for item in metadata]),
        "peak_gpu_memory_mb_median": _median([_nested_float(item, "peak_gpu_memory_mb") for item in metadata]),
        "speedup_vs_eager_generation": None,
        "speedup_vs_eager_e2e": None,
        "ssim_vs_official_median": _median([item["ssim"] for item in official_values]),
        "comparison_baseline": case.compare_baseline,
        "speedup_vs_comparison_generation": None,
        "speedup_vs_comparison_e2e": None,
        "psnr_vs_official_median": _median([item["psnr"] for item in official_values]),
        "lpips_vs_official_median": _median([item["lpips"] for item in official_values]),
        "audio_cosine_vs_official_median": _median([item["audio_cosine"] for item in official_values]),
        "ssim_vs_eager_median": _median([item["ssim"] for item in eager_values]),
        "psnr_vs_eager_median": _median([item["psnr"] for item in eager_values]),
        "lpips_vs_eager_median": _median([item["lpips"] for item in eager_values]),
        "audio_cosine_vs_eager_median": _median([item["audio_cosine"] for item in eager_values]),
        "quality_gate": "fail" if "fail" in gates else (gates[0] if gates else "not_evaluated"),
        "quality_profile": case.quality_profile,
        "compare_baseline": case.compare_baseline,
        "omni_kwargs": case.omni_kwargs,
        "parallel_kwargs": case.parallel_kwargs,
        "ssim_vs_comparison_median": _median([item["ssim"] for item in comparison_values]),
        "psnr_vs_comparison_median": _median([item["psnr"] for item in comparison_values]),
        "lpips_vs_comparison_median": _median([item["lpips"] for item in comparison_values]),
        "audio_cosine_vs_comparison_median": _median([item["audio_cosine"] for item in comparison_values]),
        "repeats": repeats,
    }


def _add_speedups(rows: list[dict[str, Any]]) -> None:
    by_name = {row["case"]: row for row in rows}
    eager = by_name.get("eager")
    for row in rows:
        for phase in ("generation", "e2e"):
            measured = row.get(f"{phase}_ms_median")
            if eager is not None:
                eager_latency = eager.get(f"{phase}_ms_median")
                if eager_latency and measured:
                    row[f"speedup_vs_eager_{phase}"] = eager_latency / measured
            comparison = by_name.get(row.get("comparison_baseline"))
            if comparison is not None:
                comparison_latency = comparison.get(f"{phase}_ms_median")
                if comparison_latency and measured:
                    row[f"speedup_vs_comparison_{phase}"] = comparison_latency / measured


def _run_comparison(
    args: argparse.Namespace,
    *,
    reference_dir: Path,
    candidate_dir: Path,
    output: Path,
    env: dict[str, str],
    log: Path,
) -> int:
    command = [
        str(args.python),
        str(COMPARISON_RUNNER),
        "--reference-dir",
        str(reference_dir),
        "--candidate-dir",
        str(candidate_dir),
        "--output",
        str(output),
        "--device",
        args.metrics_device,
    ]
    return _run_command(command, env=env, cwd=SCRIPT_DIR.parents[3], log=log)[0]


def _run_or_identity_comparison(
    args: argparse.Namespace,
    *,
    reference_dir: Path,
    candidate_dir: Path,
    output: Path,
    env: dict[str, str],
    log: Path,
    metadata: dict[str, Any],
    case_name: str | None = None,
    compare_baseline: str | None = None,
) -> tuple[int, dict[str, Any] | None]:
    if _comparison_is_identity(
        reference_dir,
        candidate_dir,
        case_name=case_name,
        compare_baseline=compare_baseline,
    ):
        metrics = _identity_quality_metrics(reference_dir, candidate_dir, metadata)
        _write_json(output, metrics)
        log.write_text("Skipped compare_ltx_outputs.py: identity comparison.\n")
        return 0, metrics
    returncode = _run_comparison(
        args,
        reference_dir=reference_dir,
        candidate_dir=candidate_dir,
        output=output,
        env=env,
        log=log,
    )
    return returncode, json.loads(output.read_text()) if returncode == 0 else None


def _run_one(
    args: argparse.Namespace,
    *,
    case: MatrixCase,
    kind: str,
    index: int,
    request: Path,
    selected_gpus: list[str],
    eager_reference: Path | None,
    comparison_reference: Path | None,
) -> dict[str, Any]:
    run_dir = args.output_root / case.name / (f"{kind}-{index:02d}")
    status_path = run_dir / "status.json"
    if args.resume and status_path.is_file():
        existing = json.loads(status_path.read_text())
        if existing.get("status") == "success":
            return existing
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run output already exists; pass --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir / "artifacts"
    runner_args = [
        "--backend",
        "omni",
        "--mode",
        args.mode,
        "--request",
        str(request),
        "--model",
        str(args.model.resolve()),
        "--output-dir",
        str(output_dir),
        "--omni-attention-backend",
        "CUDNN_ATTN",
    ]
    if args.image is not None:
        runner_args.extend(("--image", str(args.image.resolve())))
    case_config = {
        "case": asdict(case),
        "kind": kind,
        "index": index,
        "runner": str(REFERENCE_RUNNER),
        "runner_args": runner_args,
        "output_dir": str(output_dir),
    }
    case_config_path = run_dir / "case-config.json"
    _write_json(case_config_path, case_config)
    command = [str(args.python), str(Path(__file__).resolve()), "--run-case-config", str(case_config_path)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
    repo_root = SCRIPT_DIR.parents[3]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo_root), env.get("PYTHONPATH", ""))))
    returncode, wall_ms = _run_command(command, env=env, cwd=repo_root, log=run_dir / "runner.log")
    result: dict[str, Any] = {
        "case": case.name,
        "kind": kind,
        "index": index,
        "status": "failed" if returncode else "success",
        "returncode": returncode,
        "wall_ms": wall_ms,
        "gpus": selected_gpus,
        "command": _command_text(command, env),
    }
    metadata_path = output_dir / "metadata.json"
    if returncode == 0 and metadata_path.is_file():
        result["metadata"] = json.loads(metadata_path.read_text())
    elif returncode == 0:
        result["status"] = "failed"
        result["error"] = "Runner returned success without metadata.json"

    if kind == "repeat" and result["status"] == "success":
        official_output = run_dir / "metrics-vs-official.json"
        compare_code = _run_comparison(
            args,
            reference_dir=args.reference_dir,
            candidate_dir=output_dir,
            output=official_output,
            env=env,
            log=run_dir / "compare-vs-official.log",
        )
        if compare_code == 0:
            result["metrics_vs_official"] = json.loads(official_output.read_text())
        else:
            result["status"] = "failed"
            result["error"] = f"Official comparison failed with exit code {compare_code}"

        quality_reference = eager_reference if eager_reference is not None else output_dir
        eager_output = run_dir / "metrics-vs-eager.json"
        compare_code, metrics = _run_or_identity_comparison(
            args,
            reference_dir=quality_reference,
            candidate_dir=output_dir,
            output=eager_output,
            env=env,
            log=run_dir / "compare-vs-eager.log",
            metadata=result["metadata"],
            case_name=case.name,
            compare_baseline=case.compare_baseline,
        )
        if compare_code == 0:
            result["metrics_vs_eager"] = metrics
        else:
            result["status"] = "failed"
            result["error"] = f"Eager comparison failed with exit code {compare_code}"

        matched_reference = comparison_reference if comparison_reference is not None else quality_reference
        if _comparison_is_identity(matched_reference, quality_reference):
            result["metrics_vs_comparison"] = result.get("metrics_vs_eager")
        else:
            matched_output = run_dir / "metrics-vs-comparison.json"
            compare_code, metrics = _run_or_identity_comparison(
                args,
                reference_dir=matched_reference,
                candidate_dir=output_dir,
                output=matched_output,
                env=env,
                log=run_dir / "compare-vs-comparison.log",
                metadata=result["metadata"],
                case_name=case.name,
                compare_baseline=case.compare_baseline,
            )
            if compare_code == 0:
                result["metrics_vs_comparison"] = metrics
            else:
                result["status"] = "failed"
                result["error"] = f"Matched-baseline comparison failed with exit code {compare_code}"
        result["quality_gate"] = _quality_gate(case.quality_profile, result.get("metrics_vs_comparison"))

    _write_json(status_path, result)
    if kind == "warmup" and not args.keep_warmup_artifacts:
        for name in ("video.npy", "audio.npy"):
            artifact = output_dir / name
            if artifact.is_file():
                artifact.unlink()
    return result


def _dry_run(args: argparse.Namespace, cases: list[MatrixCase], gpus: list[str], request: Path) -> None:
    print(f"request={request}")
    print(f"reference={args.reference_dir.resolve()}")
    print("attention_backend=CUDNN_ATTN")
    for case in cases:
        selected = gpus[: case.gpu_count]
        if len(selected) < case.gpu_count:
            print(f"SKIP {case.name}: requires {case.gpu_count} GPUs, only {len(gpus)} supplied")
            continue
        print(f"CASE {case.name}: GPUs={','.join(selected)}")
        print(f"  omni_kwargs={json.dumps(case.omni_kwargs, sort_keys=True)}")
        print(f"  parallel_kwargs={json.dumps(case.parallel_kwargs, sort_keys=True)}")
        for kind, count in (("warmup", args.warmups), ("repeat", args.repeats)):
            for index in range(1, count + 1):
                suffix = "" if kind == "warmup" else ", then compare_ltx_outputs.py"
                print(f"  {kind}-{index:02d}: execute run_ltx_reference.py{suffix}")
    for skipped in SKIPPED_ACCELERATIONS:
        print(f"SKIP {skipped['name']}: {skipped['reason']}")


def _write_summary(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    _write_json(args.output_root / "summary.json", payload)
    with (args.output_root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["cases"])


def _run_matrix(args: argparse.Namespace) -> int:
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("--warmups must be non-negative and --repeats must be positive")
    cases = _selected_cases(args.cases)
    gpus = _parse_gpus(args.gpus)
    request, request_hash = _request_snapshot(
        args.request.resolve(), args.output_root.resolve(), dry_run=args.dry_run, resume=args.resume
    )
    args.output_root = args.output_root.resolve()
    args.reference_dir = args.reference_dir.resolve()
    if args.dry_run:
        _dry_run(args, cases, gpus, request)
        return 0
    if not args.reference_dir.is_dir():
        raise FileNotFoundError(f"Reference artifact directory not found: {args.reference_dir}")

    case_summaries: list[dict[str, Any]] = []
    eager_reference: Path | None = None
    case_references: dict[str, Path] = {}
    for case in cases:
        if len(gpus) < case.gpu_count:
            case_summaries.append(
                {
                    "case": case.name,
                    "status": "skipped",
                    "description": case.description,
                    "gpu_count": case.gpu_count,
                    "successful_repeats": 0,
                    "failed_repeats": 0,
                    "quality_gate": "not_evaluated",
                    "reason": f"Requires {case.gpu_count} GPUs, but --gpus supplied {len(gpus)}",
                }
            )
            continue
        selected_gpus = gpus[: case.gpu_count]
        for index in range(1, args.warmups + 1):
            try:
                _run_one(
                    args,
                    case=case,
                    kind="warmup",
                    index=index,
                    request=request,
                    selected_gpus=selected_gpus,
                    eager_reference=eager_reference,
                    comparison_reference=case_references.get(case.compare_baseline),
                )
            except Exception as exc:
                print(f"{case.name} warmup-{index:02d} failed: {exc}", file=sys.stderr)

        repeats: list[dict[str, Any]] = []
        for index in range(1, args.repeats + 1):
            try:
                result = _run_one(
                    args,
                    case=case,
                    kind="repeat",
                    index=index,
                    request=request,
                    selected_gpus=selected_gpus,
                    eager_reference=eager_reference,
                    comparison_reference=case_references.get(case.compare_baseline),
                )
            except Exception as exc:
                result = {"case": case.name, "kind": "repeat", "index": index, "status": "failed", "error": str(exc)}
            repeats.append(result)
            if case.name == "eager" and result.get("status") == "success" and eager_reference is None:
                eager_reference = args.output_root / case.name / f"repeat-{index:02d}" / "artifacts"
            if result.get("status") == "success" and case.name not in case_references:
                case_references[case.name] = args.output_root / case.name / f"repeat-{index:02d}" / "artifacts"
        case_summaries.append(_summarize_case(case, repeats))
        _add_speedups(case_summaries)
        _write_summary(
            args,
            {
                "model": str(args.model.resolve()),
                "mode": args.mode,
                "request_sha256": request_hash,
                "reference_dir": str(args.reference_dir),
                "warmups": args.warmups,
                "repeats": args.repeats,
                "gpus": gpus,
                "quality_gates_vs_comparison_baseline": QUALITY_GATES,
                "skipped_accelerations": SKIPPED_ACCELERATIONS,
                "cases": case_summaries,
            },
        )
    return 0 if all(row["status"] in {"success", "skipped"} for row in case_summaries) else 1


def _run_case_from_config(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    case = config["case"]
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints import omni as omni_module

    original_omni = omni_module.Omni

    def configured_omni(*omni_args: Any, **omni_kwargs: Any) -> Any:
        omni_kwargs.update(case["omni_kwargs"])
        omni_kwargs["parallel_config"] = DiffusionParallelConfig(**case["parallel_kwargs"])
        return original_omni(*omni_args, **omni_kwargs)

    omni_module.Omni = configured_omni
    original_argv = sys.argv
    try:
        sys.argv = [config["runner"], *config["runner_args"]]
        runpy.run_path(config["runner"], run_name="__main__")
    finally:
        sys.argv = original_argv
        omni_module.Omni = original_omni

    metadata_path = Path(config["output_dir"]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["acceleration_case"] = case["name"]
    metadata["acceleration_omni_kwargs"] = case["omni_kwargs"]
    metadata["acceleration_parallel_kwargs"] = case["parallel_kwargs"]
    metadata["benchmark_kind"] = config["kind"]
    metadata["benchmark_index"] = config["index"]
    _write_json(metadata_path, metadata)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--run-case-config":
        _run_case_from_config(Path(sys.argv[2]))
        return 0
    args = _parse_args()
    if args.list_cases:
        for case in CASES:
            print(f"{case.name:36} {case.gpu_count} GPU  {case.description}")
        return 0
    return _run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
