# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare raw LTX audio/video artifacts and their runtime metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lpips-net", choices=("alex", "squeeze", "vgg"), default="alex")
    parser.add_argument("--lpips-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="LPIPS device: auto, cpu, or cuda")
    return parser.parse_args()


def _load_artifact(root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    missing = [name for name in ("video.npy", "audio.npy", "metadata.json") if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifact files in {root}: {', '.join(missing)}")
    video = np.load(root / "video.npy")
    audio = np.load(root / "audio.npy")
    metadata = json.loads((root / "metadata.json").read_text())
    return video, audio, metadata


def _gaussian_kernel(channels: int, *, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    positions = torch.arange(size, dtype=torch.float32) - size // 2
    one_dimensional = torch.exp(-(positions**2) / (2 * sigma**2))
    one_dimensional /= one_dimensional.sum()
    kernel = torch.outer(one_dimensional, one_dimensional)
    return kernel.expand(channels, 1, size, size).contiguous()


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_tensor = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).float()
    candidate_tensor = torch.from_numpy(candidate).permute(2, 0, 1).unsqueeze(0).float()
    channels = reference_tensor.shape[1]
    kernel = _gaussian_kernel(channels)
    padding = kernel.shape[-1] // 2
    reference_mean = F.conv2d(reference_tensor, kernel, padding=padding, groups=channels)
    candidate_mean = F.conv2d(candidate_tensor, kernel, padding=padding, groups=channels)
    reference_variance = F.conv2d(reference_tensor.square(), kernel, padding=padding, groups=channels)
    candidate_variance = F.conv2d(candidate_tensor.square(), kernel, padding=padding, groups=channels)
    covariance = F.conv2d(reference_tensor * candidate_tensor, kernel, padding=padding, groups=channels)
    reference_variance -= reference_mean.square()
    candidate_variance -= candidate_mean.square()
    covariance -= reference_mean * candidate_mean
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * reference_mean * candidate_mean + c1) * (2 * covariance + c2)) / (
        (reference_mean.square() + candidate_mean.square() + c1) * (reference_variance + candidate_variance + c2)
    )
    return float(score.mean())


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2))
    # Exact matches are capped at 120 dB so the emitted JSON remains standards-compliant.
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def _lpips_scores(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    network: str,
    batch_size: int,
    device: str,
) -> list[float]:
    if batch_size < 1:
        raise ValueError("--lpips-batch-size must be positive")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    import lpips

    model = lpips.LPIPS(net=network, verbose=False).eval().to(device)
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, reference.shape[0], batch_size):
            stop = min(start + batch_size, reference.shape[0])
            reference_batch = torch.from_numpy(reference[start:stop]).permute(0, 3, 1, 2).float()
            candidate_batch = torch.from_numpy(candidate[start:stop]).permute(0, 3, 1, 2).float()
            reference_batch = reference_batch.to(device).mul(2.0).sub(1.0)
            candidate_batch = candidate_batch.to(device).mul(2.0).sub(1.0)
            values = model(reference_batch, candidate_batch).flatten().float().cpu().tolist()
            scores.extend(float(value) for value in values)
    return scores


def _summary(scores: list[float]) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _video_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    lpips_net: str,
    lpips_batch_size: int,
    device: str,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Video shape mismatch: {reference.shape} != {candidate.shape}")
    if reference.ndim != 4 or reference.shape[-1] != 3:
        raise ValueError(f"Expected [frames, height, width, 3] video, got {reference.shape}")
    reference = np.clip(reference.astype(np.float32), 0.0, 1.0)
    candidate = np.clip(candidate.astype(np.float32), 0.0, 1.0)
    ssim = [_ssim(first, second) for first, second in zip(reference, candidate, strict=True)]
    psnr = [_psnr(first, second) for first, second in zip(reference, candidate, strict=True)]
    perceptual = _lpips_scores(
        reference,
        candidate,
        network=lpips_net,
        batch_size=lpips_batch_size,
        device=device,
    )
    return {
        "frame_count": int(reference.shape[0]),
        "ssim": {"summary": _summary(ssim), "per_frame": ssim},
        "psnr_db": {"summary": _summary(psnr), "per_frame": psnr},
        "lpips": {"network": lpips_net, "summary": _summary(perceptual), "per_frame": perceptual},
    }


def _canonical_audio(audio: np.ndarray) -> np.ndarray:
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio[0]
    return audio.astype(np.float64)


def _audio_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | bool]:
    reference = _canonical_audio(reference)
    candidate = _canonical_audio(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"Audio shape mismatch: {reference.shape} != {candidate.shape}")
    difference = reference - candidate
    reference_norm = max(float(np.linalg.norm(reference)), 1e-12)
    candidate_norm = max(float(np.linalg.norm(candidate)), 1e-12)
    return {
        "bitwise_equal": bool(np.array_equal(reference, candidate)),
        "relative_l2": float(np.linalg.norm(difference) / reference_norm),
        "cosine_similarity": float(np.vdot(reference.ravel(), candidate.ravel()) / (reference_norm * candidate_norm)),
        "mean_abs": float(np.abs(difference).mean()),
        "max_abs": float(np.abs(difference).max()),
    }


def _performance(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_latency = reference.get("latency_ms", {})
    candidate_latency = candidate.get("latency_ms", {})
    speedup: dict[str, float] = {}
    for phase in ("load", "generation", "e2e"):
        baseline = float(reference_latency.get(phase, 0.0) or 0.0)
        measured = float(candidate_latency.get(phase, 0.0) or 0.0)
        if baseline > 0 and measured > 0:
            speedup[phase] = baseline / measured
    return {
        "reference": {
            "latency_ms": reference_latency,
            "stage_latency_ms": reference.get("stage_latency_ms", {}),
            "peak_gpu_memory_mb": float(reference.get("peak_gpu_memory_mb", 0.0) or 0.0),
        },
        "candidate": {
            "latency_ms": candidate_latency,
            "stage_latency_ms": candidate.get("stage_latency_ms", {}),
            "peak_gpu_memory_mb": float(candidate.get("peak_gpu_memory_mb", 0.0) or 0.0),
        },
        "speedup": speedup,
    }


def main() -> None:
    args = _parse_args()
    reference_video, reference_audio, reference_metadata = _load_artifact(args.reference_dir)
    candidate_video, candidate_audio, candidate_metadata = _load_artifact(args.candidate_dir)
    reference_rate = int(reference_metadata["audio_sample_rate"])
    candidate_rate = int(candidate_metadata["audio_sample_rate"])
    if reference_rate != candidate_rate:
        raise ValueError(f"Audio sample-rate mismatch: {reference_rate} != {candidate_rate}")

    result = {
        "reference_dir": str(args.reference_dir.resolve()),
        "candidate_dir": str(args.candidate_dir.resolve()),
        "video": _video_metrics(
            reference_video,
            candidate_video,
            lpips_net=args.lpips_net,
            lpips_batch_size=args.lpips_batch_size,
            device=args.device,
        ),
        "audio": {"sample_rate": reference_rate, **_audio_metrics(reference_audio, candidate_audio)},
        "performance": _performance(reference_metadata, candidate_metadata),
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
