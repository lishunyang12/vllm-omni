# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 vs FP8 perf/mem/quality bench for HunyuanVideo-1.5 and Wan2.2.

Runs each (model, precision) pair with the same seed, records peak VRAM and
wall time, then computes frame-averaged LPIPS / PSNR / SSIM between the BF16
and FP8 outputs. Emits a markdown table suitable for pasting into a PR.

Target: 1×H100 80GB. Wan2.2 uses the TI2V-5B dense checkpoint (fits in 80GB
BF16); for A14B MoE you need 2×H100 + TP=2 or CPU offload.

Deps:
    pip install lpips scikit-image pynvml

Usage:
    python benchmarks/diffusion/bench_video_fp8.py              # both models
    python benchmarks/diffusion/bench_video_fp8.py --only hv15  # HV-1.5 only
    python benchmarks/diffusion/bench_video_fp8.py --only wan22 # Wan2.2 only
"""

from __future__ import annotations

import argparse
import gc
import threading
import time

import numpy as np
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

CASES = {
    "hv15": dict(
        id="HunyuanVideo-1.5",
        model="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        task="T2V 480x832, 33 frames, 30 steps",
        kwargs=dict(
            height=480,
            width=832,
            num_inference_steps=30,
            num_frames=33,
            guidance_scale=6.0,
        ),
    ),
    "wan22": dict(
        id="Wan2.2 (TI2V 5B)",
        model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        task="T2V 704x1280, 49 frames, 30 steps",
        kwargs=dict(
            height=704,
            width=1280,
            num_inference_steps=30,
            num_frames=49,
            guidance_scale=5.0,
        ),
    ),
}
PROMPT = "A dog running across a field of golden wheat."
SEED = 42


def _extract_video(outputs) -> np.ndarray:
    """Return [T, H, W, 3] float in [0, 1]."""
    from vllm_omni.outputs import OmniRequestOutput

    first = outputs[0]
    if hasattr(first, "request_output") and isinstance(first.request_output, list):
        inner = first.request_output[0]
        frames = inner.images[0] if isinstance(inner, OmniRequestOutput) and inner.images else inner
    elif hasattr(first, "images") and first.images:
        frames = first.images
    else:
        raise ValueError("Cannot extract frames from output.")

    if isinstance(frames, torch.Tensor):
        v = frames.detach().cpu()
        if v.dim() == 5:
            v = v[0]
        if v.dim() == 4 and v.shape[0] in (3, 4):
            v = v.permute(1, 2, 3, 0)
        if v.is_floating_point():
            v = v.clamp(-1, 1) * 0.5 + 0.5
        return v.float().numpy()
    return np.asarray(frames).astype(np.float32) / 255.0


def _nvml_used_gib() -> float:
    """Process-wide GPU-0 used memory in GiB. Captures the worker subprocess too."""
    import pynvml

    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024**3)
    finally:
        pynvml.nvmlShutdown()


class _PeakPoller:
    """Polls process-wide NVML 'used' memory every 100ms, tracks peak."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _PeakPoller:
        def loop() -> None:
            while not self._stop.is_set():
                self.peak = max(self.peak, _nvml_used_gib())
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def run(model: str, quantization: str | None, kwargs: dict) -> tuple[float, float, np.ndarray]:
    """One (model, precision) pair: warmup + timed run. Returns (peak_gib, seconds, video).

    Uses integer `seed=` via sampling params (pickles to the worker subprocess correctly —
    `torch.Generator` handles do not propagate across processes). Peak VRAM is captured via
    pynvml polling so the worker's memory counts.
    """
    omni_kw: dict = {"model": model}
    if quantization:
        omni_kw["quantization"] = quantization
    omni = Omni(**omni_kw)
    try:
        # warmup
        omni.generate(PROMPT, OmniDiffusionSamplingParams(**kwargs, seed=0))
        torch.cuda.synchronize()

        with _PeakPoller() as poll:
            t0 = time.perf_counter()
            outputs = omni.generate(PROMPT, OmniDiffusionSamplingParams(**kwargs, seed=SEED))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        peak = poll.peak
        video = _extract_video(outputs)
        return peak, elapsed, video
    finally:
        del omni
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_metrics(bf16: np.ndarray, fp8: np.ndarray) -> dict[str, float]:
    """Frame-averaged LPIPS (lower better), PSNR (higher better), SSIM (higher better)."""
    import lpips
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    t = min(bf16.shape[0], fp8.shape[0])
    bf16, fp8 = bf16[:t], fp8[:t]

    # LPIPS expects [B, 3, H, W] in [-1, 1]
    lp = lpips.LPIPS(net="alex").cuda().eval()

    def to_lp(x: np.ndarray) -> torch.Tensor:
        return (torch.from_numpy(x).permute(0, 3, 1, 2).cuda() * 2 - 1).float()

    with torch.no_grad():
        lpips_scores = lp(to_lp(bf16), to_lp(fp8)).squeeze().cpu().numpy()
    mean_lpips = float(np.mean(lpips_scores))

    psnrs, ssims = [], []
    for a, b in zip(bf16, fp8, strict=False):
        psnrs.append(peak_signal_noise_ratio(a, b, data_range=1.0))
        ssims.append(structural_similarity(a, b, data_range=1.0, channel_axis=-1))
    return dict(lpips=mean_lpips, psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=list(CASES), default=None)
    args = p.parse_args()
    cases = [CASES[args.only]] if args.only else list(CASES.values())

    rows: list[str] = []
    for c in cases:
        print(f"\n=== {c['id']} BF16 ===", flush=True)
        mem_bf16, t_bf16, v_bf16 = run(c["model"], None, c["kwargs"])
        print(f"  peak {mem_bf16:.2f} GiB, {t_bf16:.2f}s", flush=True)

        print(f"\n=== {c['id']} FP8 ===", flush=True)
        mem_fp8, t_fp8, v_fp8 = run(c["model"], "fp8", c["kwargs"])
        print(f"  peak {mem_fp8:.2f} GiB, {t_fp8:.2f}s", flush=True)

        print("\n  computing LPIPS/PSNR/SSIM...", flush=True)
        m = compute_metrics(v_bf16, v_fp8)
        print(
            f"  LPIPS={m['lpips']:.4f}  PSNR={m['psnr']:.2f}dB  SSIM={m['ssim']:.4f}",
            flush=True,
        )

        mem_red = (1 - mem_fp8 / mem_bf16) if mem_bf16 > 0 else 0.0
        speedup = (1 - t_fp8 / t_bf16) if t_bf16 > 0 else 0.0
        rows.append(
            f"| {c['id']} | {c['task']} "
            f"| {mem_bf16:.2f} GiB | {mem_fp8:.2f} GiB | {mem_red:.0%} "
            f"| {t_bf16:.2f}s | {t_fp8:.2f}s | {speedup:.0%} "
            f"| {m['lpips']:.4f} | {m['psnr']:.2f} | {m['ssim']:.4f} |"
        )

    print("\n\n## Benchmark (H100 80GB × 1)\n")
    print(
        "| Model | Task | Mem BF16 | Mem FP8 | Mem Red | Time BF16 | Time FP8 | Speedup | LPIPS ↓ | PSNR ↑ | SSIM ↑ |"
    )
    print(
        "|-------|------|----------|---------|---------|-----------|----------|---------|---------|--------|--------|"
    )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
