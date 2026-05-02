#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure per-layer activation amax variance across denoising timesteps.

Runs a single BF16 inference pass with forward hooks on every Linear layer
and reports the ratio between the minimum and maximum amax observed across
all denoising steps.  A high ratio (>3x) means a single static calibration
scale cannot cover the full dynamic range at every timestep -- the layer is
a poor candidate for W4A4 (NVFP4 default) and should use W4A8 or be kept
full-precision.

Tuned for Wan2.2 T2V A14B at 720x1280 / 81 frames by default, but the
constants at the top of the file can be changed for any diffusers pipeline.

Example:
    python examples/quantization/check_activation_variance.py \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --output wan22_a14b_activation_variance.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

DEFAULT_PROMPT = "A dog running across a field of golden wheat."


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Diffusers model id or local directory.")
    p.add_argument("--output", default="activation_variance.csv", help="CSV output path.")
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--num-inference-steps", type=int, default=40)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument(
        "--min-hits",
        type=int,
        default=None,
        help="Minimum forward-pass hits to include a layer. Defaults to "
             "--num-inference-steps (filters sparse MoE experts that fire "
             "on fewer than 1 pass per step).",
    )
    p.add_argument(
        "--ratio-threshold",
        type=float,
        default=3.0,
        help="Layers with amax ratio above this are flagged in console output (default 3.0).",
    )
    return p


def _collect_stats(pipe: DiffusionPipeline, args: argparse.Namespace) -> dict[str, list[float]]:
    stats: dict[str, list[float]] = defaultdict(list)

    def make_hook(name: str):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor) and out.numel() > 0:
                stats[name].append(out.detach().float().abs().max().item())
        return hook

    handles = [
        mod.register_forward_hook(make_hook(name))
        for name, mod in pipe.transformer.named_modules()
        if isinstance(mod, torch.nn.Linear)
    ]

    call_kwargs: dict = dict(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        output_type="latent",
        generator=torch.Generator("cuda").manual_seed(args.seed),
    )

    with torch.inference_mode():
        try:
            pipe(**call_kwargs)
        except TypeError as exc:
            if "guidance_scale" not in str(exc):
                raise
            call_kwargs.pop("guidance_scale")
            pipe(**call_kwargs)

    for h in handles:
        h.remove()

    return dict(stats)


def _analyse(
    stats: dict[str, list[float]],
    min_hits: int,
    ratio_threshold: float,
) -> list[tuple[str, float, float, float, int]]:
    rows = []
    for name, vals in stats.items():
        if len(vals) < min_hits:
            continue
        lo = min(vals)
        hi = max(vals)
        ratio = hi / max(lo, 1e-9)
        rows.append((name, lo, hi, ratio, len(vals)))
    rows.sort(key=lambda r: -r[3])
    return rows


def _print_report(rows: list, ratio_threshold: float) -> None:
    W = 70
    header = f"{'Layer':<{W}}  {'amax_min':>9}  {'amax_max':>9}  {'ratio':>7}  hits"
    print(f"\n{header}")
    print("-" * len(header))
    for name, lo, hi, ratio, hits in rows:
        flag = " ***" if ratio > ratio_threshold else ""
        print(f"{name[-W:]:<{W}}  {lo:9.4f}  {hi:9.4f}  {ratio:6.1f}x  {hits:3d}{flag}")
    flagged = sum(1 for _, _, _, r, _ in rows if r > ratio_threshold)
    print(f"\n{len(rows)} layers tracked, {flagged} flagged (ratio > {ratio_threshold}x)")


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    min_hits = args.min_hits if args.min_hits is not None else args.num_inference_steps

    print(f"Loading {args.model} in {args.dtype}...")
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    print(
        f"Running {args.num_inference_steps}-step inference at "
        f"{args.height}x{args.width} / {args.num_frames} frames..."
    )
    stats = _collect_stats(pipe, args)

    rows = _analyse(stats, min_hits=min_hits, ratio_threshold=args.ratio_threshold)
    _print_report(rows, args.ratio_threshold)

    out = Path(args.output)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "amax_min", "amax_max", "ratio", "hits"])
        w.writerows(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()