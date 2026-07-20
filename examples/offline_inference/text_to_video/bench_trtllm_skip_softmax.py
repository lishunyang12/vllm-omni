# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E latency benchmark for the trtllm attention backend (Skip-Softmax).

Compares BF16 dense vs Skip-Softmax end-to-end on a real Wan 2.2 checkpoint.
Skip can be driven two ways:
  - skip_softmax_threshold: direct, no calibration.
  - target_sparsity: needs a ModelOpt-calibrated checkpoint (carries the a,b
    curve in config.json); without it, target_sparsity is ignored (dense).

Run on a Blackwell (SM100/SM103) box:
    python bench_trtllm_skip_softmax.py --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --height 720 --width 1280 --num-frames 81 --num-inference-steps 50
"""

import argparse
import time

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# (label, diffusion_attention_backend, diffusion_attention_config).
# None backend = platform default (BF16 baseline).
CONFIGS = [
    ("baseline (platform default, BF16)", None, None),
    ("trtllm BF16", "trtllm", {"backend": "TRTLLM", "extra": {}}),
    (
        "trtllm + Skip (threshold, no calibration)",
        "trtllm",
        {"backend": "TRTLLM", "extra": {"skip_softmax_threshold": 0.02, "disabled_until_timestep": 0.86}},
    ),
]


def run_one(args, backend, attn_cfg):
    kwargs = dict(model=args.model, enforce_eager=args.enforce_eager)
    # diffusion_attention_backend and diffusion_attention_config.default.backend are
    # mutually exclusive. When a per-layer config is given it carries the backend, so
    # use only that; otherwise fall back to the plain backend selector.
    if attn_cfg is not None:
        kwargs["diffusion_attention_config"] = {"default": attn_cfg}
    elif backend is not None:
        kwargs["diffusion_attention_backend"] = backend
    omni = Omni(**kwargs)
    sp = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
    )
    prompt = {"prompt": args.prompt}
    omni.generate(prompt, sp)  # warmup
    t0 = time.perf_counter()
    for _ in range(args.iters):
        omni.generate(prompt, sp)
    dt = (time.perf_counter() - t0) / args.iters
    del omni
    return dt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    p.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true")
    args = p.parse_args()

    print(f"Workload: {args.model} {args.width}x{args.height} {args.num_frames}f {args.num_inference_steps} steps\n")
    baseline = None
    for label, backend, attn_cfg in CONFIGS:
        try:
            dt = run_one(args, backend, attn_cfg)
        except Exception as e:  # noqa: BLE001
            print(f"{label:38} FAILED: {e}")
            continue
        if baseline is None:
            baseline = dt
        print(f"{label:38} {dt:8.2f} s   speedup={baseline / dt:.3f}x")


if __name__ == "__main__":
    main()
