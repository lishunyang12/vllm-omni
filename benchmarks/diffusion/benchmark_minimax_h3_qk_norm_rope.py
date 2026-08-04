# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark the MiniMax H3 Q/K RMSNorm + packed 3D RoPE fusion.

Example:

    CUDA_VISIBLE_DEVICES=0 python benchmarks/diffusion/benchmark_minimax_h3_qk_norm_rope.py

The default shape is the TP=4 H3 5-second workload: 37,760 aligned tokens and
14 local Q/K heads of width 128.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=37_760)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=14)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _eager(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.rms_norm(q, (q.shape[-1],), q_weight, eps)
    k = F.rms_norm(k, (k.shape[-1],), k_weight, eps)
    half = rope_table.shape[-1] // 2
    cos = rope_table[..., :half].to(q.dtype).unsqueeze(1)
    sin = rope_table[..., half:].to(q.dtype).unsqueeze(1)

    def _apply(x: torch.Tensor) -> torch.Tensor:
        first = x[..., :half]
        second = x[..., half : 2 * half]
        return torch.cat(
            (
                first * cos - second * sin,
                second * cos + first * sin,
                x[..., 2 * half :],
            ),
            dim=-1,
        )

    return _apply(q), _apply(k)


def _measure(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.head_dim != 128:
        raise ValueError("MiniMax H3 fused QK/RoPE requires --head-dim 128")
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    eps = 1e-5
    torch.manual_seed(17)
    q = torch.randn(args.tokens, args.heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.tokens, args.kv_heads, args.head_dim, device=device, dtype=dtype)
    q_weight = torch.randn(args.head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(args.head_dim, device=device, dtype=dtype)
    raw_freqs = torch.randn(args.tokens, 48, device=device, dtype=torch.float32)
    rope_table = torch.cat((torch.cos(raw_freqs), torch.sin(raw_freqs)), dim=-1).to(dtype)

    def eager():
        return _eager(q, k, q_weight, k_weight, rope_table, eps)

    def fused():
        return fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps)

    eager_samples = _measure(eager, args.warmup, args.iters)
    fused_samples = _measure(fused, args.warmup, args.iters)
    ref_q, ref_k = eager()
    out_q, out_k = fused()
    torch.accelerator.synchronize()

    def stats(samples: list[float]) -> dict[str, float]:
        samples = sorted(samples)
        return {
            "median_ms": samples[len(samples) // 2],
            "p90_ms": samples[int(len(samples) * 0.9)],
            "mean_ms": sum(samples) / len(samples),
        }

    eager_stats = stats(eager_samples)
    fused_stats = stats(fused_samples)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "tokens": args.tokens,
                "heads": args.heads,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
                "dtype": args.dtype,
                "eager": eager_stats,
                "fused": fused_stats,
                "speedup": eager_stats["median_ms"] / fused_stats["median_ms"],
                "max_abs_q": (out_q.float() - ref_q.float()).abs().max().item(),
                "max_abs_k": (out_k.float() - ref_k.float()).abs().max().item(),
                "mean_abs_q": (out_q.float() - ref_q.float()).abs().mean().item(),
                "mean_abs_k": (out_k.float() - ref_k.float()).abs().mean().item(),
                "bitwise_equal_q": torch.equal(out_q, ref_q),
                "bitwise_equal_k": torch.equal(out_k, ref_k),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
