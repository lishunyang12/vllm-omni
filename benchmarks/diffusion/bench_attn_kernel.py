# SPDX-License-Identifier: Apache-2.0
"""
Kernel-level benchmark: FA3 vs SageAttention vs SDPA
Follows SageAttention official bench style (TFLOPS + sweep seq lengths).
Reference: https://github.com/thu-ml/SageAttention/tree/main/bench

HunyuanVideo 1.5 diffusion config: B=1, H=16, D=128
LLM-style config (SageAttention default): B=4, H=32, D=128

Usage:
  # Diffusion config (default) — HunyuanVideo 1.5
  python bench_attn_kernel.py

  # LLM config (matches SageAttention official bench)
  python bench_attn_kernel.py --batch-size 4 --num-heads 32 --dtype float16

  # Single seq length
  python bench_attn_kernel.py --seq-len 48360

  # Sweep mode (multiple seq lengths)
  python bench_attn_kernel.py --sweep
"""

import argparse
import time

import torch


def _flush_l2():
    """Flush L2 cache with 256 MB zeros (same as SageAttention bench)."""
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    cache.zero_()


def benchmark_fn(fn, warmup=5, repeat=100, flush_l2=True):
    """Benchmark with CUDA events (matches SageAttention bench style)."""
    # warmup
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        if flush_l2:
            _flush_l2()
        fn()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / repeat
    return elapsed_ms


def calc_flops(batch, heads, headdim, seq_len, causal=False):
    """Standard attention FLOPS: 4 * B * H * D * S^2 (halved if causal)."""
    flops = 4 * batch * heads * headdim * seq_len * seq_len
    if causal:
        flops //= 2
    return flops


def _get_flash_attn_func():
    """Try fa3_fwd_interface -> flash_attn_interface -> flash_attn."""
    for module_name in [
        "fa3_fwd_interface",
        "flash_attn_interface",
        "flash_attn",
    ]:
        try:
            mod = __import__(module_name, fromlist=["flash_attn_func"])
            return getattr(mod, "flash_attn_func"), module_name
        except (ImportError, AttributeError):
            continue
    return None, None


def run_single(B, S, H, D, dtype, repeat, causal=False):
    """Run all backends for a single (B, S, H, D) config."""
    device = "cuda"
    flops = calc_flops(B, H, D, S, causal)

    q = torch.randn(B, S, H, D, dtype=dtype, device=device)
    k = torch.randn(B, S, H, D, dtype=dtype, device=device)
    v = torch.randn(B, S, H, D, dtype=dtype, device=device)

    results = {}

    # --- FA3 / FlashAttention ---
    fa_func, fa_module = _get_flash_attn_func()
    if fa_func is not None:
        try:
            ms = benchmark_fn(lambda: fa_func(q, k, v, causal=causal), repeat=repeat)
            tflops = flops / ms / 1e9  # ms -> s -> TFLOPS
            results["FA3"] = (ms, tflops)
        except Exception as e:
            results["FA3"] = (None, f"ERROR: {e}")
    else:
        results["FA3"] = (None, "N/A")

    # --- SageAttention ---
    try:
        from sageattention import sageattn
        ms = benchmark_fn(
            lambda: sageattn(q, k, v, tensor_layout="NHD", is_causal=causal),
            repeat=repeat,
        )
        tflops = flops / ms / 1e9
        results["SageAttn"] = (ms, tflops)
    except ImportError:
        results["SageAttn"] = (None, "N/A")
    except Exception as e:
        results["SageAttn"] = (None, f"ERROR: {e}")

    # --- torch SDPA ---
    try:
        q_sdpa = q.transpose(1, 2).contiguous()
        k_sdpa = k.transpose(1, 2).contiguous()
        v_sdpa = v.transpose(1, 2).contiguous()
        ms = benchmark_fn(
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, is_causal=causal
            ),
            repeat=repeat,
        )
        tflops = flops / ms / 1e9
        results["SDPA"] = (ms, tflops)
    except Exception as e:
        results["SDPA"] = (None, f"ERROR: {e}")

    return results


def print_row(seq_len, results):
    """Print one row of results."""
    parts = [f"S={seq_len:>6d}"]
    for name in ["FA3", "SageAttn", "SDPA"]:
        ms, tflops = results.get(name, (None, "N/A"))
        if ms is not None:
            parts.append(f"{name}: {ms:7.2f} ms ({tflops:6.1f} TFLOPS)")
        else:
            parts.append(f"{name}: {tflops}")
    print("  ".join(parts))


def main():
    parser = argparse.ArgumentParser(
        description="Attention kernel benchmark (FA3 vs SageAttn vs SDPA)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Single seq length to test")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep standard seq lengths (1K-32K) + diffusion lengths")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    B, H, D = args.batch_size, args.num_heads, args.head_dim

    fa_func, fa_module = _get_flash_attn_func()
    fa_label = f"fa3_fwd ({fa_module})" if fa_module else "N/A"

    print(f"Config: B={B}, H={H}, D={D}, dtype={args.dtype}, causal={args.causal}")
    print(f"FlashAttn source: {fa_label}")
    print(f"Repeat: {args.repeat}, L2 flush: enabled")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    if args.seq_len is not None:
        # Single seq length mode
        seq_lens = [args.seq_len]
    elif args.sweep:
        # Sweep: standard (SageAttention bench) + diffusion-specific
        seq_lens = [1024, 2048, 4096, 8192, 14040, 16384, 32768, 48360]
    else:
        # Default: diffusion-relevant lengths
        seq_lens = [14040, 48360]

    print(f"{'S':>8s}  {'FA3':>24s}  {'SageAttn':>24s}  {'SDPA':>24s}")
    print("-" * 90)

    all_results = {}
    for S in seq_lens:
        results = run_single(B, S, H, D, dtype, args.repeat, causal=args.causal)
        all_results[S] = results

        parts = [f"{S:>8d}"]
        for name in ["FA3", "SageAttn", "SDPA"]:
            ms, tflops = results.get(name, (None, "N/A"))
            if ms is not None:
                parts.append(f"{ms:7.2f} ms / {tflops:6.1f} TF")
            else:
                parts.append(f"{'N/A':>24s}")
        print("  ".join(parts))

    # Summary
    print()
    print("Speedup vs FA3:")
    for S in seq_lens:
        results = all_results[S]
        fa3_ms = results["FA3"][0] if results["FA3"][0] else None
        if fa3_ms is None:
            continue
        parts = [f"  S={S:>6d}"]
        for name in ["SageAttn", "SDPA"]:
            ms = results[name][0] if results[name][0] else None
            if ms:
                parts.append(f"{name}: {ms/fa3_ms:.2f}x")
            else:
                parts.append(f"{name}: N/A")
        print("  ".join(parts))


if __name__ == "__main__":
    main()
