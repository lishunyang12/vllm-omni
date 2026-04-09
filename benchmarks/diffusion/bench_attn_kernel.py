# SPDX-License-Identifier: Apache-2.0
"""
Kernel-level benchmark: FlashAttention vs SageAttention
Isolates attention kernel performance from model loading, torch.compile, VAE, etc.

HunyuanVideo 1.5 config: 16 heads, head_dim=128
Latent seq lengths (after VAE compression 4x temporal, 16x spatial):
  480x832, 33 frames  -> ~9 x 30 x 52  = ~14,040
  480x832, 121 frames -> ~31 x 30 x 52 = ~48,360

Usage:
  python bench_attn_kernel.py
  python bench_attn_kernel.py --seq-len 48360 --num-heads 16 --head-dim 128
"""

import argparse
import time

import torch


def benchmark_fn(fn, warmup=5, repeat=20, **kwargs):
    """Benchmark a function with CUDA synchronization."""
    for _ in range(warmup):
        fn(**kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(**kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times.sort()
    # trim top/bottom 20%
    trim = max(1, len(times) // 5)
    trimmed = times[trim:-trim] if trim < len(times) // 2 else times
    avg = sum(trimmed) / len(trimmed)
    return avg, min(times), max(times)


def bench_flash_attn(q, k, v):
    from flash_attn import flash_attn_func
    return flash_attn_func(q, k, v, causal=False)


def bench_sage_attn(q, k, v):
    from sageattention import sageattn
    return sageattn(q, k, v, tensor_layout="NHD", is_causal=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=48360,
                        help="Sequence length (default: 48360 for 121 frames)")
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda"
    B, S, H, D = args.batch_size, args.seq_len, args.num_heads, args.head_dim

    print(f"Config: B={B}, S={S}, H={H}, D={D}, dtype={args.dtype}")
    print(f"Tensor shape: ({B}, {S}, {H}, {D})")
    mem_per_tensor = B * S * H * D * (2 if dtype == torch.float16 or dtype == torch.bfloat16 else 4)
    print(f"Memory per Q/K/V tensor: {mem_per_tensor / 1e6:.1f} MB")
    print(f"Warmup={args.warmup}, Repeat={args.repeat}")
    print()

    q = torch.randn(B, S, H, D, dtype=dtype, device=device)
    k = torch.randn(B, S, H, D, dtype=dtype, device=device)
    v = torch.randn(B, S, H, D, dtype=dtype, device=device)

    results = {}

    # --- FlashAttention ---
    try:
        from flash_attn import flash_attn_func  # noqa: F401
        avg, lo, hi = benchmark_fn(bench_flash_attn, warmup=args.warmup,
                                   repeat=args.repeat, q=q, k=k, v=v)
        results["FlashAttention"] = (avg, lo, hi)
        print(f"FlashAttention:  avg={avg:7.2f} ms  min={lo:7.2f} ms  max={hi:7.2f} ms")
    except ImportError:
        print("FlashAttention:  NOT AVAILABLE (flash_attn not installed)")
    except Exception as e:
        print(f"FlashAttention:  ERROR - {e}")

    # --- SageAttention ---
    try:
        from sageattention import sageattn  # noqa: F401
        avg, lo, hi = benchmark_fn(bench_sage_attn, warmup=args.warmup,
                                   repeat=args.repeat, q=q, k=k, v=v)
        results["SageAttention"] = (avg, lo, hi)
        print(f"SageAttention:   avg={avg:7.2f} ms  min={lo:7.2f} ms  max={hi:7.2f} ms")
    except ImportError:
        print("SageAttention:   NOT AVAILABLE (sageattention not installed)")
    except Exception as e:
        print(f"SageAttention:   ERROR - {e}")

    # --- torch SDPA ---
    try:
        # SDPA expects (B, H, S, D)
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)

        def bench_sdpa(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False)

        avg, lo, hi = benchmark_fn(bench_sdpa, warmup=args.warmup,
                                   repeat=args.repeat, q=q_sdpa, k=k_sdpa, v=v_sdpa)
        results["torch SDPA"] = (avg, lo, hi)
        print(f"torch SDPA:      avg={avg:7.2f} ms  min={lo:7.2f} ms  max={hi:7.2f} ms")
    except Exception as e:
        print(f"torch SDPA:      ERROR - {e}")

    # --- Summary ---
    if len(results) >= 2:
        print()
        baseline_name = "FlashAttention" if "FlashAttention" in results else list(results.keys())[0]
        baseline_avg = results[baseline_name][0]
        for name, (avg, lo, hi) in results.items():
            ratio = avg / baseline_avg
            print(f"  {name:20s}  {avg:7.2f} ms  ({ratio:.2f}x vs {baseline_name})")


if __name__ == "__main__":
    main()
