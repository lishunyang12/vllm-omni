"""
Reimplemented from SageAttention official bench:
https://github.com/thu-ml/SageAttention/tree/main/bench

Scripts:
  bench_baseline.py       -> --method fa2/torch/xformers
  bench_fa3.py            -> --method fa3
  bench_qk_int8_pv_fp16_cuda.py   -> --method sage_int8_fp16_cuda
  bench_qk_int8_pv_fp16_triton.py -> --method sage_int8_fp16_triton
  bench_qk_int8_pv_fp8_cuda.py    -> --method sage_int8_fp8_cuda (SM89, RTX 4090)
  bench_qk_int8_pv_fp8_cuda_sm90.py -> --method sage_int8_fp8_cuda_sm90 (H100)

Usage:
  python bench_attn_kernel.py --method fa3 --dtype bfloat16
  python bench_attn_kernel.py --method sageattn --dtype bfloat16
  python bench_attn_kernel.py --method fa2
  python bench_attn_kernel.py --method torch
  python bench_attn_kernel.py --method sage_int8_fp16_cuda
  python bench_attn_kernel.py --method sage_int8_fp16_triton
  python bench_attn_kernel.py --method sage_int8_fp8_cuda
  python bench_attn_kernel.py --method sage_int8_fp8_cuda_sm90
"""

import argparse
import re
import subprocess

import torch
import torch.utils.benchmark as benchmark


def benchmark_forward(fn, *inputs, repeats=100, desc="", verbose=False, **kwinputs):
    """Reimplemented from flash_attn.utils.benchmark.benchmark_forward
    so we don't need flash_attn installed just for the timer."""
    t = benchmark.Timer(
        stmt="fn(*inputs, **kwinputs)",
        globals={"fn": fn, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(desc, "- Forward pass")
        print(m)
    return t, m


def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode()
        match = re.search(r'release (\d+)\.(\d+)', output)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            return major, minor
    except Exception as e:
        print("Failed to get CUDA version:", e)
    return None, None


parser = argparse.ArgumentParser(description='Attention Kernel Benchmark (SageAttention official style)')
parser.add_argument('--method', type=str, default='fa3',
                    choices=['fa2', 'torch', 'xformers', 'fa3', 'sageattn',
                             'sage_int8_fp16_cuda', 'sage_int8_fp16_triton',
                             'sage_int8_fp8_cuda', 'sage_int8_fp8_cuda_sm90'])
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--quant_gran', type=str, default='per_warp', choices=['per_warp', 'per_thread'],
                    help='Quantization granularity (sage kernels only)')
parser.add_argument('--pv_accum_dtype', type=str, default=None,
                    help='PV accumulation dtype (sage kernels only)')
parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'],
                    help='Data type for FA3/sageattn/baseline (default: float16)')
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim
dtype = getattr(torch, args.dtype)

# ============================================================
# bench_baseline: fa2 / torch / xformers
# ============================================================
if args.method in ('fa2', 'torch', 'xformers'):
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    torch.backends.cuda.enable_flash_sdp(args.method == 'fa2')
    torch.backends.cuda.enable_math_sdp(args.method == 'torch')
    torch.backends.cuda.enable_mem_efficient_sdp(args.method == 'xformers')

    print(f"Baseline: {args.method}")
    print(f"batch: {batch}, head: {head}, headdim: {headdim}, dtype: {args.dtype}")

    for is_causal in [False, True]:
        print(f"is_causal: {is_causal}")
        for seq_len in sorted({1024, 2048, 4096, 8192, 16384, 32768}):
            flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
            q = torch.randn(batch, head, seq_len, headdim, dtype=dtype, device="cuda")
            k = torch.randn(batch, head, seq_len, headdim, dtype=dtype, device="cuda")
            v = torch.randn(batch, head, seq_len, headdim, dtype=dtype, device="cuda")
            for i in range(5): sdpa(q, k, v, is_causal=is_causal)
            torch.cuda.synchronize()
            _, time = benchmark_forward(sdpa, q, k, v, is_causal=is_causal, repeats=100, verbose=False, desc='Triton')
            print(f'{seq_len} flops:{flops/time.mean*1e-12}')

# ============================================================
# bench_fa3
# ============================================================
elif args.method == 'fa3':
    # Try fa3_fwd_interface first (vllm-omni custom build), then flash_attn_interface
    flash_attn_func_v3 = None
    fa3_source = None
    for mod_name in ['fa3_fwd_interface', 'flash_attn_interface']:
        try:
            mod = __import__(mod_name, fromlist=['flash_attn_func'])
            flash_attn_func_v3 = getattr(mod, 'flash_attn_func')
            fa3_source = mod_name
            break
        except (ImportError, AttributeError):
            continue

    if flash_attn_func_v3 is None:
        raise ImportError("Neither fa3_fwd_interface nor flash_attn_interface found. Install FA3.")

    print(f"FlashAttention3 Benchmark (source: {fa3_source})")
    print(f"batch: {batch}, head: {head}, headdim: {headdim}, dtype: {args.dtype}")

    for is_causal in [False, True]:
        print(f"is_causal: {is_causal}")
        for seq_len in sorted({1024, 2048, 4096, 8192, 16384, 32768}):
            flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
            q = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
            k = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
            v = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
            for i in range(5): flash_attn_func_v3(q, k, v, causal=is_causal)
            torch.cuda.synchronize()
            _, time = benchmark_forward(flash_attn_func_v3, q, k, v, causal=is_causal, repeats=100, verbose=False, desc='Triton')
            print(f'{seq_len} flops:{flops/time.mean*1e-12}')

# ============================================================
# bench sageattn high-level API (what vllm-omni actually calls)
# ============================================================
elif args.method == 'sageattn':
    from sageattention import sageattn

    print(f"SageAttention (sageattn high-level API) Benchmark")
    print(f"batch: {batch}, head: {head}, headdim: {headdim}, dtype: {args.dtype}")

    for is_causal in [False, True]:
        print(f"is_causal: {is_causal}")
        for seq_len in sorted({1024, 2048, 4096, 8192, 16384, 32768}):
            flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
            q = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
            k = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
            v = torch.randn(batch, seq_len, head, headdim, dtype=dtype, device="cuda")
            for i in range(5): sageattn(q, k, v, tensor_layout="NHD", is_causal=is_causal)
            torch.cuda.synchronize()
            _, time = benchmark_forward(sageattn, q, k, v, tensor_layout="NHD", is_causal=is_causal, repeats=100, verbose=False, desc='SageAttn')
            print(f'{seq_len} flops:{flops/time.mean*1e-12}')

# ============================================================
# bench_qk_int8_pv_fp16_cuda
# ============================================================
elif args.method == 'sage_int8_fp16_cuda':
    import sageattention._qattn_sm80 as qattn

    pv_accum = args.pv_accum_dtype or 'fp16'
    assert pv_accum in ('fp16', 'fp16+fp32', 'fp32')

    WARP_Q = 16 if (headdim == 128 and pv_accum == "fp16+fp32") else 32
    WARP_K = 64

    if pv_accum == 'fp32':
        kernel = qattn.qk_int8_sv_f16_accum_f32_attn
    elif pv_accum == 'fp16+fp32':
        kernel = qattn.qk_int8_sv_f16_accum_f16_attn_inst_buf
    elif pv_accum == 'fp16':
        kernel = qattn.qk_int8_sv_f16_accum_f16_attn

    _qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2

    print(f"CUDA QK Int8 PV FP16 Benchmark")
    print(f"batch: {batch}, head: {head}, headdim: {headdim}, pv_accum_dtype: {pv_accum}")

    for is_causal in [False, True]:
        _is_causal = 1 if is_causal else 0
        print(f"is_causal: {is_causal}")
        for seq_len in sorted({1024, 2048, 4096, 8192, 16384, 32768}):
            flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)

            q = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
            k = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")

            if args.quant_gran == 'per_warp':
                q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float, device="cuda")
                k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float, device="cuda")
            elif args.quant_gran == 'per_thread':
                q_scale = torch.randn(batch, head, seq_len // WARP_Q * 8, dtype=torch.float, device="cuda")
                k_scale = torch.randn(batch, head, seq_len // WARP_K * 4, dtype=torch.float, device="cuda")

            v = torch.randn(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")
            o = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")
            sm_scale = 1 / (headdim ** 0.5)
            for i in range(5): kernel(q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
            torch.cuda.synchronize()
            _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
            print(f'{seq_len} flops:{flops/time.mean*1e-12}')

# ============================================================
# bench_qk_int8_pv_fp16_triton
# ============================================================
elif args.method == 'sage_int8_fp16_triton':
    from sageattention.triton.attn_qk_int8_per_block import forward
    from sageattention.triton.attn_qk_int8_per_block_causal import forward as forward_causal

    print(f"Triton QK Int8 PV FP16 Benchmark")
    print(f"batch_size: {batch}, num_heads: {head}, head_dim: {headdim}")

    # non-causal
    print("is_causal: False")
    for seq_len in sorted({1024, 2048, 4096, 8192, 16384, 32768}):
        flops = 4 * head * batch * headdim * seq_len * seq_len

        q = torch.randint(-100, 100, (batch, head, seq_len, headdim), dtype=torch.int8, device='cuda')
        k = torch.randint(-100, 100, (batch, head, seq_len, headdim), dtype=torch.int8, device='cuda')
        v = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device='cuda')

        q_scale = torch.randn(batch, head, (seq_len // 128), 1, dtype=torch.float16, device='cuda')
        k_scale = torch.randn(batch, head, (seq_len // 64), 1, dtype=torch.float16, device='cuda')

        for i in range(5): forward(q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        _, time = benchmark_forward(forward, q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16, repeats=100, verbose=False, desc='Triton')
        print(f'{seq_len} flops:{flops/time.mean*1e-12}')

    # causal
    print("is_causal: True")
    for seq_len in sorted({1024, 2048, 4096, 8192, 16384, 32768}):
        flops = 4 * head * batch * headdim * seq_len * seq_len // 2

        q = torch.randint(-100, 100, (batch, head, seq_len, headdim), dtype=torch.int8, device='cuda')
        k = torch.randint(-100, 100, (batch, head, seq_len, headdim), dtype=torch.int8, device='cuda')
        v = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device='cuda')

        q_scale = torch.randn(batch, head, (seq_len // 128), 1, dtype=torch.float16, device='cuda')
        k_scale = torch.randn(batch, head, (seq_len // 64), 1, dtype=torch.float16, device='cuda')

        for i in range(5): forward_causal(q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        _, time = benchmark_forward(forward_causal, q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16, repeats=100, verbose=False, desc='Triton')
        print(f'{seq_len} flops:{flops/time.mean*1e-12}')

# ============================================================
# bench_qk_int8_pv_fp8_cuda (SM89 / RTX 4090)
# ============================================================
elif args.method == 'sage_int8_fp8_cuda':
    import sageattention._qattn_sm89 as qattn

    pv_accum = args.pv_accum_dtype or 'fp32+fp16'
    assert pv_accum in ('fp32', 'fp32+fp32', 'fp32+fp16')

    cuda_major, cuda_minor = get_cuda_version()
    if (cuda_major, cuda_minor) < (12, 8) and pv_accum == 'fp32+fp16':
        print("=============\n NOTE: cuda version < 12.8, not support pv_accum_dtype fp32+fp16.")
        print(" Switch to 'fp32+fp32' automatically\n=============")
        pv_accum = 'fp32+fp32'

    WARP_Q = 32
    WARP_K = 64

    if pv_accum == 'fp32':
        kernel = qattn.qk_int8_sv_f8_accum_f32_attn
    elif pv_accum == 'fp32+fp32':
        kernel = qattn.qk_int8_sv_f8_accum_f32_attn_inst_buf
    elif pv_accum == 'fp32+fp16':
        kernel = qattn.qk_int8_sv_f8_accum_f16_attn_inst_buf

    _qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2

    print(f"CUDA QK Int8 PV FP8 Benchmark (SM89)")
    print(f"batch: {batch}, head: {head}, headdim: {headdim}, pv_accum_dtype: {pv_accum}")

    for is_causal in [False, True]:
        _is_causal = 1 if is_causal else 0
        print(f"is_causal: {is_causal}")
        for seq_len in sorted({1024, 2048, 4096, 8192, 16384, 32768}):
            flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)

            q = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
            k = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
            o = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")

            vm = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")
            v_scale = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")

            if args.quant_gran == 'per_warp':
                q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float, device="cuda")
                k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float, device="cuda")
            elif args.quant_gran == 'per_thread':
                q_scale = torch.randn(batch, head, seq_len // WARP_Q * 8, dtype=torch.float, device="cuda")
                k_scale = torch.randn(batch, head, seq_len // WARP_K * 4, dtype=torch.float, device="cuda")

            v = torch.randn(batch, headdim, head, seq_len, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
            sm_scale = 1 / (headdim ** 0.5)
            for i in range(5): kernel(q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
            torch.cuda.synchronize()
            _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
            print(f'{seq_len} flops:{flops/time.mean*1e-12}')

# ============================================================
# bench_qk_int8_pv_fp8_cuda_sm90 (H100)
# ============================================================
elif args.method == 'sage_int8_fp8_cuda_sm90':
    import sageattention._qattn_sm90 as qattn

    pv_accum = args.pv_accum_dtype or 'fp32+fp32'
    assert pv_accum == 'fp32+fp32', "pure fp32 accumulator is not supported for now"

    WARP_Q = 32
    WARP_K = 64

    kernel = qattn.qk_int8_sv_f8_accum_f32_attn_inst_buf

    _qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2

    print(f"CUDA QK Int8 PV FP8 SM90 Benchmark")
    print(f"batch: {batch}, head: {head}, headdim: {headdim}")

    for is_causal in [False, True]:
        _is_causal = 1 if is_causal else 0
        print(f"is_causal: {is_causal}")
        for seq_len in sorted({1024, 2048, 4096, 8192, 16384, 32768}):
            flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)

            q = torch.randint(-95, 95, (batch, head, seq_len, headdim), dtype=torch.int8, device="cuda")
            k = torch.randint(-95, 95, (batch, head, seq_len, headdim), dtype=torch.int8, device="cuda")
            o = torch.empty(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")

            v_scale = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")

            if args.quant_gran == 'per_warp':
                q_scale = torch.randn(batch, head, seq_len // 64 * 4, dtype=torch.float, device="cuda")
                k_scale = torch.randn(batch, head, seq_len // 128, dtype=torch.float, device="cuda")
            elif args.quant_gran == 'per_thread':
                q_scale = torch.randn(batch, head, seq_len // 64 * 4 * 8, dtype=torch.float, device="cuda")
                k_scale = torch.randn(batch, head, seq_len // 128 * 4, dtype=torch.float, device="cuda")

            v = torch.randn(batch, head, headdim, seq_len, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
            sm_scale = 1 / (headdim ** 0.5)
            for i in range(5): kernel(q, k, v, o, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0)
            torch.cuda.synchronize()
            _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
            print(f'{seq_len} flops:{flops/time.mean*1e-12}')
