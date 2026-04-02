"""Debug script: check FA3 FP8 capabilities, run micro-benchmarks,
and profile attention kernel performance for optimization planning."""
import inspect
import time

import torch

# ============================================================
# 1. FA3 FP8 descale support check
# ============================================================
print("=" * 60)
print("1. FA3 FP8 descale support check")
print("=" * 60)

try:
    from fa3_fwd_interface import _flash_attn_forward
    sig = inspect.signature(_flash_attn_forward)
    params = list(sig.parameters.keys())
    print(f"fa3_fwd params: {params}")
    has_descale = any("descale" in p for p in params)
    print(f"Has descale support: {has_descale}")
except Exception as e:
    print(f"fa3_fwd not available: {e}")

try:
    from vllm_omni.diffusion.attention.backends.utils.fa import (
        flash_attn_varlen_func,
    )
    sig = inspect.signature(flash_attn_varlen_func)
    params = list(sig.parameters.keys())
    print(f"\nvarlen func params: {params}")
    has_descale = any("descale" in p for p in params)
    print(f"Varlen has descale: {has_descale}")
except Exception as e:
    print(f"varlen not available: {e}")

try:
    from vllm_omni.diffusion.attention.backends.ring.ring_globals import (
        HAS_FA3,
        fa3_attn_func,
    )
    print(f"\nHAS_FA3: {HAS_FA3}")
    if fa3_attn_func is not None:
        sig = inspect.signature(fa3_attn_func)
        params = list(sig.parameters.keys())
        print(f"fa3_attn_func params: {params}")
except Exception as e:
    print(f"ring_globals import failed: {e}")

# ============================================================
# 2. Package versions and GPU info
# ============================================================
print("\n" + "=" * 60)
print("2. Environment")
print("=" * 60)

try:
    import flash_attn
    print(f"flash_attn version: {flash_attn.__version__}")
except Exception:
    print("flash_attn: not installed or no __version__")

print(f"torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
    print(f"GPU: {gpu} (SM {cap[0]}{cap[1]}, {mem:.1f} GB)")
    print(f"  FP8 tensor cores: {'Yes' if cap[0] >= 9 else 'No'} (need SM90+)")
else:
    print("GPU: N/A")

# Check vLLM fused quant kernel
try:
    from vllm._custom_ops import scaled_fp8_quant
    print(f"vLLM scaled_fp8_quant: available")
except Exception:
    print("vLLM scaled_fp8_quant: NOT available (will use PyTorch fallback)")

# Check torch.compile status
print(f"torch.compile available: {hasattr(torch, 'compile')}")
try:
    import triton
    print(f"triton version: {triton.__version__}")
except Exception:
    print("triton: not installed")

# ============================================================
# 3. FP8 micro-benchmarks (quantization overhead)
# ============================================================
if not torch.cuda.is_available():
    print("\nSkipping benchmarks (no GPU)")
    exit()

print("\n" + "=" * 60)
print("3. FP8 quantization overhead micro-benchmark")
print("=" * 60)

device = "cuda"

# Simulate HunyuanVideo tensor shapes
# 33 frames: ~(1, 2640, 24, 128) for single-stream, (1, 2640+256, 24, 128) for joint
# 121 frames: ~(1, 9680, 24, 128) for single-stream
test_shapes = [
    ("33f single-stream", (1, 2640, 24, 128)),
    ("33f joint (img+txt)", (1, 2896, 24, 128)),
    ("121f single-stream", (1, 9680, 24, 128)),
    ("121f joint (img+txt)", (1, 9936, 24, 128)),
]

for name, shape in test_shapes:
    q = torch.randn(shape, dtype=torch.bfloat16, device=device)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(3):
        from vllm_omni.quantization.kv_quant import quantize_qkv_fp8
        quantize_qkv_fp8(q, k, v)
    torch.cuda.synchronize()

    # Benchmark quantization
    n_iters = 20
    start = time.perf_counter()
    for _ in range(n_iters):
        fp8_q, fp8_k, fp8_v, qs, ks, vs = quantize_qkv_fp8(q, k, v)
    torch.cuda.synchronize()
    quant_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"  {name} {list(shape)}: quant={quant_time:.2f} ms")

# ============================================================
# 4. FA3 attention kernel benchmark (BF16 vs FP8)
# ============================================================
print("\n" + "=" * 60)
print("4. FA3 attention kernel benchmark (BF16 vs FP8)")
print("=" * 60)

try:
    from vllm_omni.diffusion.attention.backends.ring.ring_globals import (
        HAS_FA3,
        fa3_attn_func,
    )
    if not HAS_FA3 or fa3_attn_func is None:
        raise RuntimeError("FA3 not available")
except Exception as e:
    print(f"Skipping: {e}")
    exit()

bench_shapes = [
    ("33f", (1, 2640, 24, 128)),
    ("121f", (1, 9680, 24, 128)),
]

n_warmup = 5
n_iters = 20

for name, shape in bench_shapes:
    B, S, H, D = shape
    softmax_scale = D ** -0.5

    # BF16 benchmark
    q_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)

    for _ in range(n_warmup):
        fa3_attn_func(q_bf16, k_bf16, v_bf16, softmax_scale=softmax_scale, causal=False)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        fa3_attn_func(q_bf16, k_bf16, v_bf16, softmax_scale=softmax_scale, causal=False)
    torch.cuda.synchronize()
    bf16_time = (time.perf_counter() - start) / n_iters * 1000

    # FP8 benchmark (quantize + attention)
    fp8_q, fp8_k, fp8_v, qs, ks, vs = quantize_qkv_fp8(q_bf16, k_bf16, v_bf16)

    for _ in range(n_warmup):
        fa3_attn_func(fp8_q, fp8_k, fp8_v, softmax_scale=softmax_scale,
                      causal=False, q_descale=qs, k_descale=ks, v_descale=vs)
    torch.cuda.synchronize()

    # FP8 kernel only (no quant overhead)
    start = time.perf_counter()
    for _ in range(n_iters):
        fa3_attn_func(fp8_q, fp8_k, fp8_v, softmax_scale=softmax_scale,
                      causal=False, q_descale=qs, k_descale=ks, v_descale=vs)
    torch.cuda.synchronize()
    fp8_kernel_time = (time.perf_counter() - start) / n_iters * 1000

    # FP8 end-to-end (quant + attention)
    start = time.perf_counter()
    for _ in range(n_iters):
        fp8_q, fp8_k, fp8_v, qs, ks, vs = quantize_qkv_fp8(q_bf16, k_bf16, v_bf16)
        fa3_attn_func(fp8_q, fp8_k, fp8_v, softmax_scale=softmax_scale,
                      causal=False, q_descale=qs, k_descale=ks, v_descale=vs)
    torch.cuda.synchronize()
    fp8_e2e_time = (time.perf_counter() - start) / n_iters * 1000

    speedup_kernel = bf16_time / fp8_kernel_time
    speedup_e2e = bf16_time / fp8_e2e_time

    print(f"\n  {name} {list(shape)}:")
    print(f"    BF16 attn:           {bf16_time:.2f} ms")
    print(f"    FP8 kernel only:     {fp8_kernel_time:.2f} ms  ({speedup_kernel:.2f}x)")
    print(f"    FP8 quant+kernel:    {fp8_e2e_time:.2f} ms  ({speedup_e2e:.2f}x)")
    print(f"    Quant overhead:      {fp8_e2e_time - fp8_kernel_time:.2f} ms")

# ============================================================
# 5. FA3 varlen FP8 benchmark (with padding mask)
# ============================================================
print("\n" + "=" * 60)
print("5. FA3 varlen FP8 benchmark (with padding mask)")
print("=" * 60)

try:
    from vllm_omni.diffusion.attention.backends.utils.fa import (
        flash_attn_varlen_func,
        _unpad_input,
        _upad_input,
    )
except Exception as e:
    print(f"Skipping varlen benchmark: {e}")
    exit()

for name, shape in bench_shapes:
    B, S, H, D = shape
    softmax_scale = D ** -0.5

    q_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)

    # Create a realistic mask: mostly True, some False at end (encoder padding)
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    n_pad = max(1, S // 20)  # 5% padding
    mask[:, -n_pad:] = False

    # Unpad inputs
    q_up, k_up, v_up, indices_q, (cu_q, cu_k), (max_q, max_k) = _upad_input(
        q_bf16, k_bf16, v_bf16, mask, S, _unpad_input
    )

    # BF16 varlen
    for _ in range(n_warmup):
        flash_attn_varlen_func(q_up, k_up, v_up, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                               max_seqlen_q=max_q, max_seqlen_k=max_k,
                               softmax_scale=softmax_scale, causal=False)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        flash_attn_varlen_func(q_up, k_up, v_up, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                               max_seqlen_q=max_q, max_seqlen_k=max_k,
                               softmax_scale=softmax_scale, causal=False)
    torch.cuda.synchronize()
    varlen_bf16 = (time.perf_counter() - start) / n_iters * 1000

    # FP8 varlen
    fp8_q, fp8_k, fp8_v, qs, ks, vs = quantize_qkv_fp8(q_bf16, k_bf16, v_bf16)
    q_up_fp8, k_up_fp8, v_up_fp8, _, _, _ = _upad_input(
        fp8_q, fp8_k, fp8_v, mask, S, _unpad_input
    )

    for _ in range(n_warmup):
        flash_attn_varlen_func(q_up_fp8, k_up_fp8, v_up_fp8,
                               cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                               max_seqlen_q=max_q, max_seqlen_k=max_k,
                               softmax_scale=softmax_scale, causal=False,
                               q_descale=qs, k_descale=ks, v_descale=vs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        flash_attn_varlen_func(q_up_fp8, k_up_fp8, v_up_fp8,
                               cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                               max_seqlen_q=max_q, max_seqlen_k=max_k,
                               softmax_scale=softmax_scale, causal=False,
                               q_descale=qs, k_descale=ks, v_descale=vs)
    torch.cuda.synchronize()
    varlen_fp8 = (time.perf_counter() - start) / n_iters * 1000

    speedup = varlen_bf16 / varlen_fp8

    print(f"\n  {name} {list(shape)} (5% padding):")
    print(f"    BF16 varlen:  {varlen_bf16:.2f} ms")
    print(f"    FP8 varlen:   {varlen_fp8:.2f} ms  ({speedup:.2f}x)")

# ============================================================
# 6. Breakdown: where time goes in a DiT layer
# ============================================================
print("\n" + "=" * 60)
print("6. Time breakdown estimate for one DiT layer")
print("=" * 60)

shape_121f = (1, 9680, 24, 128)
B, S, H, D = shape_121f
hidden_dim = H * D  # 3072
softmax_scale = D ** -0.5

# Linear projections (Q/K/V projection + output projection)
x = torch.randn(B, S, hidden_dim, dtype=torch.bfloat16, device=device)
w_qkv = torch.randn(hidden_dim * 3, hidden_dim, dtype=torch.bfloat16, device=device)
w_out = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16, device=device)

for _ in range(n_warmup):
    torch.nn.functional.linear(x, w_qkv)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(n_iters):
    torch.nn.functional.linear(x, w_qkv)
torch.cuda.synchronize()
linear_qkv_time = (time.perf_counter() - start) / n_iters * 1000

for _ in range(n_warmup):
    torch.nn.functional.linear(x, w_out)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(n_iters):
    torch.nn.functional.linear(x, w_out)
torch.cuda.synchronize()
linear_out_time = (time.perf_counter() - start) / n_iters * 1000

# Attention (already measured above)
q_bf16 = torch.randn(shape_121f, dtype=torch.bfloat16, device=device)
k_bf16 = torch.randn(shape_121f, dtype=torch.bfloat16, device=device)
v_bf16 = torch.randn(shape_121f, dtype=torch.bfloat16, device=device)

for _ in range(n_warmup):
    fa3_attn_func(q_bf16, k_bf16, v_bf16, softmax_scale=softmax_scale, causal=False)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(n_iters):
    fa3_attn_func(q_bf16, k_bf16, v_bf16, softmax_scale=softmax_scale, causal=False)
torch.cuda.synchronize()
attn_time = (time.perf_counter() - start) / n_iters * 1000

total = linear_qkv_time + attn_time + linear_out_time
print(f"  121f single layer breakdown (estimated):")
print(f"    QKV projection:  {linear_qkv_time:.2f} ms ({linear_qkv_time/total*100:.0f}%)")
print(f"    Attention:       {attn_time:.2f} ms ({attn_time/total*100:.0f}%)")
print(f"    Output proj:     {linear_out_time:.2f} ms ({linear_out_time/total*100:.0f}%)")
print(f"    Total:           {total:.2f} ms")
print(f"    Layers x steps:  54 layers x 30 steps = {54*30} calls")
print(f"    Attn total est:  {attn_time * 54 * 30 / 1000:.1f}s out of ~{total * 54 * 30 / 1000:.1f}s")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
