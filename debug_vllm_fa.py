"""Check vllm.vllm_flash_attn for FP8 support and two-level accumulation."""
import inspect
import os

print("=" * 60)
print("1. vllm.vllm_flash_attn contents")
print("=" * 60)
try:
    import vllm.vllm_flash_attn as vfa
    print(f"Location: {vfa.__file__}")
    pkg_dir = os.path.dirname(vfa.__file__)
    for f in sorted(os.listdir(pkg_dir)):
        full = os.path.join(pkg_dir, f)
        if os.path.isfile(full):
            size = os.path.getsize(full)
            print(f"  {f} ({size/1024:.1f} KB)")

    print(f"\nExported names: {dir(vfa)}")
except Exception as e:
    print(f"Not available: {e}")

print("\n" + "=" * 60)
print("2. Check for flash_attn_func / varlen with descale")
print("=" * 60)
funcs_to_check = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
]
for fname in funcs_to_check:
    try:
        func = getattr(vfa, fname, None)
        if func is None:
            # Try submodule
            try:
                from vllm.vllm_flash_attn import flash_attn_interface
                func = getattr(flash_attn_interface, fname, None)
            except:
                pass
        if func is not None:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            has_descale = any("descale" in p for p in params)
            print(f"\n  {fname}:")
            print(f"    params: {params}")
            print(f"    has descale: {has_descale}")
        else:
            print(f"\n  {fname}: not found")
    except Exception as e:
        print(f"\n  {fname}: error - {e}")

print("\n" + "=" * 60)
print("3. Check for two-level accumulation in source")
print("=" * 60)
try:
    pkg_dir = os.path.dirname(vfa.__file__)
    keywords = [
        "two_level", "TWO_LEVEL", "fp8_two_level",
        "FP8_TWO_LEVEL", "accum", "flush",
    ]
    for f in os.listdir(pkg_dir):
        if f.endswith('.py'):
            filepath = os.path.join(pkg_dir, f)
            with open(filepath, 'r') as fh:
                content = fh.read()
            for kw in keywords:
                if kw.lower() in content.lower():
                    # Find the line
                    for i, line in enumerate(content.split('\n')):
                        if kw.lower() in line.lower():
                            print(f"  {f}:{i+1}: {line.strip()[:100]}")
                            break
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("4. Check CUDA backend for FP8")
print("=" * 60)
try:
    pkg_dir = os.path.dirname(vfa.__file__)
    for f in os.listdir(pkg_dir):
        if f.endswith('.so') or f.endswith('.pyd'):
            full = os.path.join(pkg_dir, f)
            size_mb = os.path.getsize(full) / 1024 / 1024
            print(f"  {f} ({size_mb:.1f} MB)")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("5. Quick FP8 functional test with vllm_flash_attn")
print("=" * 60)
try:
    import torch
    if not torch.cuda.is_available():
        print("  No GPU, skipping")
    else:
        # Try to use vllm's flash_attn for FP8
        func = getattr(vfa, 'flash_attn_func', None)
        if func is None:
            try:
                from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_func as func
            except:
                pass

        if func is not None:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if any("descale" in p for p in params):
                # Test FP8 attention
                B, S, H, D = 1, 1024, 16, 128
                q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")

                # Quantize to FP8
                from vllm_omni.quantization.kv_quant import quantize_qkv_fp8
                fp8_q, fp8_k, fp8_v, qs, ks, vs = quantize_qkv_fp8(q, k, v)

                qs_2d = qs.view(1,1).expand(B, H).contiguous()
                ks_2d = ks.view(1,1).expand(B, H).contiguous()
                vs_2d = vs.view(1,1).expand(B, H).contiguous()

                # Find the right param names
                descale_params = [p for p in params if "descale" in p]
                print(f"  Descale param names: {descale_params}")

                kwargs = {"softmax_scale": D**-0.5, "causal": False}
                for p in descale_params:
                    if "q" in p: kwargs[p] = qs_2d
                    elif "k" in p: kwargs[p] = ks_2d
                    elif "v" in p: kwargs[p] = vs_2d

                out = func(fp8_q, fp8_k, fp8_v, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]
                print(f"  FP8 test passed! Output shape: {out.shape}, dtype: {out.dtype}")
                print(f"  Output has NaN: {torch.isnan(out).any()}")
                print(f"  Output has Inf: {torch.isinf(out).any()}")

                # Compare with BF16
                out_bf16 = func(q, k, v, softmax_scale=D**-0.5, causal=False)
                if isinstance(out_bf16, tuple):
                    out_bf16 = out_bf16[0]
                diff = (out.float() - out_bf16.float()).abs().mean().item()
                print(f"  Mean abs diff vs BF16: {diff:.6f}")
            else:
                print("  vllm flash_attn_func has no descale params")
        else:
            print("  No flash_attn_func found in vllm_flash_attn")
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
