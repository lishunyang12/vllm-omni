"""Debug: check which flash-attention builds are available and whether
the FP8 two-level accumulation fix is present."""
import importlib
import os
import sys

print("=" * 60)
print("1. fa3-fwd (current FP8 attention backend)")
print("=" * 60)
try:
    import fa3_fwd_interface
    print(f"  Location: {fa3_fwd_interface.__file__}")
    # Check if the source has two-level accumulation
    src_path = os.path.dirname(fa3_fwd_interface.__file__)
    print(f"  Package dir: {src_path}")
    # List files
    for f in sorted(os.listdir(src_path)):
        if f.endswith(('.py', '.so', '.pyd')):
            print(f"    {f}")
except Exception as e:
    print(f"  Not available: {e}")

try:
    import fa3_fwd_cuda
    print(f"\n  fa3_fwd_cuda location: {fa3_fwd_cuda.__file__}")
except Exception as e:
    print(f"\n  fa3_fwd_cuda: {e}")

print("\n" + "=" * 60)
print("2. vLLM's flash-attention (may have the fix)")
print("=" * 60)

# Check vLLM's internal flash-attn
paths_to_check = [
    "vllm.attention.backends.flash_attn",
    "vllm.vllm_flash_attn",
    "vllm._custom_ops",
]
for mod_path in paths_to_check:
    try:
        mod = importlib.import_module(mod_path)
        print(f"  {mod_path}: {mod.__file__}")
    except Exception as e:
        print(f"  {mod_path}: not available ({e})")

# Check if vLLM ships its own flash_attn_func with descale
try:
    from vllm.attention.backends.flash_attn import flash_attn_varlen_func
    import inspect
    sig = inspect.signature(flash_attn_varlen_func)
    params = list(sig.parameters.keys())
    has_descale = any("descale" in p for p in params)
    print(f"\n  vLLM flash_attn_varlen_func params: {params}")
    print(f"  Has descale: {has_descale}")
except Exception as e:
    print(f"\n  vLLM flash_attn_varlen_func: {e}")

print("\n" + "=" * 60)
print("3. flash_attn pip package")
print("=" * 60)
try:
    import flash_attn
    print(f"  Version: {flash_attn.__version__}")
    print(f"  Location: {flash_attn.__file__}")
except Exception as e:
    print(f"  Not installed: {e}")

print("\n" + "=" * 60)
print("4. Check for two-level accumulation in fa3-fwd source")
print("=" * 60)
try:
    import fa3_fwd_interface
    src_file = fa3_fwd_interface.__file__
    with open(src_file, 'r') as f:
        content = f.read()
    # Search for signs of two-level accumulation
    keywords = [
        "two_level", "TWO_LEVEL",
        "accum_fp32", "ACCUM_FP32",
        "fp8_two_level", "FP8_TWO_LEVEL",
        "accumulation_fix",
        "flush_accum",
    ]
    found = False
    for kw in keywords:
        if kw.lower() in content.lower():
            print(f"  Found '{kw}' in fa3_fwd_interface.py")
            found = True
    if not found:
        print("  No two-level accumulation keywords found in fa3_fwd_interface.py")
        print("  -> This build likely does NOT have the FP8 accumulation fix")
except Exception as e:
    print(f"  Could not read source: {e}")

print("\n" + "=" * 60)
print("5. vllm-project/flash-attention fork check")
print("=" * 60)
# Check if there's a vllm flash_attn with the fix
search_paths = [
    "/workspace/.venv/lib/python3.12/site-packages/vllm",
    "/workspace/.venv/lib/python3.12/site-packages/flash_attn",
    "/workspace/.venv/lib/python3.12/site-packages",
]
for base in search_paths:
    if os.path.isdir(base):
        for root, dirs, files in os.walk(base):
            for f in files:
                if "flash" in f.lower() and f.endswith('.so'):
                    full = os.path.join(root, f)
                    size_mb = os.path.getsize(full) / 1024 / 1024
                    print(f"  {full} ({size_mb:.1f} MB)")
            # Don't recurse too deep
            if root.count(os.sep) - base.count(os.sep) > 2:
                dirs.clear()

print("\n" + "=" * 60)
print("6. Environment variable check")
print("=" * 60)
env_vars = [
    "FLASH_ATTENTION_DISABLE_FP8_TWO_LEVEL_ACCUMULATION",
    "FLASH_ATTENTION_FORCE_FP8_TWO_LEVEL_ACCUMULATION",
    "VLLM_FLASH_ATTN_SRC_DIR",
]
for var in env_vars:
    val = os.environ.get(var, "<not set>")
    print(f"  {var}: {val}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
