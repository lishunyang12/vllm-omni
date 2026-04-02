"""Debug script: check FA3 FP8 capabilities and attention mask behavior."""
import inspect

print("=" * 60)
print("1. FA3 FP8 descale support check")
print("=" * 60)

# Check fa3_fwd interface
try:
    from fa3_fwd_interface import _flash_attn_forward
    sig = inspect.signature(_flash_attn_forward)
    params = list(sig.parameters.keys())
    print(f"fa3_fwd params: {params}")
    has_descale = any("descale" in p for p in params)
    print(f"Has descale support: {has_descale}")
except Exception as e:
    print(f"fa3_fwd not available: {e}")

# Check flash_attn varlen
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    sig = inspect.signature(flash_attn_varlen_func)
    params = list(sig.parameters.keys())
    print(f"\nvarlen params: {params}")
    has_descale = any("descale" in p for p in params)
    print(f"Varlen has descale: {has_descale}")
except Exception as e:
    print(f"flash_attn varlen not available: {e}")

# Check flash_attn regular func
try:
    from flash_attn.flash_attn_interface import flash_attn_func
    sig = inspect.signature(flash_attn_func)
    params = list(sig.parameters.keys())
    print(f"\nflash_attn_func params: {params}")
    has_descale = any("descale" in p for p in params)
    print(f"flash_attn_func has descale: {has_descale}")
except Exception as e:
    print(f"flash_attn_func not available: {e}")

# Check FA3 version
print("\n" + "=" * 60)
print("2. Package versions")
print("=" * 60)
try:
    import flash_attn
    print(f"flash_attn version: {flash_attn.__version__}")
except Exception:
    print("flash_attn: not installed or no __version__")

try:
    import fa3_fwd_cuda
    print(f"fa3_fwd_cuda: available")
except Exception:
    print("fa3_fwd_cuda: not available")

import torch
print(f"torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Check ring_globals imports (what vllm-omni actually uses)
print("\n" + "=" * 60)
print("3. vllm-omni FA3 imports")
print("=" * 60)
try:
    from vllm_omni.diffusion.attention.backends.ring.ring_globals import (
        HAS_FA3,
        fa3_attn_func,
    )
    print(f"HAS_FA3: {HAS_FA3}")
    print(f"fa3_attn_func: {fa3_attn_func}")
    if fa3_attn_func is not None:
        sig = inspect.signature(fa3_attn_func)
        params = list(sig.parameters.keys())
        print(f"fa3_attn_func params: {params}")
        has_descale = any("descale" in p for p in params)
        print(f"fa3_attn_func has descale: {has_descale}")
except Exception as e:
    print(f"ring_globals import failed: {e}")

# Check if flash_attn_varlen_func is available through vllm-omni's utils
try:
    from vllm_omni.diffusion.attention.backends.utils.fa import (
        flash_attn_varlen_func,
    )
    sig = inspect.signature(flash_attn_varlen_func)
    params = list(sig.parameters.keys())
    print(f"\nvllm-omni varlen func params: {params}")
    has_descale = any("descale" in p for p in params)
    print(f"vllm-omni varlen has descale: {has_descale}")
except Exception as e:
    print(f"vllm-omni varlen not available: {e}")
