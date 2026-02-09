# Quantization for Diffusion Transformers

vLLM-Omni supports quantization of DiT linear layers to reduce memory usage and accelerate inference.

## Supported Methods

| Method | Guide |
|--------|-------|
| FP8 | [Online FP8](fp8_online.md) |

## Device Compatibility

| GPU Generation | Compute Capability | FP8 Mode |
|---------------|-------------------|----------|
| Turing (SM 75+) | T4, RTX 2080 | Weight-only via Marlin kernel |
| Ada/Hopper (SM 89+) | RTX 4090, H100, H200 | Full W8A8 with native hardware |

Kernel selection is automatic.
