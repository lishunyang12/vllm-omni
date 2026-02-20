# Quantization for Diffusion Models

vLLM-Omni supports quantization of diffusion model components to reduce memory usage and accelerate inference. This includes DiT linear layers, text encoders, and VAEs.

## Supported Methods

| Method | Components | Guide |
|--------|-----------|-------|
| FP8 | DiT, Text Encoder, VAE | [FP8](fp8.md) |

## Quantization Scope

When `--quantization fp8` is enabled, the following components are quantized:

| Component | What Gets Quantized | Mechanism |
|-----------|-------------------|-----------|
| **DiT (transformer)** | `nn.Linear` layers | vLLM W8A8 FP8 compute (Ada/Hopper) or weight-only (older GPUs) |
| **Text encoder** | `nn.Linear` layers | FP8 weight storage, BF16 compute |
| **VAE** | `nn.Conv2d`, `nn.Conv3d` layers | FP8 weight storage, BF16 compute |

!!! note
    Not all models support all three components. See the [FP8 supported models table](fp8.md#supported-models) for per-model details.

## Device Compatibility

| GPU Generation | Example GPUs | FP8 Mode |
|---------------|-------------------|----------|
| Ada/Hopper (SM 89+) | RTX 4090, H100, H200 | Full W8A8 with native hardware (DiT) + weight storage (encoder/VAE) |
| Turing/Ampere (SM 75-86) | RTX 3090, A100 | Weight-only via Marlin kernel (DiT) + weight storage (encoder/VAE) |

Kernel selection is automatic.
