# FP8 Quantization

## Overview

FP8 quantization converts BF16/FP16 weights to FP8 at model load time. No calibration or pre-quantized checkpoint needed.

vLLM-Omni supports FP8 quantization for three types of diffusion model components:

| Component | Layer Types | Mechanism | Memory Savings |
|-----------|------------|-----------|---------------|
| **DiT (transformer)** | `nn.Linear` | vLLM W8A8 quantized linear layers | ~50% weights + compute speedup |
| **Text encoder** | `nn.Linear` | FP8 weight storage with hooks | ~50% weights |
| **VAE** | `nn.Conv2d`, `nn.Conv3d` | FP8 weight storage with hooks | ~50% weights |

### DiT Quantization

For DiT linear layers, vLLM-Omni uses vLLM's native FP8 W8A8 quantization infrastructure. On Ada/Hopper GPUs (SM 89+), this provides both memory savings and inference speedup through hardware-accelerated FP8 compute.

Depending on the model, either all layers can be quantized, or some sensitive layers should stay in BF16. See the [per-model table](#supported-models) for which case applies.

Common sensitive layers in DiT-based diffusion models include **image-stream MLPs** (`img_mlp`). These are particularly vulnerable to FP8 precision loss because they process denoising latents whose dynamic range shifts significantly across timesteps, and unlike attention projections (which benefit from QK-Norm stabilization), MLPs have no built-in normalization to absorb quantization error. In deep architectures (e.g., 60+ residual blocks), small per-layer errors compound and degrade output quality. Other layers such as **attention projections** (`to_qkv`, `to_out`) and **text-stream MLPs** (`txt_mlp`) are generally more robust due to normalization or more stable input statistics.

### Text Encoder and VAE Quantization

For text encoders and VAEs loaded via `from_pretrained()`, vLLM-Omni uses **FP8 weight-only storage**. Weights are stored in `float8_e4m3fn` and dequantized to BF16 before each forward pass. This saves ~50% memory with no accuracy loss since computation still happens in BF16.

This approach is necessary because:

- **Text encoders** use standard `nn.Linear` layers but are loaded outside vLLM's weight pipeline
- **VAEs** use `nn.Conv2d`/`nn.Conv3d` layers, for which PyTorch has no FP8 compute kernels

The hook mechanism ensures only one layer's BF16 weight exists in memory at a time:

```
At rest:     All weights stored in FP8 (half memory)
Pre-hook:    Dequantize current layer's weight to BF16
Forward:     Normal computation in BF16
Post-hook:   Re-quantize weight back to FP8 (free BF16)
```

## Configuration

1. **Python API**: set `quantization="fp8"`. To skip sensitive layers, use `quantization_config` with `ignored_layers`.

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# All layers quantized
omni = Omni(model="<your-model>", quantization="fp8")

# Skip sensitive layers
omni = Omni(
    model="<your-model>",
    quantization_config={
        "method": "fp8",
        "ignored_layers": ["<layer-name>"],
    },
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

2. **CLI**: pass `--quantization fp8` and optionally `--ignored-layers`.

```bash
# All layers
python text_to_image.py --model <your-model> --quantization fp8

# Skip sensitive layers
python text_to_image.py --model <your-model> --quantization fp8 --ignored-layers "img_mlp"

# Online serving
vllm serve <your-model> --omni --quantization fp8
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | — | Quantization method (`"fp8"`) |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16 |
| `activation_scheme` | str | `"dynamic"` | `"dynamic"` (no calibration) or `"static"` |
| `weight_block_size` | list[int] \| None | `None` | Block size for block-wise weight quantization |

The available `ignored_layers` names depend on the model architecture (e.g., `to_qkv`, `to_out`, `img_mlp`, `txt_mlp`). Consult the transformer source for your target model.

!!! note
    The `ignored_layers` parameter only applies to DiT linear layers. Text encoder and VAE FP8 weight storage is applied to all layers when quantization is enabled.

## Supported Models

| Model | HF Models | DiT FP8 | Text Encoder FP8 | VAE FP8 | `ignored_layers` |
|-------|-----------|:-------:|:-----------------:|:-------:|------------------|
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | ✅ | ✅ | — | None |
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | ✅ | ✅ | ✅ | `img_mlp` |
| Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` | ✅ | ✅ | ✅ | — |
| Qwen-Image-Edit-Plus | `Qwen/Qwen-Image-Layered` | ✅ | ✅ | ✅ | — |
| Wan 2.2 | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | ✅ | — | — | — |

## Combining with Other Features

FP8 quantization can be combined with cache acceleration:

```python
omni = Omni(
    model="<your-model>",
    quantization="fp8",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```
