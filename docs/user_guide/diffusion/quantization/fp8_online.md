# Online FP8 Quantization

Online FP8 converts BF16/FP16 weights to FP8 at model load time. No calibration or pre-quantized checkpoint needed.

## Quick Start

Depending on the model, either all layers can be quantized, or some sensitive layers should stay in BF16. See the [per-model table](#per-model-recommendations) for which case applies.

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

**CLI / Online serving:**

```bash
# All layers
python text_to_image.py --model <your-model> --quantization fp8 ...

# Skip sensitive layers
python text_to_image.py --model <your-model> --quantization fp8 --ignored-layers "img_mlp" ...

# Online serving
vllm serve <your-model> --omni --quantization fp8
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | — | Quantization method (`"fp8"`) |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16 |
| `activation_scheme` | str | `"dynamic"` | `"dynamic"` (no calibration) or `"static"` |
| `weight_block_size` | list[int] \| None | `None` | Block size for block-wise weight quantization |

The available `ignored_layers` names depend on the model architecture (e.g., `to_qkv`, `to_out`, `img_mlp`, `txt_mlp`, `proj_out`). Consult the transformer source for your target model.

## Per-Model Recommendations

!!! note
    Benchmarks are for general reference. Actual performance varies by hardware and settings.

| Model | HF Models | Recommendation | `ignored_layers` |
|-------|-----------|---------------|------------------|
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | All layers | None |
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Skip sensitive layers | `img_mlp` |

### Z-Image

All layers are FP8-safe.

| Config | Time (s) | Speedup |
|--------|----------|---------|
| BF16 baseline | 3.42 | 1.00x |
| FP8 (all layers) | 2.85 | **1.20x** |

### Qwen-Image-2512

The `img_mlp` layer is sensitive — keep it in BF16.

| Config | Time (s) | Speedup |
|--------|----------|---------|
| BF16 baseline | 6.79 | 1.00x |
| FP8 (all layers) | 5.30 | **1.28x** |
| FP8, skip `img_mlp` (recommended) | 6.14 | **1.11x** |

<details>
<summary>Full ablation (14 configs)</summary>

| # | Config | to_qkv | add_kv_proj | to_out | to_add_out | img_mlp | txt_mlp | Time (s) |
|---|--------|:------:|:-----------:|:------:|:----------:|:-------:|:-------:|----------|
| 1 | BF16 baseline | BF16 | BF16 | BF16 | BF16 | BF16 | BF16 | 6.79 |
| 2 | FP8 all layers | FP8 | FP8 | FP8 | FP8 | FP8 | FP8 | 5.30 |
| 3 | skip text attn KV | FP8 | BF16 | FP8 | FP8 | FP8 | FP8 | 5.33 |
| 4 | skip text attn out | FP8 | FP8 | FP8 | BF16 | FP8 | FP8 | 5.33 |
| 5 | skip text attn all | FP8 | BF16 | FP8 | BF16 | FP8 | FP8 | 5.31 |
| 6 | skip img attn QKV | BF16 | FP8 | FP8 | FP8 | FP8 | FP8 | 5.78 |
| 7 | skip img attn out | FP8 | FP8 | BF16 | FP8 | FP8 | FP8 | 5.36 |
| 8 | skip img attn all | BF16 | FP8 | BF16 | FP8 | FP8 | FP8 | 5.85 |
| 9 | skip img_mlp | FP8 | FP8 | FP8 | FP8 | **BF16** | FP8 | 6.14 |
| 10 | skip txt_mlp | FP8 | FP8 | FP8 | FP8 | FP8 | BF16 | 5.32 |
| 11 | skip all MLP | FP8 | FP8 | FP8 | FP8 | BF16 | BF16 | 6.18 |
| 12 | skip all attn | BF16 | BF16 | BF16 | BF16 | FP8 | FP8 | 5.88 |
| 13 | skip text stream | FP8 | BF16 | FP8 | BF16 | FP8 | BF16 | 5.32 |
| 14 | skip attn + txt_mlp | BF16 | BF16 | BF16 | BF16 | FP8 | BF16 | 5.87 |

**Why `img_mlp` is sensitive:** Image MLP processes denoising latents with wide dynamic range across timesteps, has no normalization protection (unlike attention with QK-Norm), and errors compound across 60 residual blocks.

</details>

<!-- Template for adding new models:

### <Model Name>

<All layers safe / `<layer>` is sensitive — keep in BF16.>

| Config | Time (s) | Speedup |
|--------|----------|---------|
| BF16 baseline | X.XX | 1.00x |
| FP8 (all layers) | X.XX | **X.XXx** |

-->

## Combining with Other Features

```python
omni = Omni(
    model="<your-model>",
    quantization="fp8",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```

## Extending

**New model:** Accept `quant_config` in your transformer, pass it to linear layers via `get_vllm_quant_config_for_layers()`, then benchmark and add results to the table above.

**New method:** Create a config class inheriting `DiffusionQuantizationConfig` and register it in `vllm_omni/diffusion/quantization/__init__.py`.
