## FP8 Quantization Benchmark

### Test Environment

| Item | Value |
|------|-------|
| GPU | |
| Driver | |
| CUDA | |
| PyTorch | |
| vLLM-OMNI | |

---

### Z-Image (Tongyi-MAI/Z-Image-Turbo)

#### Performance

| Metric | BF16 | FP8 | Change |
|--------|------|-----|--------|
| Peak GPU Memory (GB) | | | |
| Inference Time (50 steps, s) | | | |

#### Visual Comparison (seed=42, 1024x1024, 50 steps)

| Prompt | BF16 | FP8 |
|--------|------|-----|
| "a cup of coffee on the table" | | |
| "a cat sitting on a windowsill at sunset" | | |
| "a mountain landscape with a lake reflection" | | |

---

### Qwen-Image (Qwen/Qwen-Image)

#### Visual Comparison (seed=42, 1024x1024, 50 steps)

| Prompt | BF16 | FP8 |
|--------|------|-----|
| "a cup of coffee on the table" | | |
| "a mountain landscape with a lake reflection" | | |

#### Performance

| Metric | BF16 | FP8 | Change |
|--------|------|-----|--------|
| Peak GPU Memory (GB) | | | |
| Inference Time (50 steps, s) | | | |

---

### Qwen-Image FP8 `ignored_layers` Ablation (seed=42, 1024x1024, 50 steps)

Qwen-Image uses dual-stream joint attention. The text-side projections
(`add_kv_proj`, `to_add_out`) feed into the shared softmax and may be
more sensitive to FP8 rounding than Z-Image's single-stream architecture.

**Prompts used:**
- Coffee: `"a cup of coffee on the table"`
- Mountain: `"a mountain landscape with a lake reflection"`

#### Visual Quality

| Strategy | Ignored Layers | Coffee | Mountain | Loaded Model Memory (GB) |
|----------|---------------|--------|----------|--------------------------|
| BF16 (baseline) | N/A | | | |
| FP8 all layers | none | | | |
| FP8 skip text attn | `add_kv_proj`, `to_add_out` | | | |
| FP8 skip all attn | `to_qkv`, `add_kv_proj`, `to_out`, `to_add_out` | | | |
| FP8 skip text QKV only | `add_kv_proj` | | | |

#### CLI Commands

```bash
# 1. BF16 (baseline)
python text_to_image.py --model Qwen/Qwen-Image \
  --prompt "a cup of coffee on the table" --seed 42 \
  --height 1024 --width 1024 --output outputs/qwen_coffee_bf16.png

python text_to_image.py --model Qwen/Qwen-Image \
  --prompt "a mountain landscape with a lake reflection" --seed 42 \
  --height 1024 --width 1024 --output outputs/qwen_mountain_bf16.png

# 2. FP8 all layers
python text_to_image.py --model Qwen/Qwen-Image \
  --prompt "a cup of coffee on the table" --seed 42 \
  --height 1024 --width 1024 --quantization fp8 \
  --output outputs/qwen_coffee_fp8_all.png

python text_to_image.py --model Qwen/Qwen-Image \
  --prompt "a mountain landscape with a lake reflection" --seed 42 \
  --height 1024 --width 1024 --quantization fp8 \
  --output outputs/qwen_mountain_fp8_all.png

# 3. FP8 skip text attn (ignored_layers: add_kv_proj, to_add_out)
python text_to_image.py --model Qwen/Qwen-Image \
  --prompt "a cup of coffee on the table" --seed 42 \
  --height 1024 --width 1024 \
  --quantization_config '{"method": "fp8", "ignored_layers": ["add_kv_proj", "to_add_out"]}' \
  --output outputs/qwen_coffee_fp8_skip_text_attn.png

python text_to_image.py --model Qwen/Qwen-Image \
  --prompt "a mountain landscape with a lake reflection" --seed 42 \
  --height 1024 --width 1024 \
  --quantization_config '{"method": "fp8", "ignored_layers": ["add_kv_proj", "to_add_out"]}' \
  --output outputs/qwen_mountain_fp8_skip_text_attn.png

# 4. FP8 skip all attn (ignored_layers: to_qkv, add_kv_proj, to_out, to_add_out)
python text_to_image.py --model Qwen/Qwen-Image \
  --prompt "a cup of coffee on the table" --seed 42 \
  --height 1024 --width 1024 \
  --quantization_config '{"method": "fp8", "ignored_layers": ["to_qkv", "add_kv_proj", "to_out", "to_add_out"]}' \
  --output outputs/qwen_coffee_fp8_skip_all_attn.png

python text_to_image.py --model Qwen/Qwen-Image \
  --prompt "a mountain landscape with a lake reflection" --seed 42 \
  --height 1024 --width 1024 \
  --quantization_config '{"method": "fp8", "ignored_layers": ["to_qkv", "add_kv_proj", "to_out", "to_add_out"]}' \
  --output outputs/qwen_mountain_fp8_skip_all_attn.png

# 5. FP8 skip text QKV only (ignored_layers: add_kv_proj)
python text_to_image.py --model Qwen/Qwen-Image \
  --prompt "a cup of coffee on the table" --seed 42 \
  --height 1024 --width 1024 \
  --quantization_config '{"method": "fp8", "ignored_layers": ["add_kv_proj"]}' \
  --output outputs/qwen_coffee_fp8_skip_text_qkv.png

python text_to_image.py --model Qwen/Qwen-Image \
  --prompt "a mountain landscape with a lake reflection" --seed 42 \
  --height 1024 --width 1024 \
  --quantization_config '{"method": "fp8", "ignored_layers": ["add_kv_proj"]}' \
  --output outputs/qwen_mountain_fp8_skip_text_qkv.png
```

> **Note:** Tests 3-5 require `--quantization_config` CLI arg which is not yet
> wired up in `text_to_image.py`. You need to add it to the argparser first,
> or use the `Omni` Python API directly with `quantization_config={"method": "fp8", "ignored_layers": [...]}`.

---

### Unit Tests

```bash
pytest tests/diffusion/quantization/test_fp8_config.py -v
```
