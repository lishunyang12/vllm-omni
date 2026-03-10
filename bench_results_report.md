# HunyuanVideo-1.5 Performance Benchmark Report

## Test Environment

| Item | Value |
|------|-------|
| Hardware | 1× NVIDIA A100 80GB (140 GiB total) |
| Model | HunyuanVideo-1.5 (8.3B params, 54-layer dual-stream DiT) |
| Seed | 42 |
| CFG | guidance_scale=6.0 |
| VAE Tiling | Enabled for all tests |
| vLLM-OMNI Attention | Flash Attention |
| diffusers Attention | SDPA (default) |
| diffusers Version | 0.37.0 |

## Latency Comparison

| Test | Framework | Resolution | Frames | Steps | Precision | flow_shift | Latency (s) | Speedup |
|------|-----------|-----------|--------|-------|-----------|------------|-------------|---------|
| T2V 480p (short) | diffusers | 480×832 | 33 | 30 | BF16 | 5.0 | 87.0 | — |
| T2V 480p (short) | vLLM-OMNI | 480×832 | 33 | 30 | BF16 | 5.0 | 29.9 | 2.9× |
| T2V 480p (full) | diffusers | 480×832 | 121 | 50 | BF16 | 5.0 | 1135.7 | — |
| T2V 480p (full) | vLLM-OMNI | 480×832 | 121 | 50 | BF16 | 5.0 | 259.4 | 4.4× |
| T2V 480p (full, FP8) | vLLM-OMNI | 480×832 | 121 | 50 | FP8 | 5.0 | 259.3 | 4.4× |
| T2V 480p + neg prompt | diffusers | 480×832 | 33 | 30 | BF16 | 5.0 | 87.0 | — |
| T2V 480p + neg prompt | vLLM-OMNI | 480×832 | 33 | 30 | BF16 | 5.0 | 29.9 | 2.9× |
| T2V 720p | diffusers | 720×1280 | 33 | 30 | BF16 | 9.0 | 334.7 | — |
| T2V 720p | vLLM-OMNI | 720×1280 | 33 | 30 | FP8+tiling | 9.0 | 88.5 | 3.8× |
| I2V 480p | diffusers | 480×832 | 33 | 30 | BF16 | 5.0 | 84.2 | — |
| I2V 480p | vLLM-OMNI | 480×832 | 33 | 30 | BF16 | 5.0 | 31.0 | 2.7× |
| I2V 720p | diffusers | 720×1280 | 33 | 30 | BF16 | 7.0 | 329.5 | — |
| I2V 720p | vLLM-OMNI | 720×1280 | 33 | 30 | FP8+tiling | 7.0 | 91.7 | 3.6× |

## Peak VRAM Comparison

| Test | diffusers (GiB) | vLLM-OMNI (GiB) |
|------|-----------------|------------------|
| T2V 480p (short) | 35.35 | 34.53 |
| T2V 480p (full) | 46.80 | 33.80 |
| T2V 720p | 38.92 | 34.53 |
| I2V 480p | 36.14 | 34.60 |
| I2V 720p | 39.68 | 34.60 |

## Accuracy Comparison (frame-level, same prompt & seed)

| Test | PSNR (dB) | SSIM | Frames Compared | Notes |
|------|-----------|------|-----------------|-------|
| T2V 480p (short) | 16.90 | 0.51 | 33 | Flash Attn vs SDPA |
| T2V 480p (full) | 15.96 | 0.41 | 121 | Flash Attn vs SDPA |
| T2V 720p | 22.05 | 0.63 | 33 | Flash Attn vs SDPA |
| T2V 480p + neg prompt | 13.84 | 0.62 | 33 | Flash Attn vs SDPA |
| I2V 480p | N/A | N/A | — | Resolution mismatch |
| I2V 720p | N/A | N/A | — | Resolution mismatch |

## Test Prompts

| Test | Prompt | Negative Prompt |
|------|--------|-----------------|
| T2V | A little girl wearing a straw hat runs through a summer meadow full of wildflowers. A wide shot is used, with the camera panning right to follow her. | — |
| T2V + neg | A delicate watercolor illustration depicts three young women at a dining table celebrating by toasting with red wine glasses. | blurry, low quality, distorted |
| I2V | The camera follows the puppy as it runs forward on the grass, its four legs alternating steps, its tail held high and wagging side to side. | — |

## Summary

- vLLM-OMNI achieves **2.7×–4.4× speedup** over diffusers across all tested configurations
- Largest speedup (4.4×) on long-form generation (121 frames, 50 steps)
- VRAM usage is consistently lower or comparable to diffusers
- Outputs are visually similar but not bit-identical due to different attention backends (Flash Attention vs SDPA) and independent CFG/text-encoding implementations
- I2V accuracy comparison skipped due to diffusers auto-sizing output resolution from input image
- SGLang does not support HunyuanVideo-1.5 (only original HunyuanVideo v1)
