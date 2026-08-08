# MiniMax-H3 BF16 vs default global FP8 outputs

These videos provide a direct quality comparison for the user-facing
`--quantization fp8` default added by this PR.

All pairs were generated from the released BF16 checkpoints with the same
official input, seed 0, 50 denoising steps, 24 FPS, 32 kHz stereo audio, and a
requested duration of 5 seconds. The FP8 run quantizes all eligible DiT and
Qwen3-VL text-decoder linears online. The vision tower, embeddings, norms,
RoPE, VAEs, and mixed-precision input/output heads retain checkpoint
precision.

| Task | BF16 output | Default global FP8 output | PSNR | SSIM | Audio correlation |
|---|---|---|---:|---:|---:|
| T2VA | [bf16/t2va.mp4](bf16/t2va.mp4) | [global_fp8/t2va.mp4](global_fp8/t2va.mp4) | 17.943 dB | 0.7290 | 0.9351 |
| I2VA | [bf16/i2va.mp4](bf16/i2va.mp4) | [global_fp8/i2va.mp4](global_fp8/i2va.mp4) | 29.403 dB | 0.9511 | 0.9704 |
| Ref2VA | [bf16/ref2va.mp4](bf16/ref2va.mp4) | [global_fp8/ref2va.mp4](global_fp8/ref2va.mp4) | 44.153 dB | 0.9839 | 0.9250 |

The metrics compare decoded outputs with the matching BF16 video. They are
descriptive rather than pass/fail thresholds. Diffusion trajectories can
diverge from small numerical differences even with the same seed, so inspect
motion, prompt adherence, subject consistency, visual artifacts, and audio
together.

The Ref2VA pair uses the official reference video and its embedded Audio 1.
It does not include a separate Audio 2 voice-timbre reference because the
current vLLM request path rejects a reference video plus an additional
independent audio condition.

## Provenance

- Benchmark source:
  [vllm-omni-rankings@802ef8e](https://github.com/lishunyang12/vllm-omni-rankings/tree/802ef8e4e61ada7ced8babf1366170d9df26f27d/scripts/minimax_h3_online_fp8)
- Hardware: 4 x NVIDIA B300 per request
- Parallelism: USP 4, text-encoder TP4, VAE patch parallel 4 (tile)
- Attention backend: cuDNN
- Warmup: one shape-matched two-step request before each measured output

## SHA256

```text
365d6e397583c8043ac88b232eea16e9239723c71a4ef450089671c5638ce86d  bf16/t2va.mp4
1a0173a1ef7f8d65e923669784a612acc9732c5fe60dc36598e2199aa146115c  bf16/i2va.mp4
b43e5eec3ce923def409db14dc0ca2d5cb59ae44b45b6da2a9bab5a0fab52145  bf16/ref2va.mp4
d852f135bf8f1da0489a889d2b87a6dcf663158f682844b87a552daafcdad199  global_fp8/t2va.mp4
7443cf5aed2dc82e02cfa70793af9b6a28ff0f0640b77ee6161c5d4997bb82fa  global_fp8/i2va.mp4
00755c6e15fa1f215d957f27bed88f35833ba7d7d7ebf5f5ec6e789bd3b47026  global_fp8/ref2va.mp4
```
