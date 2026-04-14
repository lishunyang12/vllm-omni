# SageAttention 2.2 with Wan2.2

## Overview

vLLM-Omni can override the diffusion attention backend with the
`DIFFUSION_ATTENTION_BACKEND` environment variable. When SageAttention is
installed in the same Python environment as vLLM-Omni, setting
`DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN` routes Wan2.2 DiT attention through
`vllm_omni/diffusion/attention/backends/sage_attn.py`, which currently calls
the generic `sageattn(...)` API.

The key question is not only whether SageAttention is faster, but whether it
preserves output quality on a real Wan2.2 workload.

## Quick Check

```bash
python -c "import sageattention; print(sageattention.__file__)"
```

## Start Wan2.2 Without SageAttention

```bash
unset DIFFUSION_ATTENTION_BACKEND

CUDA_VISIBLE_DEVICES=6,7 \
vllm serve \
  /mnt/data1/huggingface/hub/models--Wan-AI--Wan2.2-I2V-A14B-Diffusers/snapshots/596658fd9ca6b7b71d5057529bbf319ecbc61d74 \
  --omni \
  --host 127.0.0.1 \
  --port 8099 \
  --tensor-parallel-size 2 \
  --log-stats
```

## Start Wan2.2 With SageAttention

```bash
export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN

CUDA_VISIBLE_DEVICES=6,7 \
vllm serve \
  /mnt/data1/huggingface/hub/models--Wan-AI--Wan2.2-I2V-A14B-Diffusers/snapshots/596658fd9ca6b7b71d5057529bbf319ecbc61d74 \
  --omni \
  --host 127.0.0.1 \
  --port 8099 \
  --tensor-parallel-size 2 \
  --log-stats
```

## Benchmark Setup

- vLLM server version: `0.19.0`
- PyTorch: `2.10.0+cu128`
- CUDA runtime: `12.8`
- GPUs: `NVIDIA H20-3e` x2 (`CUDA_VISIBLE_DEVICES=6,7`)
- Model: local Wan2.2 I2V snapshot at the path above
- Serving mode: `--tensor-parallel-size 2 --omni --log-stats`
- Input image: `https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/rabbit.png`
- Request: `1280x720`, `5s`, `16 fps`, `num_inference_steps=8`, fixed prompt, fixed negative prompt, `seed=42`
- Method: start a fresh server for each backend, wait for `/health`, run one warm-up request, then record the second request

## Performance Result

### Measured Request Latency

| Backend | Warm-up `server_inference_time_s` | Measured `server_inference_time_s` | Measured `artifact_ready_wall_s` | Measured MP4 Size |
|---------|----------------------------------:|-----------------------------------:|---------------------------------:|------------------:|
| Default `FLASH_ATTN` | 400.487 s | 409.379 s | 410.418 s | 4.1 MB |
| `SAGE_ATTN` | 146.054 s | 141.638 s | 142.672 s | 20 KB |

### Speedup

- Steady-state server inference: `409.379 / 141.638 = 2.89x`
- End-to-end artifact-ready latency: `410.418 / 142.672 = 2.88x`

Pure latency numbers make SageAttention look substantially faster.

## Effect Comparison

### Metric Definition

- **PSNR**: computed per frame on decoded RGB frames, then averaged across all 81 aligned frames
- **SSIM**: computed per frame with an 11x11 Gaussian window on each RGB channel, then averaged across channels and frames

### Cross-Backend Quality

| Comparison | Avg PSNR | Avg SSIM | Interpretation |
|------------|---------:|---------:|----------------|
| `FLASH_ATTN measured` vs `SAGE_ATTN warm-up` | 8.5283 dB | 0.177464 | Sage warm-up output already differs heavily from baseline |
| `FLASH_ATTN measured` vs `SAGE_ATTN measured` | 3.5492 dB | 0.001058 | Sage measured output is effectively unusable relative to baseline |

### Backend Self-Consistency

| Comparison | Avg PSNR | Avg SSIM | Interpretation |
|------------|---------:|---------:|----------------|
| `FLASH_ATTN warm-up` vs `FLASH_ATTN measured` | `inf` | 1.000000 | Baseline is fully deterministic for repeated identical requests |
| `SAGE_ATTN warm-up` vs `SAGE_ATTN measured` | 7.0682 dB | 0.000583 | SageAttention output is not stable across repeated identical requests |

### Additional Observations

- `wan22_real_steps8_sage_measured.mp4` is only about `20 KB`, while the baseline measured file is about `4.1 MB`
- Both videos decode as `1280x720`, `16 fps`, `81 frames`, so the mismatch is not due to shape or frame-count differences
- The decoded pixel statistics for `wan22_real_steps8_sage_measured.mp4` are:
  - mean pixel average: `0.0`
  - frame std average: `0.0`
- In other words, the measured SageAttention output collapsed to all-black frames

## Conclusion

For Wan2.2 I2V with the real `rabbit.png` input and `num_inference_steps=8` on
two H20 GPUs:

- **Performance**: enabling `SAGE_ATTN` improves latency by about `2.9x`
- **Quality / correctness**: enabling `SAGE_ATTN` is currently **not safe**

The current mainline SageAttention integration is faster, but it introduces a
serious correctness problem on this workload. The baseline backend is perfectly
repeatable under the same seed, while SageAttention produces a drastically
different warm-up result and then collapses to an all-black measured video on
the second identical request.

The practical conclusion is:

- use `FLASH_ATTN` for Wan2.2 I2V production validation in the current state
- do not enable `SAGE_ATTN` for this workload until the correctness issue is fixed
