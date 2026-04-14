# SageAttention 2.2 with Wan2.2

## Overview

vLLM-Omni can override the diffusion attention backend with the
`DIFFUSION_ATTENTION_BACKEND` environment variable. When SageAttention is
installed in the same Python environment as vLLM-Omni, setting
`DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN` routes Wan2.2 DiT attention through
`vllm_omni/diffusion/attention/backends/sage_attn.py`.

This tutorial focuses on the practical path:

1. Confirm SageAttention is importable.
2. Start Wan2.2 with and without `SAGE_ATTN`.
3. Compare steady-state latency on the same request.

## Quick Check

```bash
python -c "import sageattention; print(sageattention.__file__)"
```

If this import succeeds, vLLM-Omni can load the SageAttention backend.

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

With SageAttention enabled, startup logs should contain:

```text
Using diffusion attention backend 'SAGE_ATTN'
```

Without the override, the current default on this machine is:

```text
Defaulting to diffusion attention backend FLASH_ATTN
```

## Benchmark Request Script

The comparison below used the same async request flow in both runs and reported:

- `server_inference_time_s`: value returned by `GET /v1/videos/{video_id}`
- `artifact_ready_wall_s`: end-to-end wall time from submit to downloaded MP4

```python
import json
import os
import pathlib
import time

import requests

base = "http://127.0.0.1:8099"
image_path = "/tmp/rabbit.png"
out_path = "/tmp/wan22_bench_out.mp4"
prompt = "same fixed prompt for both runs"
negative = "same fixed negative prompt for both runs"

data = {
    "prompt": prompt,
    "negative_prompt": negative,
    "size": "1280x720",
    "seconds": "5",
    "fps": "16",
    "num_inference_steps": "1",
    "guidance_scale": "3.5",
    "guidance_scale_2": "3.5",
    "boundary_ratio": "0.875",
    "flow_shift": "5.0",
    "seed": "42",
}

with open(image_path, "rb") as img:
    t0 = time.perf_counter()
    create = requests.post(
        f"{base}/v1/videos",
        headers={"Accept": "application/json"},
        data=data,
        files={"input_reference": (os.path.basename(image_path), img, "image/png")},
        timeout=120,
    )

create.raise_for_status()
video_id = create.json()["id"]

while True:
    resp = requests.get(f"{base}/v1/videos/{video_id}", timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") in ("completed", "failed"):
        final_json = payload
        break
    time.sleep(1.0)

download = requests.get(f"{base}/v1/videos/{video_id}/content", timeout=120)
download.raise_for_status()
pathlib.Path(out_path).write_bytes(download.content)
t3 = time.perf_counter()

print(
    json.dumps(
        {
            "artifact_ready_wall_s": round(t3 - t0, 3),
            "server_inference_time_s": final_json.get("inference_time_s"),
            "final_status": final_json.get("status"),
        },
        ensure_ascii=False,
        indent=2,
    )
)
```

## Wan2.2 Measured Result

### Test Setup

- vLLM server version: `0.19.0`
- PyTorch: `2.10.0+cu128`
- CUDA runtime: `12.8`
- GPUs: `NVIDIA H20-3e` x2 (`CUDA_VISIBLE_DEVICES=6,7`)
- Model: local Wan2.2 I2V snapshot at the path above
- Serving mode: `--tensor-parallel-size 2 --omni --log-stats`
- Request: `1280x720`, `5s`, `16 fps`, `num_inference_steps=1`, fixed prompt, fixed negative prompt, `seed=42`
- Method: start a fresh server for each backend, wait for `/health`, run one warm-up request, then record the second request

### Result Table

| Backend | Warm-up `server_inference_time_s` | Measured `server_inference_time_s` | Measured `artifact_ready_wall_s` |
|---------|----------------------------------:|-----------------------------------:|---------------------------------:|
| Default `FLASH_ATTN` | 70.364 s | 70.209 s | 71.225 s |
| `SAGE_ATTN` | 40.185 s | 36.987 s | 38.001 s |

### Speedup

- Steady-state server inference: `70.209 / 36.987 = 1.90x`
- End-to-end artifact-ready latency: `71.225 / 38.001 = 1.87x`
- Relative reduction: about `47.3%` lower server inference time and `46.6%` lower end-to-end wall time

For this Wan2.2 I2V workload on two H20 GPUs, enabling SageAttention was a
clear win.

## Notes

- The first server-side dummy warm-up during initialization was slower with
  SageAttention than with FlashAttention on this machine, because
  `torch.compile` emitted graph-break warnings around SageAttention custom ops.
  That did not prevent serving, and steady-state request latency was still much
  better with `SAGE_ATTN`.
- The current vLLM-Omni SageAttention backend calls the generic `sageattn(...)`
  function. The numbers above are therefore the real behavior of the current
  mainline integration, not an estimate from upstream microbenchmarks.
- Re-run the comparison if you change GPU type, Torch/CUDA version, model
  resolution, frame count, or inference step count.
