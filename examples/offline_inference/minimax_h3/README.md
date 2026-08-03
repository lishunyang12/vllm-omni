# MiniMax-H3 four-GPU all-task offline runner

This example runs every MiniMax-H3 task path currently implemented by
vLLM-Omni in one command. It targets four NVIDIA RTX PRO 5000 72GB Blackwell
(SM120) cards and uses the same capacity-safe topology validated on four B300
GPUs. The 48 GB RTX PRO 5000 variant is below this resident recipe's measured
memory requirement and is rejected by the launch preflight.

| Order | Checkpoint | vLLM-Omni task | Condition |
| --- | --- | --- | --- |
| 1 | `FL2VA` | `t2va` | Text only |
| 2 | `FL2VA` | `fl2va` | One first-frame image |
| 3 | `Ref2VA` | `ref2va` | One image and one audio file |
| 4 | `Ref2VA` | `ref2va` | Two videos and their soundtracks |

The first T2VA output supplies the image, audio, and video assets used by the
remaining tasks, so no separate reference media is required. vLLM-Omni does
not currently expose last-frame-only or first-plus-last-frame FL2VA as separate
offline input signatures; those SGLang-specific combinations are not claimed
by this example.

## Prerequisites

Install vLLM and this checkout in a Python 3.12 virtual environment. The runner
uses cuDNN attention because official FlashAttention-4 kernels do not target
SM120.

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm==0.26.0 --torch-backend=auto
uv pip install -e .
```

Both independently served checkpoint partitions are required:

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 \
  --include 'FL2VA/**' \
  --local-dir "${MODEL_ROOT}"
hf download MiniMaxAI/MiniMax-H3 \
  --include 'Ref2VA/**' \
  --local-dir "${MODEL_ROOT}"
```

Use `hf auth login` first if the checkpoint requires approved Hugging Face
access. `ffmpeg`, `ffprobe`, and `nvidia-smi` must be on `PATH`.

## Run all tasks

From the repository root:

```bash
MODEL_ROOT=/path/to/MiniMax-H3 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/offline_inference/minimax_h3/run_4gpu_all_tasks.sh
```

The default workload is 1344x768, 24 FPS, 5 seconds, and 50 denoising steps.
Override it with environment variables:

```bash
MODEL_ROOT=/path/to/MiniMax-H3 \
OUTPUT_DIR=/path/to/results/minimax-h3-all-tasks \
NUM_INFERENCE_STEPS=5 \
DURATION_SECONDS=4 \
bash examples/offline_inference/minimax_h3/run_4gpu_all_tasks.sh
```

Useful controls:

- `PYTHON`: virtual-environment Python executable.
- `WORK_ROOT`: location for Hugging Face, TorchInductor, and Triton caches.
- `INSTALL_EDITABLE=0`: skip the default `pip install --no-deps -e` step.
- `ENFORCE_EAGER=1`: disable regional `torch.compile` for debugging.
- `MAX_PREFLIGHT_MEMORY_MIB` and `MAX_PREFLIGHT_GPU_UTIL`: change the safety
  thresholds that prevent running on an occupied GPU.
- `MIN_GPU_MEMORY_MIB`: minimum physical memory per card; defaults to 70000 MiB
  to select the 72 GB RTX PRO 5000 rather than the 48 GB variant.

The launch uses TP4, Ulysses1, Ring1, text-encoder TP4, and native tiled VAE
patch parallelism across all four GPUs. It intentionally keeps checkpoint
BF16/FP32 precision and disables the experimental online FP8 path.

## Outputs

The output directory contains:

- `01_t2va.mp4`
- `02_fl2va_first_frame.mp4`
- `03_ref2va_image_audio.mp4`
- `04_ref2va_two_videos.mp4`
- `summary.json` with per-task latency, stage timings, memory, shapes, and hashes
- one `ffprobe.json` file per MP4
- `gpu_peak_memory.csv`, `nvidia-smi.csv`, and `artifact_sha256.txt`

Each checkpoint partition is loaded once. Its first request includes regional
compilation, so the four task timings are functional end-to-end measurements,
not steady-state benchmark medians.
