# LTX-2.5

> Text-to-video and first-frame image-to-video generation with synchronized
> audio

## Summary

- Vendor: Lightricks
- Model:
  [`Lightricks/LTX-2.5-Diffusers`](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
- Tasks: one-stage T2V/I2V, two-stage distilled T2V, and Full/SFT T2V
- Modes: offline inference and OpenAI-compatible `/v1/videos` HTTP serving
- Maintainer: Community

LTX-2.5 generates video and synchronized 48 kHz stereo audio. vLLM-Omni
supports four generation modes through three pipeline classes:

| Mode | `--model-class-name` | Output | Steps | Transformer |
|---|---|---:|---:|---|
| One-stage T2V | `LTX2Pipeline` | 960x544, 121 frames at 24 FPS | 8 | `transformer/` |
| First-frame I2V | `LTX2Pipeline` | 960x544, 121 frames at 24 FPS | 8 | `transformer/` |
| Two-stage distilled T2V | `LTX2DistilledPipeline` | 1920x1088, 121 frames at 24 FPS | 8 + 3 | `transformer/` |
| Full/SFT T2V | `LTX2FullPipeline` | 960x544, 121 frames at 24 FPS | 30 | `transformer_full/` |

The two-stage pipeline first generates at 960x544, applies the model's x2
latent upsampler, and runs the official three-step refinement tail.

## When to use this recipe

Use this recipe to reproduce the release-qualified single-B300 path, compare
the four supported modes, or start an online video endpoint. Use one-stage T2V
or I2V for the lowest latency, two-stage T2V for 1920x1088 output, and
Full/SFT T2V when the non-distilled `transformer_full/` weights are required.

## References

- [LTX-2.5-Diffusers checkpoint](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
- [LTX-2.5 model and license](https://huggingface.co/Lightricks/LTX-2.5)
- [Diffusers LTX-2.5 integration PR](https://github.com/huggingface/diffusers/pull/14447)
- [Text-to-video offline example](../../examples/offline_inference/text_to_video/text_to_video.py)
- [Image-to-video offline example](../../examples/offline_inference/image_to_video/image_to_video.py)

## Prerequisites

The checkpoint is gated. Accept its Hugging Face license and authenticate
before starting a download:

```bash
hf auth login
export MODEL=Lightricks/LTX-2.5-Diffusers
```

Install matching vLLM and vLLM-Omni versions. The validation environment used
Python 3.12, vLLM 0.27.0, vLLM-Omni 0.27.0.dev106, PyTorch 2.13.0+cu130, and
one NVIDIA B300:

```bash
uv venv --python 3.12
source .venv/bin/activate
export VLLM_VERSION=0.27.0
uv pip install "vllm==${VLLM_VERSION}" --torch-backend=auto
uv pip install -e .
```

`ffmpeg` and `ffprobe` must be on `PATH` for MP4 output. I2V also requires
PyAV backed by an FFmpeg build with `libx264`; the LTX-2.5 conditioning path
performs the model's required H.264 CRF-18 first-frame round trip.

## Offline inference

The commands below use the existing generic offline examples; no
model-specific runner is required.

### One-stage T2V

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2.5-Diffusers \
  --model-class-name LTX2Pipeline \
  --prompt "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot." \
  --height 544 \
  --width 960 \
  --num-frames 121 \
  --num-inference-steps 8 \
  --frame-rate 24 \
  --fps 24 \
  --enforce-eager \
  --output ltx25-one-stage.mp4
```

### First-frame I2V

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Lightricks/LTX-2.5-Diffusers \
  --model-class-name LTX2Pipeline \
  --image /absolute/path/to/first-frame.png \
  --prompt "The red fox walks forward while the camera tracks alongside." \
  --height 544 \
  --width 960 \
  --num-frames 121 \
  --num-inference-steps 8 \
  --frame-rate 24 \
  --fps 24 \
  --enforce-eager \
  --output ltx25-one-stage-i2v.mp4
```

### Two-stage distilled T2V

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2.5-Diffusers \
  --model-class-name LTX2DistilledPipeline \
  --prompt "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot." \
  --height 1088 \
  --width 1920 \
  --num-frames 121 \
  --num-inference-steps 8 \
  --frame-rate 24 \
  --fps 24 \
  --enforce-eager \
  --output ltx25-two-stage.mp4
```

### Full/SFT T2V

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2.5-Diffusers \
  --model-class-name LTX2FullPipeline \
  --prompt "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot." \
  --height 544 \
  --width 960 \
  --num-frames 121 \
  --num-inference-steps 30 \
  --frame-rate 24 \
  --fps 24 \
  --enforce-eager \
  --output ltx25-full.mp4
```

## Online serving

The release-qualified B300 path uses cuDNN attention explicitly:

```bash
export MODEL=Lightricks/LTX-2.5-Diffusers
export PORT=8000

CUDA_VISIBLE_DEVICES=0 \
vllm serve "${MODEL}" \
  --omni \
  --model-class-name LTX2Pipeline \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

Check readiness before submitting a generation request:

```bash
curl -sS -o /dev/null -w 'health HTTP %{http_code}\n' \
  "http://127.0.0.1:${PORT}/health"
```

Run one-stage T2V through the synchronous video endpoint:

```bash
curl -sS --fail-with-body \
  -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
  -F 'prompt=A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot.' \
  -F 'width=960' \
  -F 'height=544' \
  -F 'num_frames=121' \
  -F 'fps=24' \
  -F 'num_inference_steps=8' \
  -F 'seed=42' \
  -o ltx25-online.mp4
```

For first-frame I2V, add an image to the same request:

```bash
export FIRST_FRAME=/absolute/path/to/first-frame.png

curl -sS --fail-with-body \
  -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
  -F 'prompt=The red fox walks forward while the camera tracks alongside.' \
  -F "input_reference=@${FIRST_FRAME};type=image/png" \
  -F 'width=960' \
  -F 'height=544' \
  -F 'num_frames=121' \
  -F 'fps=24' \
  -F 'num_inference_steps=8' \
  -F 'seed=42' \
  -o ltx25-online-i2v.mp4
```

Restart the server with `LTX2DistilledPipeline` or `LTX2FullPipeline` to run
the corresponding two-stage or Full/SFT T2V command. Use the dimensions and
step count from the pipeline table.

## B300 validation

The generic offline examples and the OpenAI-compatible online server were
validated on one NVIDIA B300 with `CUDNN_ATTN`. All four modes returned H.264
video with synchronized 48 kHz stereo AAC and a duration of 5.0417 seconds.

| Mode | Public offline generation | Online status | Output |
|---|---:|---:|---|
| One-stage T2V | 4.323 s | HTTP 200 | 960x544, 121 frames |
| First-frame I2V | 4.627 s | HTTP 200 | 960x544, 121 frames |
| Two-stage distilled T2V | 12.954 s | HTTP 200 | 1920x1088, 121 frames |
| Full/SFT T2V | 39.134 s | HTTP 200 | 960x544, 121 frames |

These are single-run diagnostics, not warmed throughput claims.

### Feature qualification

| Feature | Status on LTX-2.5 | Notes |
|---|---|---|
| `CUDNN_ATTN` | Release-qualified | Recommended B300 path; all four modes passed. |
| `TORCH_SDPA` | Functional baseline | All four modes passed; intended for debugging and portability. |
| Native DP2 | Unverified | The current run initialized two workers inside one stage replica; independent replica scheduling was not demonstrated. |
| HSDP2 | Capacity fallback | Output matched eager; peak memory decreased, with lower performance. |
| Distributed layerwise offload DP2 | Capacity fallback | Output matched eager; primary-rank peak memory decreased by 43.3%, with lower performance. |
| VAE slicing | Release-qualified | Output matched eager. |
| VAE tiling | Release-qualified | Audio matched eager; decoded-video SSIM was 0.9867 and PSNR was 40.29 dB; peak memory decreased by 5.6%. |
| Whole-model CPU offload | Capacity fallback | Output matched eager; peak memory decreased by 35.3%, with lower performance. |
| TP2 / Ulysses SP2 | Experimental | Both completed, but did not pass the fixed-seed quality gate. |
| Regional `torch.compile` | Experimental | Generation completed, but the first run was slower and did not pass the fixed-seed quality gate. |
| FP8 | Experimental | Generation completed with lower memory, but did not pass the quality gate. |
| Cache-DiT | Experimental | Threshold 0.12 produced no cache hit; threshold 0.15 produced one hit but failed the quality gate. |
| `TRTLLM_ATTN` / Ring SP2 / TeaCache | Unsupported | Current kernels or model-specific adapters do not support these paths. |
| FlashAttention-3 on H100 | Unverified | No H100 was available; the installed Hopper extension cannot be validated on B300. |

## Constraints and unsupported paths

- The checkpoint must contain `transformer/`; Full/SFT additionally requires
  `transformer_full/`.
- `num_frames` must be `8k+1`. One-stage dimensions must be divisible by 32;
  two-stage final dimensions must be divisible by 64.
- One-stage and two-stage distilled modes use the fixed eight-step schedule.
  Full/SFT uses 30 steps.
- Set `num_frames` explicitly for online requests because the generic video
  API default is one frame.
- Only first-frame I2V through `LTX2Pipeline` is release-qualified. Two-stage
  I2V and Full/SFT I2V are not claimed by this recipe.
- `TRTLLM_ATTN` currently rejects the LTX-2.5 head dimension of 64. Use
  `CUDNN_ATTN` on B300.
- DLO, CPU offload, and HSDP are memory-capacity fallbacks, not latency
  accelerators for this model.
