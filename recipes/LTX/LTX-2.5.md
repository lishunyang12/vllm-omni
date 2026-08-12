# LTX-2.5

> Text-to-video and first-frame image-to-video generation with synchronized
> audio

## Summary

- Vendor: Lightricks
- Supported checkpoint:
  [`Lightricks/LTX-2.5-Diffusers`](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
- Tasks: one-stage T2V/I2V, two-stage distilled T2V, and Full/SFT one-stage T2V/I2V
- Modes: offline inference and OpenAI-compatible `/v1/videos` HTTP serving
- Maintainer: Community

LTX-2.5 generates video and synchronized 48 kHz stereo audio. vLLM-Omni
supports the following generation paths through three pipeline classes:

> **Checkpoint layout:** This integration directly supports only
> `Lightricks/LTX-2.5-Diffusers`. The raw `Lightricks/LTX-2.5` repository uses
> split artifacts and cannot be passed directly to `--model` on this path.

| Mode | `--model-class-name` | Output | Steps | Transformer |
|---|---|---:|---:|---|
| One-stage T2V | `LTX2Pipeline` | 960x544, 121 frames at 24 FPS | 8 | `transformer/` |
| First-frame I2V | `LTX2Pipeline` | 960x544, 121 frames at 24 FPS | 8 | `transformer/` |
| Two-stage distilled T2V | `LTX2DistilledPipeline` | 1920x1088, 121 frames at 24 FPS | 8 + 3 | `transformer/` |
| Full/SFT one-stage T2V/I2V | `LTX2FullPipeline` | 960x544, 121 frames at 24 FPS | 30 | `transformer_full/` |

The two-stage pipeline first generates at 960x544, applies the model's x2
latent upsampler, and runs the official three-step refinement tail. One-stage
requests default to the official eight-step sigma schedule and may override it
with an explicit `sigmas` list. Distilled two-stage requests may override
`stage_1_sigmas` and `stage_2_sigmas` independently.

## When to use this recipe

Use this recipe to reproduce the release-qualified single-B300 path, compare
the recorded modes, or start an online video endpoint. Use one-stage T2V
or I2V for the lowest latency, two-stage T2V for 1920x1088 output, and
Full/SFT T2V or I2V when the non-distilled `transformer_full/` weights are required.

## References

- [LTX-2.5-Diffusers checkpoint](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
- [LTX-2.5 model and license](https://huggingface.co/Lightricks/LTX-2.5)
- [Official LTX-2 implementation](https://github.com/Lightricks/LTX-2)
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
uses the model's H.264 CRF-18 first-frame round trip by default.

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

To override the one-stage schedule, add `--extra-body` with a sigma list, for
example `--extra-body '{"sigmas":[1.0,0.5,0.0]}'`. The number of
denoising steps is derived from the supplied schedule.

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

CRF 18 is the LTX-2.5 default. Add `--extra-body '{"image_crf":0}'` only
when an application explicitly needs to bypass the conditioning round trip.

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

To override either distilled phase, pass `--extra-body` with
`stage_1_sigmas` and/or `stage_2_sigmas`; an omitted phase keeps its official
schedule. Stage 2 re-noise uses the first value of its effective schedule.

### Full/SFT one-stage T2V

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

Full/SFT first-frame I2V uses the same conditioning path with 30 denoising
steps:

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Lightricks/LTX-2.5-Diffusers \
  --model-class-name LTX2FullPipeline \
  --image /absolute/path/to/first-frame.png \
  --prompt "The red fox walks forward while the camera tracks alongside." \
  --height 544 \
  --width 960 \
  --num-frames 121 \
  --num-inference-steps 30 \
  --frame-rate 24 \
  --fps 24 \
  --enforce-eager \
  --output ltx25-full-i2v.mp4
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

Request-level overrides use multipart JSON: `image_crf` for I2V, `sigmas`
for one-stage/Full, or independent `stage_1_sigmas` and `stage_2_sigmas`
for distilled two-stage.

Restart the server with `LTX2DistilledPipeline` or `LTX2FullPipeline` to run
the corresponding two-stage or Full/SFT command. `LTX2FullPipeline` accepts
the same first-frame multipart I2V request with 30 steps. Use the dimensions
and step count from the pipeline table.

## B300 validation

The generic offline examples and OpenAI-compatible online server were validated
on one NVIDIA B300 with `CUDNN_ATTN`; the four recorded online modes returned HTTP 200.
The following are single-process cold-run diagnostics, not warmed throughput
claims. Generation excludes MP4 encoding; end to end includes model loading.

Correctness is guarded against a pinned official Lightricks implementation.

| Mode | Generation | End to end | Peak GPU memory |
|---|---:|---:|---:|
| One-stage T2V · 960x544 | 4.487 s | 39.018 s | 78,640 MiB |
| First-frame I2V · 960x544 | 4.567 s | 37.388 s | 78,640 MiB |
| Two-stage distilled T2V · 1920x1088 | 13.770 s | 47.264 s | 109,666 MiB |
| Full/SFT T2V · 960x544 | 40.917 s | 77.137 s | 79,678 MiB |

All decoded outputs included synchronized 48 kHz stereo audio. Extended
1920x1088, 481-frame official-prompt results are available in the
[LTX-2.5 B300 gallery](https://lishunyang12.github.io/vllm-omni-rankings/scripts/ltx25_official_b300_1080p20s/).

### Feature qualification

| Feature | Status on LTX-2.5 | Notes |
|---|---|---|
| `CUDNN_ATTN` | Release-qualified | Recommended B300 path; all four modes passed offline and online. |
| `TORCH_SDPA` | Functional baseline | All four modes passed; intended for debugging and portability. |
| Native DP2 | Unverified | The run initialized two diffusion workers inside one stage replica; every request reported `replica_id=0`, so independent replica scheduling was not demonstrated. |
| HSDP2 | Capacity fallback | Output matched eager; peak memory decreased by 21.9%, with lower performance. |
| Distributed layerwise offload DP2 | Capacity fallback | Output matched eager; primary-rank peak memory decreased by 43.3%, with lower performance. |
| Whole-model CPU offload | Capacity fallback | Output matched eager; peak memory decreased by 35.3%, with lower performance. |
| VAE slicing | Release-qualified | Output matched eager. |
| VAE tiling | Release-qualified | Audio was bit-exact with eager; decoded-video mean SSIM was 0.9730 and mean PSNR was 36.72 dB; peak memory decreased by 5.6%. |
| TP2 / Ulysses SP2 | Experimental | Both completed, but neither passed the fixed-seed quality gate. |
| Regional `torch.compile` | Experimental | Generation completed, but the first run was slower and did not pass the fixed-seed quality gate. |
| FP8 | Experimental | Generation completed with 23.4% lower peak memory, but did not pass the quality gate. |
| Cache-DiT | Experimental | Threshold 0.12 preserved eager output but recorded zero cache steps; threshold 0.15 exercised caching but failed the fixed-seed quality gate. |
| `TRTLLM_ATTN` | Unsupported | The current kernel rejects LTX-2.5's head dimension of 64. |
| Ring SP2 | Unsupported | The tested path did not complete successfully. |
| TeaCache | Unsupported | LTX-2.5 has no validated TeaCache residual extractor or coefficient profile. |
| FlashAttention-3 on H100 | Unverified | No H100 was available; the installed Hopper extension cannot be validated on B300. |

## Constraints and unsupported paths

- The checkpoint must contain `transformer/`; Full/SFT additionally requires
  `transformer_full/`.
- `num_frames` must be `8k+1`. One-stage dimensions must be divisible by 32;
  two-stage final dimensions must be divisible by 64.
- One-stage defaults to the official eight-step schedule and Full/SFT defaults
  to 30 steps; both accept a custom `sigmas` list. Distilled two-stage
  accepts independent `stage_1_sigmas` and `stage_2_sigmas` lists.
- Set `num_frames` explicitly for online requests because the generic video
  API default is one frame.
- First-frame I2V is implemented for `LTX2Pipeline` and
  `LTX2FullPipeline`; only `LTX2Pipeline` has the recorded B300 quality
  result above. Distilled two-stage I2V and multi-frame conditioning are not
  release-qualified.
- `TRTLLM_ATTN` currently rejects the LTX-2.5 head dimension of 64. Use
  `CUDNN_ATTN` on B300.
- DLO, CPU offload, and HSDP are memory-capacity fallbacks, not latency
  accelerators for this model.
