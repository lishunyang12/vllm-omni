# LTX-2.5

> Official LTX-2.5 text-to-video and first-frame image-to-video generation
> with synchronized audio

## Summary

- Vendor: Lightricks
- Model:
  [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5)
- Task: Full one-stage T2V/I2V, distilled two-stage T2V/I2V, and Full
  two-stage T2V/I2V
- Mode: offline inference and OpenAI-compatible `/v1/videos` HTTP serving
- Maintainer: Community

## When to use this recipe

Use this recipe when generating synchronized video and audio from text or a
single first-frame image with the official LTX-2.5 split checkpoint. Choose
one-stage Full for lower-resolution direct generation, distilled two-stage for
the default fast high-resolution path, or Full two-stage for the official
guided high-resolution path and LoRA450 refinement.

vLLM-Omni loads the official split safetensors repository directly. Pipeline
class selects the topology; `--task-type` selects the trained weight profile.
They are independent axes, with the following public contract:

| Startup selector | Topology | Weight profile | Official execution path |
|---|---|---|---|
| No LTX selector | Two-stage | Distilled | Default; positive-only 8-step generation plus 3-step refinement |
| `--task-type full` | Two-stage | Full/Dev | Guided Full stage followed by the official LoRA450 refinement |
| `--model-class-name LTX2Pipeline` | One-stage | Full/Dev | Guided Full one-stage generation |
| `--model-class-name LTX2Pipeline --task-type full` | One-stage | Full/Dev | Explicit form of the preceding row |
| `--model-class-name LTX2Pipeline --task-type distilled` | One-stage | Distilled | Rejected; the official release does not define this combination |

`LTX2Pipeline` means one-stage and `LTX2TwoStagePipeline` means two-stage.
Neither class name encodes whether the checkpoint is Full or distilled. For an
explicit default-equivalent command, use `LTX2TwoStagePipeline` together with
`--task-type distilled`.

Both topologies support T2V when no image is supplied and first-frame I2V when
one image is supplied. The two-stage pipeline generates the first phase at
half spatial resolution, upsamples the video latent by 2x, applies the official
FP32 re-noise, and runs a three-step full-resolution refinement phase. Audio
remains in its model-defined latent structure while video is spatially
upsampled.

## References

- [Official LTX-2.5 checkpoint and license](https://huggingface.co/Lightricks/LTX-2.5)
- [Official LTX-2 implementation](https://github.com/Lightricks/LTX-2)
- [Text-to-video offline example](../../examples/offline_inference/text_to_video/text_to_video.py)
- [Image-to-video offline example](../../examples/offline_inference/image_to_video/image_to_video.py)

## Hardware Support

## GPU

### 1x NVIDIA B300 275040 MiB

#### Environment

- OS: Linux 6.8.0-90-generic
- Python: 3.12.3
- NVIDIA driver: 610.43.02
- vLLM: 0.27.0
- vLLM-Omni: 0.27.0.dev106 (PR #6070 validation branch)

#### Command

The following Full one-stage command is the B300 parity configuration. Replace
the prompt as needed; keep seed and dimensions fixed when comparing outputs:

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2.5 \
  --model-class-name LTX2Pipeline \
  --prompt "A cinematic tracking shot follows a red fox through a snowy forest at dawn, with synchronized footsteps, wind, and distant birdsong." \
  --height 544 \
  --width 960 \
  --num-frames 481 \
  --num-inference-steps 30 \
  --frame-rate 24 \
  --fps 24 \
  --seed 42 \
  --enforce-eager \
  --output ltx25-full-one-stage.mp4
```

#### Verification

```bash
ffprobe -v error \
  -show_entries stream=codec_type,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 ltx25-full-one-stage.mp4
```

The command must finish without an exception and report both video and audio
streams. The video stream should be 960x544 at 24 fps.

#### Notes

- This is the reference qualification platform for the commands below.
- `--enforce-eager` is used for official numerical-parity runs.
- Peak-memory and acceleration claims are intentionally omitted until their
  raw-checkpoint qualification matrices are complete.

## Installing vLLM-Omni

The repository is gated. Accept its Hugging Face license and authenticate
before downloading it:

```bash
hf auth login
export MODEL=Lightricks/LTX-2.5
```

Install current vLLM-Omni with its matching vLLM version. LTX-2.5 requires a
Transformers version that provides `Gemma4UnifiedForConditionalGeneration`:

```bash
uv venv --python 3.12
source .venv/bin/activate
export VLLM_VERSION=0.27.0
uv pip install "vllm==${VLLM_VERSION}" --torch-backend=auto
uv pip install -e .
uv pip install -U "transformers>=5.10.1,<5.15"
```

`ffmpeg` and `ffprobe` must be on `PATH` for MP4 output. I2V additionally
requires PyAV backed by an FFmpeg build with `libx264`; the LTX-2.5
conditioning path applies the official H.264 CRF-18 round trip by default.

The loader selects exact official artifact filenames from `--task-type` rather
than guessing from a wildcard. Two-stage execution also resolves the x2
spatial latent upsampler. Full two-stage additionally resolves the official
Distilled LoRA450 for its second phase.

## Text-to-video and image-to-video generation

The generic offline examples implement the contract above. Dimensions passed
to a two-stage pipeline are final output dimensions; its first phase derives
the half-resolution shape internally.

### Default distilled two-stage T2V

No LTX-specific startup selector is required:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2.5 \
  --prompt "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot." \
  --height 1088 \
  --width 1920 \
  --num-frames 121 \
  --num-inference-steps 8 \
  --frame-rate 24 \
  --fps 24 \
  --enforce-eager \
  --output ltx25-distilled-two-stage.mp4
```

The explicit equivalent adds:

```bash
--model-class-name LTX2TwoStagePipeline --task-type distilled
```

### Full two-stage T2V

Select Full/Dev weights while retaining the default two-stage topology:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2.5 \
  --task-type full \
  --prompt "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot." \
  --height 1088 \
  --width 1920 \
  --num-frames 121 \
  --num-inference-steps 30 \
  --frame-rate 24 \
  --fps 24 \
  --enforce-eager \
  --output ltx25-full-two-stage.mp4
```

Full two-stage requires the official LoRA450 only for its refinement phase.

### Full one-stage T2V

Select the one-stage topology. Full is its default and only supported LTX-2.5
weight profile:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2.5 \
  --model-class-name LTX2Pipeline \
  --prompt "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot." \
  --height 544 \
  --width 960 \
  --num-frames 121 \
  --num-inference-steps 30 \
  --frame-rate 24 \
  --fps 24 \
  --enforce-eager \
  --output ltx25-full-one-stage.mp4
```

Adding `--task-type full` is equivalent but optional.

### First-frame I2V

Use the image-to-video example with the same topology and task selectors. This
example uses the default distilled two-stage path:

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Lightricks/LTX-2.5 \
  --image /absolute/path/to/first-frame.png \
  --prompt "The red fox walks forward while the camera tracks alongside." \
  --height 1088 \
  --width 1920 \
  --num-frames 121 \
  --num-inference-steps 8 \
  --frame-rate 24 \
  --fps 24 \
  --enforce-eager \
  --output ltx25-distilled-two-stage-i2v.mp4
```

Add `--task-type full` for Full two-stage I2V, or add
`--model-class-name LTX2Pipeline` for Full one-stage I2V. CRF 18 is the
default. Use `--extra-body '{"image_crf":0}'` only when an application
explicitly needs to bypass the conditioning round trip.

## Online serving

### Default distilled two-stage

```bash
vllm serve Lightricks/LTX-2.5 \
  --omni \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager
```

### Full two-stage

```bash
vllm serve Lightricks/LTX-2.5 \
  --omni \
  --task-type full \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager
```

### Full one-stage

```bash
vllm serve Lightricks/LTX-2.5 \
  --omni \
  --model-class-name LTX2Pipeline \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager
```

Check readiness before submitting a request:

```bash
curl -sS -o /dev/null -w 'health HTTP %{http_code}\n' \
  http://127.0.0.1:8000/health
```

A default distilled two-stage T2V request is:

```bash
curl -sS --fail-with-body \
  -X POST http://127.0.0.1:8000/v1/videos/sync \
  -F 'prompt=A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot.' \
  -F 'width=1920' \
  -F 'height=1088' \
  -F 'num_frames=121' \
  -F 'fps=24' \
  -F 'num_inference_steps=8' \
  -F 'seed=42' \
  -o ltx25-online.mp4
```

For I2V, add exactly one image to the same request:

```bash
-F 'input_reference=@/absolute/path/to/first-frame.png;type=image/png'
```

Distilled execution is positive-only and rejects negative prompts rather than
silently ignoring them. Full execution uses the official guided path.

## Key parameters and constraints

| Parameter | Full one-stage | Distilled two-stage | Full two-stage |
|---|---:|---:|---:|
| Final width x height | 960x544 | 1920x1088 | 1920x1088 |
| Frames / frame rate | 121 / 24 | 121 / 24 | 121 / 24 |
| Stage 1 denoise steps | 30 | 8 | 30 |
| Stage 2 refinement steps | N/A | 3 | 3 |
| Guidance | Full guided | Positive-only | Full guided, then LoRA450 positive-only refinement |

- `num_frames` must be `8k+1`.
- One-stage dimensions must be divisible by 32. Two-stage final dimensions
  must be divisible by 64.
- Set `num_frames` explicitly for online requests because the generic video API
  default is one frame.
- One-stage accepts a request-level `sigmas` override. Two-stage accepts
  `stage_1_sigmas` and `stage_2_sigmas` independently; omitted phases retain
  their official schedule.
- T2V and I2V share weights. I2V accepts one RGB first frame; multi-frame
  conditioning is not documented as release-qualified here.

## Feature qualification

This page documents the official checkpoint and pipeline-selection contract.
It does not transfer speed, peak-memory, numerical-parity, or acceleration
claims from prior Diffusers-layout experiments to the raw checkpoint path.
Consult [Diffusion Features](../../docs/user_guide/diffusion_features.md) for
the current conservative feature matrix; entries marked unverified are not
recommended until their raw-checkpoint validation is recorded.
