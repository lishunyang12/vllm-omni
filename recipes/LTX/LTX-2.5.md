# LTX-2.5

> Text-to-video and first-frame image-to-video generation with synchronized
> audio

## Summary

- Vendor: Lightricks
- Supported checkpoint:
  [`Lightricks/LTX-2.5-Diffusers`](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
- Tasks: Full/SFT and merged-distilled one-stage/two-stage T2V/I2V
- Modes: offline inference and OpenAI-compatible `/v1/videos` HTTP serving
- Maintainer: Community

LTX-2.5 generates video and synchronized 48 kHz stereo audio. vLLM-Omni
uses four explicit pipeline names so checkpoint type and topology are never
inferred from a request flag:

> **Checkpoint layout:** This integration directly supports only
> `Lightricks/LTX-2.5-Diffusers`. The raw `Lightricks/LTX-2.5` repository uses
> split artifacts and cannot be passed directly to `--model`. The regular
> two-stage path resolves its official upsampler and distilled LoRA sidecars
> from that gated raw repository, so authenticate for both repositories.

| Mode | `--model-class-name` | Output | Steps | Transformer |
|---|---|---:|---:|---|
| Full/SFT one-stage T2V/I2V | `LTX2Pipeline` | 960x544, 121 frames at 24 FPS | 30 | `transformer_full/` |
| Full/SFT two-stage T2V/I2V | `LTX2TwoStagePipeline` | 1920x1088, 121 frames at 24 FPS | 30 + 3 | `transformer_full/` + distilled LoRA |
| Distilled one-stage T2V/I2V | `LTX2DistilledOneStagePipeline` | 960x544, 121 frames at 24 FPS | 8 | `transformer/` |
| Distilled two-stage T2V/I2V | `LTX2DistilledTwoStagePipeline` | 1920x1088, 121 frames at 24 FPS | 8 + 3 | `transformer/` |

Both two-stage paths generate at 960x544, apply the x2 latent upsampler, and
run the official three-step refinement tail. `LTX2TwoStagePipeline` uses the
Full/SFT transformer with guided Stage 1 and applies the official distilled
LoRA only for Stage 2. `LTX2DistilledTwoStagePipeline` uses the positive-only
merged-distilled transformer in both stages. The one-stage variants omit
upsampling and refinement. The deprecated `LTX2DistilledPipeline` name remains
an alias for `LTX2DistilledTwoStagePipeline`.

## When to use this recipe

Choose the class directly:

- `LTX2Pipeline`: Full/SFT, one stage, simplest quality baseline.
- `LTX2TwoStagePipeline`: Full/SFT Stage 1 plus LoRA-refined Stage 2,
  highest-quality official topology.
- `LTX2DistilledOneStagePipeline`: fastest low-resolution path.
- `LTX2DistilledTwoStagePipeline`: fast 1080p path using the merged-distilled
  transformer.

No `--task-type` flag is required.

## References

- [LTX-2.5-Diffusers checkpoint](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
- [LTX-2.5 upstream source/license artifacts (not a directly loadable `--model` checkpoint)](https://huggingface.co/Lightricks/LTX-2.5)
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

### Full/SFT one-stage T2V (canonical)

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2.5-Diffusers \
  --model-class-name LTX2Pipeline \
  --prompt "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot." \
  --height 544 \
  --width 960 \
  --num-frames 121 \
  --num-inference-steps 30 \
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
  --num-inference-steps 30 \
  --frame-rate 24 \
  --fps 24 \
  --enforce-eager \
  --output ltx25-one-stage-i2v.mp4
```

CRF 18 is the LTX-2.5 default. Add `--extra-body '{"image_crf":0}'` only
when an application explicitly needs to bypass the conditioning round trip.

### Full/SFT two-stage T2V

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2.5-Diffusers \
  --model-class-name LTX2TwoStagePipeline \
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

This path loads `transformer_full/` for guided Stage 1, then applies the
official distilled LoRA only during the three-step Stage-2 refinement.

### Distilled one-stage T2V

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2.5-Diffusers \
  --model-class-name LTX2DistilledOneStagePipeline \
  --prompt "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot." \
  --height 544 \
  --width 960 \
  --num-frames 121 \
  --num-inference-steps 8 \
  --frame-rate 24 \
  --fps 24 \
  --enforce-eager \
  --output ltx25-distilled-one-stage.mp4
```

This path uses the official positive-only eight-step schedule without latent
upsampling or Stage-2 refinement.


### Distilled two-stage T2V

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2.5-Diffusers \
  --model-class-name LTX2DistilledTwoStagePipeline \
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

To override either distilled phase, pass `--extra-body` with
`stage_1_sigmas` and/or `stage_2_sigmas`; an omitted phase keeps its official
schedule. Stage 2 re-noise uses the first value of its effective schedule.

### Distilled two-stage first-frame I2V

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Lightricks/LTX-2.5-Diffusers \
  --model-class-name LTX2DistilledTwoStagePipeline \
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

This path uses the same CRF-18 first-frame conditioning and official 8+3
two-stage schedule as the qualified T2V path.

The same I2V command supports all four classes. Select the matching values:

| `--model-class-name` | Height | Width | `--num-inference-steps` |
|---|---:|---:|---:|
| `LTX2Pipeline` | 544 | 960 | 30 |
| `LTX2TwoStagePipeline` | 1088 | 1920 | 30 |
| `LTX2DistilledOneStagePipeline` | 544 | 960 | 8 |
| `LTX2DistilledTwoStagePipeline` | 1088 | 1920 | 8 |

Use the same absolute first-frame path, prompt, frame count, frame rate, seed,
and output options shown above.

## Online serving

The release-qualified B300 path uses cuDNN attention explicitly:

```bash
export MODEL=Lightricks/LTX-2.5-Diffusers
export PORT=8000
export PIPELINE=LTX2Pipeline

CUDA_VISIBLE_DEVICES=0 \
vllm serve "${MODEL}" \
  --omni \
  --model-class-name "${PIPELINE}" \
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
  -F 'num_inference_steps=30' \
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
  -F 'num_inference_steps=30' \
  -F 'seed=42' \
  -o ltx25-online-i2v.mp4
```

Request-level overrides use multipart JSON: `image_crf` for I2V, `sigmas`
for one-stage/Full, or independent `stage_1_sigmas` and `stage_2_sigmas`
for distilled two-stage.

Set `PIPELINE` to any class in the offline table and restart the server to
change checkpoint type or topology. The class selects the matching weights from
the same `Lightricks/LTX-2.5-Diffusers` checkpoint; no request flag is needed.

## B300 validation

The generic offline examples and OpenAI-compatible online server were validated
on one NVIDIA B300 with `CUDNN_ATTN`; the recorded Full one-stage and
distilled two-stage modes returned HTTP 200.
The following are single-process cold-run diagnostics, not warmed throughput
claims. Generation excludes MP4 encoding; end to end includes model loading.

The accuracy CI fixes both the official and vLLM-Omni runtimes to PyTorch SDPA.
It runs the official Lightricks implementation pinned to commit `7954dcb` with the raw official transformer/VAE artifacts and connector weights
from the same `Lightricks/LTX-2.5-Diffusers` checkpoint used by vLLM-Omni. Its
decoded-video SSIM/PSNR and audio relative-L2/cosine gates are output-level
checks, not per-tensor parity.

| Mode | Generation | End to end | Peak GPU memory |
|---|---:|---:|---:|
| Distilled two-stage T2V · 1920x1088 | 13.770 s | 47.264 s | 109,666 MiB |
| Full/SFT one-stage T2V · 960x544 | 40.917 s | 77.137 s | 79,678 MiB |

All decoded outputs included synchronized 48 kHz stereo audio. Extended
1920x1088, 481-frame official-prompt results are available in the
[LTX-2.5 B300 gallery](https://lishunyang12.github.io/vllm-omni-rankings/scripts/ltx25_official_b300_1080p20s/).

### Feature qualification

| Feature | Status on LTX-2.5 | Notes |
|---|---|---|
| `CUDNN_ATTN` | Release-qualified | Recommended B300 path; Full one-stage and distilled two-stage passed offline and online. |
| `TORCH_SDPA` | Functional baseline | Full one-stage and distilled two-stage passed; intended for debugging and portability. |
| Native DP2 | Unverified | The run initialized two diffusion workers inside one stage replica; every request reported `replica_id=0`, so independent replica scheduling was not demonstrated. |
| HSDP2 | Capacity fallback | Output matched eager; peak memory decreased by 21.9%, with lower performance. |
| Distributed layerwise offload DP2 | Capacity fallback | Output matched eager; primary-rank peak memory decreased by 43.3%, with lower performance. |
| Whole-model CPU offload | Capacity fallback | Output matched eager; peak memory decreased by 35.3%, with lower performance. |
| VAE slicing | Release-qualified | Output matched eager. |
| VAE tiling | Release-qualified | The tiled decode completed successfully and reduced peak memory; unlike slicing, tiling is not bit-exact with eager. |
| TP2 / Ulysses SP2 | Experimental | Strict Ulysses completed, but neither TP2 nor SP2 passed the fixed-seed quality gate; `advanced_uaa` is rejected. |
| Regional `torch.compile` | Experimental | Generation completed, but the first run was slower and did not pass the fixed-seed quality gate. |
| FP8 | Experimental | Generation completed with 23.4% lower peak memory, but did not pass the quality gate. |
| Cache-DiT | Experimental | One-stage only: threshold 0.12 preserved eager output but recorded zero cache steps; threshold 0.15 exercised caching but failed the fixed-seed quality gate. Two-stage requests fail fast until phase-aware cache refresh is implemented. |
| `TRTLLM_ATTN` | Unsupported | The current kernel rejects LTX-2.5's head dimension of 64. |
| Ring SP2 | Unsupported | The tested path did not complete successfully. |
| TeaCache | Unsupported | LTX-2.5 has no validated TeaCache residual extractor or coefficient profile. |
| FlashAttention-3 on H100 | Unverified | No H100 was available; the installed Hopper extension cannot be validated on B300. |

## Constraints and unsupported paths

- The checkpoint must contain `transformer/`; Full/SFT additionally requires
  `transformer_full/`.
- `num_frames` must be `8k+1`. One-stage dimensions must be divisible by 32;
  two-stage final dimensions must be divisible by 64.
- Full/SFT one-stage defaults to 30 steps and accepts a custom `sigmas` list.
  Full/SFT two-stage uses 30 guided steps plus three refinement steps.
  Distilled one-stage is fixed at 8 steps; distilled two-stage accepts independent `stage_1_sigmas` and `stage_2_sigmas` lists.
- Set `num_frames` explicitly for online requests because the generic video
  API default is one frame.
- First-frame I2V is implemented for all four pipeline classes. Current B300
  parity artifacts cover Full one-stage and distilled two-stage; multi-frame
  conditioning is not yet release-qualified.
- LTX sequence parallelism supports only strict Ulysses. `advanced_uaa` is
  rejected because its mask redistribution does not preserve LTX cross-modal
  key-padding semantics.
- Cache-DiT is enabled only for one-stage recipes. Two-stage recipes change
  phase and spatial resolution, so they reject Cache-DiT until cache state and
  refresh policy become phase-aware.
- `TRTLLM_ATTN` currently rejects the LTX-2.5 head dimension of 64. Use
  `CUDNN_ATTN` on B300.
- DLO, CPU offload, and HSDP are memory-capacity fallbacks, not latency
  accelerators for this model.
