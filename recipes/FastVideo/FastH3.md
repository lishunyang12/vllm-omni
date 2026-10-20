# FastH3 — four-step Preview V1

[FastH3 family](README.md) · [FastH3 V2](FastH3-V2.md) · [FastH3 Omni](FastH3-Omni.md)

## Summary

- Task: T2VA, text to synchronized video and audio.
- Release: a load-time adapter for `MiniMaxAI/MiniMax-H3`.
- Platform: CUDA with a compatible dense or tile64 VSA backend.
- Mode: existing `/v1/videos` API.
- Scope: the Dense / Data-Free variant has historical measurements below;
  the other adapter variants are described without additional performance claims.

## Hardware and backend selection

The adapter is not tied to a GPU architecture. Select a backend for the
adapter variant and the installed hardware:

| Variant | Backend | Hardware requirements |
| --- | --- | --- |
| Dense | `FLASH_ATTN`, or another H3-compatible dense backend | Follow that backend's GPU and build requirements |
| Dense, historical B300 profile | `TRTLLM_ATTN` | The measurements below used 8× B300 with NVLink |
| VSA | `FASTVIDEO_VSA`, BF16 | FastVideo tile64 on compatible CUDA GPUs; automatic FlashInfer selection on SM120/SM121 when available |

The eight-GPU command below is a deployment example. GPU memory, interconnect,
and output shape determine suitable parallel settings. The B300 measurements
do not qualify other devices.

## Model and installation

[FastH3](https://haoailab.com/blogs/fasth3-preview/) is FastVideo's four-step
DMD2 student of H3-Base. It reuses H3's text encoder, VAEs, tokenizers, and
schedulers unchanged, replacing 49 denoiser evaluations with four. It is fused
into the checkpoint at load time rather than switched per request, because it
carries full-rank deltas that no LoRA layer can express.

Install the matching vLLM/vLLM-Omni source environment using the
[CUDA installation guide](../../docs/getting_started/installation/gpu.md).
`ffmpeg` and `ffprobe` are required for video output. Authenticate with
`hf auth login` when the base model or adapter requires access.

The bundle publishes four variants, so download one and point `--lora-path` at
it; the repository root is refused rather than guessed at:

```bash
export FASTH3_DIR=/path/to/fasth3
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors --local-dir "${FASTH3_DIR}"
export FASTH3_LORA="${FASTH3_DIR}/dense-datafree/adapter_model.safetensors"
```

## Serve the Dense / Data-Free adapter

```bash
export FASTH3_DENSE_BACKEND=FLASH_ATTN

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve MiniMaxAI/MiniMax-H3 --omni \
  --host 127.0.0.1 --port 8094 --trust-remote-code \
  --task-type fl2va --served-model-name FastH3 \
  --lora-path "${FASTH3_LORA}" \
  --num-gpus 8 --tensor-parallel-size 1 --usp 8 --ring 1 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-config.default.backend "${FASTH3_DENSE_BACKEND}"
```

T2VA uses H3's FL2VA partition, so `--task-type fl2va` is intentional.
The adapter is fused at load time; `--lora-backend` and per-request `lora=`
selection do not apply. For the historical B300 profile, set
`FASTH3_DENSE_BACKEND=TRTLLM_ATTN` before launching.

## Generate video and audio

```bash
curl --fail-with-body -sS http://127.0.0.1:8094/v1/videos/sync \
  -F 'model=FastH3' \
  -F 'prompt=A red fox runs through fresh snow at dawn, with pawsteps, winter wind and distant birds.' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' -F 'fps=24' \
  -F 'num_inference_steps=4' -F 'guidance_scale=1' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":4.4}' \
  -o fasth3.mp4

ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,sample_rate,channels \
  -of json fasth3.mp4
```

Requests must ask for `num_inference_steps=4` and `task=t2va`: the release's five
sigma points bound four denoiser evaluations, and that count is what the step
scheduler admits a request on. The server denoises on the release's own ladder,
keeping H3's per-modality shifts at the checkpoint values, so a request that
overrides `flow_shift` or `audio_flow_shift` is rejected - it would sample the
student at noise levels it was never distilled at.

Only a release that identifies itself as FastH3 is fused; any other
`fastvideo-lora-v2` adapter stays on the dynamic LoRA route. A claimed artifact
is then held to its own metadata: one that misdeclares its tensor counts or
leaves a transformer block unedited is refused at startup instead of serving
mostly base H3 weights on a four-step schedule. DiT-wide offload is refused for the same
reason - `--enable-cpu-offload`, `--enable-layerwise-offload` and
`--enable-distributed-layerwise-offload` all bypass the fusion, so they fail fast.

## VSA adapter variants

For a VSA adapter variant, download that variant's adapter file, install the
`vllm-omni[vsa]` extra, and replace the dense attention backend in the serve
command with these flags. The provider is selected for the actual GPU;
SM90 and SM100/SM103 use FastVideo's tile64 route, while compatible SM120/SM121
installations use FlashInfer BF16. The video request API is unchanged:

```bash
--diffusion-attention-config.default.backend FASTVIDEO_VSA \
--diffusion-attention-config.default.fastvideo_vsa_topk 64
```

See
[provider selection](../../docs/user_guide/diffusion/attention_backends/fastvideo_vsa.md#automatic-provider-selection)
for hardware requirements and provider overrides. This recipe's
historical B300 measurements below use the dense adapter.

FastH3 VSA applies its learned `.set_weight` compression gates to the complete
packed `[text | cond | audio | video]` document using the official H3 geometry:
text/condition/audio prefix tiles never cross segment boundaries, and target
video rows use `(4, 4, 4)` 3-D tiles (64 tokens). Prefix queries remain dense;
video queries select all prefix tiles plus the configured top-k video tiles.
Pure Ulysses sequence parallelism is supported: the learned gate follows the
same sequence-to-head all-to-all as Q/K/V before VSA runs. Ring and all-gather
SP remain unsupported because they do not present a complete packed sequence
to each block-sparse kernel rank.

The Dense / Data-Free variant does not require `fastvideo-kernel` and should be
served with a dense attention backend.

## Historical validation

The following measurements were migrated from the native H3 recipe. The
original entry did not record a source SHA or the B300 memory configuration;
these results do not qualify other software or hardware configurations.

Measured on 8x NVIDIA B300 with USP8, VAE patch-parallel 8, `TRTLLM_ATTN`, at
1344x768, 4.4 s, seed 1101, one warmup excluded and two runs recorded:

| Adapter | Steps | End-to-end | Diffusion engine |
| --- | ---: | ---: | ---: |
| none (base H3) | 50 | 25.8 / 26.4 s | 16.22 / 16.28 s |
| FastH3 Dense | 4 (5 sigma points) | 11.7 / 11.8 s | 2.37 / 2.36 s |

The denoising speedup is 6.9x. End-to-end is 2.2x because text encoding, VAE
decoding and muxing are a fixed cost that dominates a clip this short; longer
generations move the end-to-end figure toward the denoising one. Fusing the
adapter does not measurably change startup: weight loading took 77.3 s with it
against 85.8 s without.
