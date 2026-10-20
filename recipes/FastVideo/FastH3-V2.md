# FastH3 V2 — T2VA

[FastH3 family](README.md) · [FastH3](FastH3.md) · [FastH3 Omni](FastH3-Omni.md)

## Summary

- Model: [`FastVideo/FastVideo-FastH3-8-Step-V2`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2).
- Task: T2VA, text to synchronized video and audio, eight transformer forwards.
- Platform: CUDA with a compatible 64-token VSA kernel.
- Mode: existing `/v1/videos` API, BF16 DiT and native VAEs.
- Status: experimental. Loader and attention regressions are covered;
  full-model serving, output quality, latency, and memory qualification on the
  current source remain pending.

## Hardware and backend selection

V2's 64-token attention policy is independent of the GPU architecture.
`FASTVIDEO_VSA` selects a provider for the actual device:

| GPU family | BF16 provider | Dependency / validation |
| --- | --- | --- |
| SM90, SM100/SM103, and other CUDA devices supported by FastVideo tile64 | FastVideo | Install the `vsa` extra and a compatible kernel build; V2 full-model qualification is pending |
| SM120/SM121 | FlashInfer when its tile64 API is installed | Real SM120 kernel regressions pass; SM121 and V2 full-model runs remain unqualified |

On SM100/SM103, the tile64 route follows the installed FastVideo provider's
dispatch; the native tile128 requirement belongs to Omni. No model or request
change is required when selecting another compatible provider.

## Model and installation

V2 is a full distilled checkpoint. It does not need the four-step FastH3
adapter. Its published inference contract uses 64-token VSA with **80%
sparsity**, video/audio shifts **10/3**, and guidance **1**. Only T2VA was
distilled; FL2VA and Ref2VA requests are not supported by this recipe.

Follow the [CUDA source installation guide](../../docs/getting_started/installation/gpu.md)
for the repository-supported vLLM/vLLM-Omni environment.
Install `ffmpeg` and `ffprobe` for video output.

For the FastVideo provider, install the optional kernel dependency:

```bash
uv pip install -e '.[vsa]'
```

For SM120/SM121, install a FlashInfer build exposing the BF16 tile64 API:

```bash
python -c 'from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_fwd'
```

The kernel regression environment uses FlashInfer `0.7.0.post1`, vLLM `0.32.0`,
PyTorch `2.13.0+cu130`, Triton `3.7.1`, and CUTLASS DSL `4.8.0`. This is an attention-kernel test
environment, not a qualified V2 serving environment. See
[VSA provider selection](../../docs/user_guide/diffusion/attention_backends/fastvideo_vsa.md#automatic-provider-selection)
for backend requirements.

The loader reads the student, text encoder, tokenizer, and processor from V2.
For tiled/parallel decoding, it uses H3's native video and audio VAEs. The
public V2 snapshot does not contain `provenance.json`, so the Omni adapter pins
these native components to MiniMax-H3 revision
`9bfb6693f2cf6de171db46d1aa586f67d773a1da`. This is an adapter compatibility pin,
not a training-provenance claim. An existing `provenance.json` must provide a
valid explicit base revision and takes precedence. Neither base transformer
is downloaded. Authenticate with `hf auth login` if the native base components
require access.

## Serve

The following configuration uses eight GPUs with HSDP8 and Ulysses8 and lets
the attention backend select the hardware provider. It is a starting
configuration for V2 qualification, not a measured V2 result. Adjust parallel
degrees for the target memory capacity and interconnect.

```bash
export FASTH3_V2_MODEL=FastVideo/FastVideo-FastH3-8-Step-V2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve "${FASTH3_V2_MODEL}" --omni \
  --revision 3da2ddfe1954d9cda4c05b643dc0f26007a655c5 \
  --host 127.0.0.1 --port 8095 --trust-remote-code --task-type fl2va \
  --served-model-name FastH3-V2 \
  --num-gpus 8 --tensor-parallel-size 1 --usp 8 --ring 1 --enforce-eager \
  --use-hsdp --hsdp-shard-size 8 --hsdp-replicate-size 1 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-config.default.backend FASTVIDEO_VSA
```

`--task-type fl2va` selects the H3 partition that implements T2VA. The request
task remains `t2va`; no reference image, video, or audio is required.
Keep the default BF16 attention precision for this recipe.

## Generate video and audio

```bash
curl --fail-with-body -sS http://127.0.0.1:8095/v1/videos/sync \
  -F 'model=FastH3-V2' \
  -F 'prompt=A red fox runs through fresh snow at dawn, with pawsteps, winter wind and distant birds.' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' -F 'fps=24' \
  -F 'num_inference_steps=8' -F 'guidance_scale=1' -F 'flow_shift=10' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5.2,"audio_flow_shift":3.0}' \
  -o fasth3_v2.mp4

ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,sample_rate,channels \
  -of json fasth3_v2.mp4
ffmpeg -v error -i fasth3_v2.mp4 -f null -
```

H3 normalizes the requested duration to its supported frame lattice. Check
the resulting frame count and audio/video durations when timing a workload.

## Sampling and attention contract

The release lists nine sigma nodes:
`[0.999, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125, 0]`.
In Omni, `num_inference_steps=8` counts the eight transformer evaluations
between those nodes. It is not `9`, despite the node count used by FastVideo's
example.

V2 selects `ceil(0.2 * number_of_video_tiles)` video blocks per query, with a
minimum of one, while keeping prefix keys and prefix-query rows dense.
Its learned compression gates are retained. Do not set
`--fastvideo-vsa-topk`: V2 derives its top-k from the trained sparsity ratio
as the video length changes.

Custom timesteps/sigmas, extra LoRA adapters, other step counts, and changed
guidance or flow shifts are rejected. Local attention and pure Ulysses are
supported; ring and all-gather sequence parallelism are rejected. A missing
packed layout or failed sparse kernel raises an error instead of silently
changing the student's attention to dense SDPA.

## Validation limits

The public snapshot's inference contract and component layout were inspected
at the revision above. Its video/audio latent normalization arrays match the
pinned native VAE configurations; this does not establish weight or output
equivalence.

CPU regressions cover the missing-provenance loader path, explicit provenance,
eight-forward schedule, request validation, ratio-derived top-k, and rejection
of dense fallback. Real SM120 BF16 tile64 kernel tests are separate from
full-model validation. No V2 end-to-end throughput or generation-time number
is claimed here. Native 50-step H3 and Omni Ref2VA timings are different
workloads and must not be reported as V2 results.
