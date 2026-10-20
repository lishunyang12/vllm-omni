# FastH3 Omni — Ref2VA

[FastH3 family](README.md) · [FastH3](FastH3.md) · [FastH3 V2](FastH3-V2.md)

## Summary

- Model: `FastVideo/FastVideo-FastH3-Omni-8-Step-V1`.
- Task: Ref2VA, reference-conditioned video and audio, eight PDD forwards.
- Platforms: SM100/SM103 and SM120/SM121 through their respective sparse providers.
- Mode: existing `/v1/videos` API, BF16 DiT and native VAEs.
- Status: experimental; full-model validation on the rebased source is pending.

## Hardware and backend selection

Omni keeps the same logical 128-token VSA contract on both backend families:

| GPU family | BF16 execution | Dependency / validation |
| --- | --- | --- |
| SM100/SM103 | FastVideo's native tile128 kernel | Requires a build exposing `block_sparse_attn_sm100a` with tile128 support; full-model validation on the current source is pending |
| SM120/SM121 | FlashInfer block64 with logical tile128 routing | SM120 kernel regressions and historical offline runs; SM121 has not been tested here |
| Other GPUs | No compatible Omni tile128 provider enabled in this integration | Requires an additional sparse backend implementation |

Use `FASTVIDEO_VSA` with default `provider=auto` and `precision=bf16`.
GPU architecture changes the kernel implementation, not the checkpoint,
sampling schedule, routing policy, or HTTP request. The SM120 results below
describe one tested deployment.

## Model access and installation

[FastH3 Omni 8-Step V1](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1)
is a reference-conditioned PDD student. This recipe is prepared for its
public release. If the checkpoint or its native base components require
access, authenticate with `hf auth login` or set `HF_TOKEN`. Use a model ID
or local snapshot that your account can access.
The loader reads `transformer_ref/` directly and fetches the native text encoder,
tokenizer, processor, and VAEs from the MiniMax-H3 revision pinned in
`fastvideo_inference.json`. Neither base transformer is downloaded.

Follow the [CUDA source installation guide](../../docs/getting_started/installation/gpu.md)
for a matching vLLM/vLLM-Omni environment. The current checkout requires
vLLM 0.32.0; `ffmpeg` and `ffprobe` must be on `PATH`.

### SM100/SM103 native tile128

Install the `vsa` extra and a compatible FastVideo kernel build:

```bash
uv pip install -e '.[vsa]'
python -c 'from fastvideo_kernel import block_sparse_attn_sm100a'
```

The import checks API availability. The installed extension must also contain
native code for the actual GPU and support BF16, head dimension 128, and
128-token blocks. The wrapper checks support at execution time. A tile64-only
build is insufficient for this checkpoint.

### SM120/SM121 logical tile128 through FlashInfer

On this GPU family, the sparse attention provider is selected
automatically and uses FlashInfer's `bsa_attn_sm120_blk64_fwd` API. Each logical
128-token query/key tile is split into 64-token compute blocks, preserving the
selected keys and one softmax reduction over all selected keys. Pooling,
top-k selection, reference-region budgets and compression remain tile-128.

Install a FlashInfer build exposing that API; older packages may not
contain it. The tested upstream FlashInfer revision is
`f75bed5da7112103b75f82a4d4b5faf4be335154`, with PyTorch `2.13.0+cu130`,
Triton `3.7.1` and CUTLASS DSL `4.6.2`. Check the provider before loading weights:

```bash
python -c 'from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_fwd'
```

The SM120 integration uses this existing BF16 FlashInfer kernel automatically;
it does not require a custom `fastvideo-kernel[sm120]` build. The former
experimental `FASTVIDEO_VSA_SM120` and `FASTVIDEO_VSA_SM120_SPLITS` variables
no longer select a provider. The shared in-repository helpers provide VSA
layout and pooling. Partial KV blocks retain their valid token counts;
empty KV halves are excluded. Transport-only padding is removed before the
kernel and restored as zero rows afterward. Missing kernels, device faults and
incompatible hardware raise an error; no dense attention fallback is used.
Keep the default `provider=auto` and `precision=bf16` for this tile-128
checkpoint. Explicit tile-64 provider overrides and Sage precision are
rejected; native H3's tile-64 provider selection remains available.

## Serve

The same model command selects the sparse provider from the actual GPU.
This eight-GPU configuration uses HSDP8 and Ulysses8; its historical
measurement is on SM120. Adjust parallel degrees to the target GPU capacity
and interconnect, and qualify the resulting deployment.

```bash
export FASTH3_OMNI_MODEL=FastVideo/FastVideo-FastH3-Omni-8-Step-V1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve "${FASTH3_OMNI_MODEL}" --omni \
  --revision 88d512c5e7cc38dda499fbb65688ce58611f9c80 \
  --host 127.0.0.1 --port 8096 --trust-remote-code --task-type ref2va \
  --served-model-name FastH3-Omni \
  --num-gpus 8 --tensor-parallel-size 1 --usp 8 --ring 1 --enforce-eager \
  --use-hsdp --hsdp-shard-size 8 --hsdp-replicate-size 1 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend FASTVIDEO_VSA
```

## Generate video and audio

Set `REF_IMAGE` to a local PNG and write the description of the desired scene,
actions, camera movement, and audio to `prompt.txt`.

```bash
export REF_IMAGE=/path/to/reference.png
curl --fail-with-body -sS http://127.0.0.1:8096/v1/videos/sync \
  -F 'model=FastH3-Omni' -F 'prompt=<prompt.txt' \
  -F "input_reference=@${REF_IMAGE};type=image/png" \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' -F 'fps=24' \
  -F 'num_inference_steps=8' -F 'guidance_scale=1' -F 'flow_shift=12' \
  -F 'seed=43' \
  -F 'extra_params={"task":"ref2va","duration":5.2,"audio_flow_shift":3.0}' \
  -o fasth3_omni.mp4
```

These eight-GPU parallel settings were validated on SM120 through offline generation;
the HTTP endpoint was not part of that check. HSDP shards the BF16 weights
without quantization. For four GPUs, use TP2/Ulysses2, text encoder TP4 and
VAE patch parallelism 4, and omit the HSDP options.

Use H3's existing image/video/audio reference fields; at least one image or
video is required. The API preserves order within each modality and packs
images, videos (with their soundtracks), then standalone audio. When comparing
with FastVideo, supply references in this same order. Arbitrary cross-modality
CLI ordering from the upstream example is not an additional Omni API feature.

## Sampling and reference contract

The checkpoint pins eight forwards over Grid32 blocks `[0,4,8,12,16,20,24,28,32]`,
video/audio shifts **12/3**, and guidance **1**. Each forward fuses four output
heads using modality-specific integration weights. The rational shift fixes
the training endpoint **0.999**, and initial target noise is scaled by that
endpoint. Reference rows remain fixed during denoising. See [FastH3 V2](FastH3-V2.md) for its separate T2VA sampling rule.

Attention uses `(4,4,8)` video tiles, **90% sparsity**, and the trained
`p2_multi_region` reference policy. Every reference video has its own 10% tile
budget; images, text, and audio remain dense. Local attention, pure Ulysses,
and TP2 with Ulysses2 are supported. Fixed top-k overrides, custom
timesteps/sigmas, additional LoRA adapters, and partial-schedule latent
refinement are rejected.

## Validation and performance

CPU regression tests cover head integration, schedules, component resolution,
and reference-region routing. Synthetic SM120 kernel tests compare outputs and
softmax normalization against FP32 references, including partial/empty blocks,
14/56 heads and sequences up to 41,472 tokens (sampled query rows for long
sequences). CUDA Graph replay with changed queries and masks is also checked.
The repository's SM120 GPU regression also tests transport padding, empty
routes, and CUDA Graph replay with changed Q/K/V, masks and valid lengths.
These kernel tests do not compare against the native SM100/SM103 kernel.

The following full-model measurements were recorded at fork commit
`f2636995c6690183b18a97d2a917599a2c767c2e`, before rebasing onto upstream
`18c3bd5d2d4323bc7adca1ac3c465c4a53bbfb6e`. They are historical validation;
full-model generation and serving must be repeated with the rebased source
and its matching vLLM version before marking the integration ready.

The measured SM120 host reported `NVIDIA Graphics Device`, 73,415 MiB per GPU,
with Ethernet/RoCE links. On four of these GPUs, the pinned checkpoint
completed 1344x768 Ref2VA generation with 124 and 345 frames, including 32 kHz
stereo audio. For 124 frames, TP2/Ulysses2 with FlashInfer took a median
**70.20 s** across three warm requests; HSDP4/Ulysses4 took **76.56 s**.
Initialization, JIT warmup and activation-capture cases are excluded. A local
experimental block128 kernel reached **69.98 s** with TP2/Ulysses2, only about
0.3% faster end to end, so the SM120 integration uses the existing provider.
These are measurements for one reference image, prompt and seed, not a general
quality or speed guarantee. TP2/Ulysses2 uses about 64 GiB at 124 frames and
approaches device capacity at 345 frames; HSDP4/Ulysses4 uses less memory.

On eight of the same SM120 GPUs, TP1/Ulysses8 with HSDP8, text encoder TP8 and
VAE patch parallelism 8 took **41.04 s** for 124 frames (median of three warm
requests: 41.040, 41.056 and 41.028 s). Reported peak memory was approximately
35.2 GiB per GPU. A 345-frame request completed in **95.93 s** with approximately
38.9 GiB reported peak memory per GPU; this longer shape was measured once.
All five requests passed full audio/video decoding and frame-count checks.
This is a comparison between four- and eight-GPU deployments,
not a kernel-only improvement. All runs retain BF16 DiT weights, eight model
forwards, tile-128 routing and the full native VAE. Different parallel degrees
can change floating-point rounding, so this is not a bitwise equivalence claim.

BF16 provider outputs are not bitwise interchangeable: same-seed video SSIM
between the experimental block128 and FlashInfer paths was 0.740 at 124 frames.
Both providers also exceeded the original strict per-element FP32 error gate
on some captured layers. Preserve these limitations when interpreting the
synthetic regression results. Full output quality still needs validation
against the upstream SM100/SM103 implementation before production use.

Inspect both streams and decode the entire output:

```bash
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,sample_rate,channels \
  -of json fasth3_omni.mp4
ffmpeg -v error -i fasth3_omni.mp4 -f null -
```

The SM100/SM103 native tile128 path is a separate provider; this recipe does
not claim cross-provider numerical equivalence.
