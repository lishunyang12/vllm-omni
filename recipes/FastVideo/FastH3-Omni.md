# FastH3 Omni — Ref2VA

[FastH3 family](README.md) · [FastH3](FastH3.md) · [FastH3 V2](FastH3-V2.md)

## Summary

- Model: `FastVideo/FastVideo-FastH3-Omni-8-Step-V1`.
- Task: Ref2VA, reference-conditioned video and audio, eight PDD forwards.
- Platforms: SM100/SM103 and SM120/SM121 through their respective sparse providers.
- Mode: existing `/v1/videos` API, BF16 DiT and native VAEs.
- Status: experimental. See the benchmark methodology and known limitations below.

## Hardware and backend selection

Omni keeps the same logical 128-token VSA contract on both backend families:

| GPU family | BF16 execution | Dependency / validation |
| --- | --- | --- |
| SM100/SM103 | FastVideo's native tile128 kernel | Requires `block_sparse_attn_sm100a` with tile128 support |
| SM120/SM121 | FlashInfer block64 with logical tile128 routing | Requires `bsa_attn_sm120_blk64_fwd`; measurements below use SM120 |
| Other GPUs | No compatible Omni tile128 provider enabled in this integration | Requires an additional sparse backend implementation |

Use `FASTVIDEO_VSA` with default `provider=auto` and `precision=bf16`.
GPU architecture changes the kernel implementation, not the checkpoint,
sampling schedule, routing policy, or HTTP request. The SM120 results below
describe one tested deployment.

## Model access and installation

[FastH3 Omni 8-Step V1](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1)
is a reference-conditioned PDD student. If the checkpoint or its native base components require
access, authenticate with `hf auth login` or set `HF_TOKEN`. Use a model ID
or local snapshot that your account can access.
The loader reads `transformer_ref/` directly and fetches the native text encoder,
tokenizer, processor, and VAEs from the MiniMax-H3 revision pinned in
`fastvideo_inference.json`. Neither base transformer is downloaded.

Follow the [CUDA source installation guide](../../docs/getting_started/installation/gpu.md)
for the repository-supported vLLM/vLLM-Omni environment.
`ffmpeg` and `ffprobe` must be on `PATH`.

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
contain it. BF16 kernel regressions pass with FlashInfer `0.7.0.post1`,
vLLM `0.32.0`, PyTorch `2.13.0+cu130`, Triton `3.7.1`, and CUTLASS DSL
`4.8.0`. Check the provider before loading weights:

```bash
python -c 'from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_fwd'
```

The SM120/SM121 path uses the shared FlashInfer BF16 backend.
The H3 adapter retains logical tile128 layout and routing. Partial KV blocks retain their valid token counts;
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
  --diffusion-attention-config.default.backend FASTVIDEO_VSA
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

## Benchmark: 8× NVIDIA RTX PRO 5000

Hardware: 8× [NVIDIA RTX PRO 5000 72GB Blackwell](https://www.nvidia.com/en-gb/products/workstations/professional-desktop-gpus/rtx-pro-5000/),
SM120, with Ethernet/RoCE links. The driver reports 73,415 MiB per device.
This is a single-request latency benchmark with batch size 1 and concurrency 1.

| Output resolution | Frames | Video duration at 24 fps | Measured requests | Request latency | Reported allocator peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1344×768 | 124 | 5.167 s | 3 | **41.04 s** (median) | **35.15 GiB** |
| 1344×768 | 345 | 14.375 s | 1 | **95.93 s** | **38.94 GiB** |

### Configuration and measurement

- Model: Omni 8-Step V1, revision `88d512c5e7cc38dda499fbb65688ce58611f9c80`;
  native components revision `9bfb6693f2cf6de171db46d1aa586f67d773a1da`.
- Precision and attention: BF16 DiT, native VAEs, eight PDD forwards,
  logical tile128 VSA with 90% sparsity, FlashInfer block64 execution.
- Parallelism: TP1, Ulysses8, HSDP8, text encoder TP8, tiled VAE parallelism 8,
  eager execution, default NCCL collectives.
- Input: one reference image, a saxophone-performance prompt, seed 43,
  guidance 1, video/audio shifts 12/3. Output includes 32 kHz stereo audio.
- Timing: wall time of `Omni.generate` through encoded MP4 bytes. Model
  initialization, weight downloads, file writes, and media validation are excluded.
- Sampling: one 124-frame warmup followed by three measured requests
  (41.0399, 41.0561, 41.0280 s). The 345-frame result is one request on the
  loaded model; no separate warmup at that shape was recorded.
- Memory: the framework's reported CUDA allocator high-water mark, converted
  from MiB to GiB. This is not whole-device NVML usage or a minimum VRAM requirement.

These are the development-baseline results published in
[fork PR #63](https://github.com/lishunyang12/vllm-omni/pull/63),
using vLLM 0.30.0, PyTorch 2.13.0+cu130, FlashInfer revision
`f75bed5da7112103b75f82a4d4b5faf4be335154`, Triton 3.7.1, and CUTLASS DSL 4.6.2.
They predate the vLLM 0.32 integration and do not qualify that runtime.
They contain no external
AdaLN weight-elision, overlap, RDMA-transport, or quantized-VAE optimization.

## Output verification and known limitations

```bash
ffprobe -v error -count_frames \
  -show_entries stream=codec_name,width,height,nb_read_frames,r_frame_rate,sample_rate,channels \
  -of json fasth3_omni.mp4
ffmpeg -v error -i fasth3_omni.mp4 -f null -
```

All five recorded outputs passed full audio/video decoding and frame-count
checks. The three measured 124-frame outputs were byte-identical; the warmup
had different decoded audio. That difference has not been diagnosed.
Cross-provider output equivalence and full-model quality against the native
SM100/SM103 path have not been established. Kernel regressions validate
sparse routing and attention independently of those model-quality checks.
