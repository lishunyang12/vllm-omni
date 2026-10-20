# FastH3 Omni

[FastH3 family](README.md) · [FastH3](FastH3.md) · [FastH3 V2](FastH3-V2.md)

[FastH3 Omni 8-Step V1](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1)
generates video and audio from text and reference images or videos.
It uses eight PDD forwards and logical tile128 VSA with 90% sparsity.

Use this page for the shared eight-GPU deployment. For RTX 4090, RTX 5090,
DGX Spark, or RTX PRO 6000 Blackwell, follow the
[single-card guide](<FastH3 RefOmni Single Card.md>).

## Setup

Follow the [CUDA installation guide](../../docs/getting_started/installation/gpu.md)
and install `ffmpeg`. Authenticate with `hf auth login` if the model or native
MiniMax-H3 components require access. The loader fetches the release-pinned
encoder and VAEs without downloading the base transformers.

Install the shared profile's cuDNN sparse provider:

```bash
uv pip install -e '.[cudnn-bsa]'
```

The profile uses cuDNN BF16 with native tile128, including on SM120.
See [provider requirements](../../docs/user_guide/diffusion/attention_backends/fastvideo_vsa.md).

## Serve

From the repository root:

```bash
vllm serve FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --omni \
  --config recipes/FastVideo/fasth3-8step-8gpu.yaml
```

The [shared configuration](fasth3-8step-8gpu.yaml) uses eight GPUs with HSDP,
Ulysses, text-encoder TP, and tiled VAE decoding. It sets eight steps and
guidance 1 for requests. The checkpoint determines the task and flow shifts;
cuDNN selects the kernel for the GPU.

A local checkpoint directory can replace the model ID. The server uses the
standard port 8000; ordinary CLI options override the YAML when needed.

The automatic-provider and cuDNN profiles both passed HTTP generation checks
on 8× RTX PRO 5000 with vLLM 0.32.0 and PyTorch 2.13.0+cu130.
The cuDNN check used frontend 1.31.0 and produced 124 frames at 1344×768,
24 fps, with 32 kHz stereo audio. Full media decoding passed.

### Optional NVFP4 video decoder

[FastH3-NVFP4-VAE-Decoder](https://huggingface.co/FastVideo/FastH3-NVFP4-VAE-Decoder)
provides `fast-8L` and `quality-36L` decoders. To select one, add this section
to the deployment YAML:

```yaml
additional-config:
  video_vae_decoder:
    model: FastVideo/FastH3-NVFP4-VAE-Decoder
    revision: ddd17469cab7e490b1a04baaf26b247f0f0dcf0e
    subfolder: fast-8L
```

Use `quality-36L` to select the deeper decoder. A local
`decoder-nvfp4.safetensors` path can also be the value of `video_vae_decoder`.
This option requires FlashInfer NVFP4 support on SM100, SM103, SM120, or SM121 and
the native VAE's tile-parallel mode.

Both decoders matched the pinned FastVideo implementation pixel for pixel
on a fixed synthetic latent producing 1344×768, 124-frame output on one
SM120 GPU. Temporal callbacks also matched complete decoding after output
quantization. The `fast-8L` override also passed HTTP generation with the
BF16 Omni model on 4× RTX PRO 5000 using cuDNN, HSDP4, Ulysses4,
text-encoder TP4, and tiled VAE parallelism 4: HTTP 200, 1344×768,
124 frames at 24 fps, and 32 kHz stereo audio. Full media decoding passed.
The single request took 45.13 seconds including response encoding and
excluding service startup; this is a smoke check, not a repeated benchmark.
The `quality-36L` decoder has the decoder-level checks above.

For serialized `FastH3-Omni-FP8` and `FastH3-Omni-NVFP4-QAD` checkpoints,
use the [single-card quantized configuration](<FastH3 RefOmni Single Card.md#2-start-the-server>).
It loads the exported DiT and NVFP4 text encoder and supports both NVFP4
video decoders. The eight-GPU measurements here use the BF16 release.

## Generate

Put the scene description in `prompt.txt` and the reference image in
`reference.jpg`:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
  -F 'prompt=<prompt.txt' \
  -F 'input_reference=@reference.jpg' \
  -F 'size=1344x768' -F 'seconds=5' \
  -o output.mp4
```

The response contains video and stereo audio. H3 aligns the requested duration
to its frame lattice; five seconds produces 124 frames at 24 fps.
For multiple images, video, or audio references, use the existing
[H3 reference API](../MiniMaxAI/MiniMax-H3.md).

## Benchmark: 8× NVIDIA RTX PRO 5000

Historical offline baseline on 8× RTX PRO 5000 Blackwell 72GB (SM120),
BF16, 1344×768, eight steps, one request at a time:

| Frames | Video duration | Generation latency | Reported allocator peak |
| ---: | ---: | ---: | ---: |
| 124 | 5.167 s | **41.04 s** median of 3 runs | **35.15 GiB** |
| 345 | 14.375 s | **95.93 s**, 1 run | **38.94 GiB** |

Configuration: TP1/Ulysses8/HSDP8, text encoder TP8, tiled VAE parallelism 8,
eager execution, default NCCL, one reference image, seed 43. The 124-frame
measurements exclude one warmup; the 345-frame shape has no separate warmup.
Timing covers `Omni.generate` through encoded MP4 bytes, excluding model
loading and file writes. Memory is the reported CUDA allocator high-water
mark, not whole-device usage.

These [baseline results](https://github.com/lishunyang12/vllm-omni/pull/63)
use vLLM 0.30.0, PyTorch 2.13.0+cu130, FlashInfer
`f75bed5da7112103b75f82a4d4b5faf4be335154`, Triton 3.7.1, and CUTLASS DSL 4.6.2.
Model revision: `88d512c5e7cc38dda499fbb65688ce58611f9c80`; native components:
`9bfb6693f2cf6de171db46d1aa586f67d773a1da`. They are not vLLM 0.32 HTTP timings.
The BF16 HTTP checks above use the same model revision. Commands follow
`main` by default; add `--revision <commit>` to reproduce that version.

Cross-provider output equivalence remains unverified. The three measured
124-frame outputs were byte-identical; warmup audio differed, with the cause
still unresolved.
