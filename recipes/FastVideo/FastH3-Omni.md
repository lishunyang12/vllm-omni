# FastH3 Omni

[FastH3 family](README.md) · [FastH3](FastH3.md) · [FastH3 V2](FastH3-V2.md)

[FastH3 Omni 8-Step V1](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1)
generates video and audio from text and reference images or videos.
It uses eight PDD forwards and logical tile128 VSA with 90% sparsity.

## Setup

Follow the [CUDA installation guide](../../docs/getting_started/installation/gpu.md)
and install `ffmpeg`. Authenticate with `hf auth login` if the model or native
MiniMax-H3 components require access. The loader fetches the release-pinned
encoder and VAEs without downloading the base transformers.

Sparse attention uses native FastVideo tile128 on SM100/SM103 and FlashInfer
BF16 block64 on SM120/SM121. See [provider requirements](../../docs/user_guide/diffusion/attention_backends/fastvideo_vsa.md).
The deployment configuration is independent of GPU architecture.

## Serve

From the repository root:

```bash
vllm serve FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --omni \
  --config recipes/FastVideo/fasth3-8step-8gpu.yaml
```

The [shared configuration](fasth3-8step-8gpu.yaml) uses eight GPUs with HSDP,
Ulysses, text-encoder TP, and tiled VAE decoding. It sets eight steps and
guidance 1 for requests. The checkpoint determines the task and flow shifts;
the attention provider follows the GPU.

A local checkpoint directory can replace the model ID. The server uses the
standard port 8000; ordinary CLI options override the YAML when needed.

This serving profile was validated on 8× RTX PRO 5000 with vLLM 0.32.0,
PyTorch 2.13.0+cu130, and FlashInfer 0.7.0.post1.

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

Cross-provider output equivalence remains unverified. The three measured
124-frame outputs were byte-identical; warmup audio differed, with the cause
still unresolved.
