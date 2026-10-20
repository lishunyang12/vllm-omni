# FastH3 V2

[FastH3 family](README.md) · [FastH3](FastH3.md) · [FastH3 Omni](FastH3-Omni.md)

[FastH3 8-Step V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2)
is a full T2VA checkpoint: text to video and audio, with eight transformer
forwards, tile64 VSA at 80% sparsity, and video/audio flow shifts 10/3.
It does not use the four-step FastH3 adapter.

## Setup

Follow the [CUDA installation guide](../../docs/getting_started/installation/gpu.md),
install `ffmpeg`, and satisfy the [VSA provider requirements](../../docs/user_guide/diffusion/attention_backends/fastvideo_vsa.md).
Install `uv pip install -e '.[cudnn-bsa]'` for the profile's cuDNN BF16
tile64 provider.
Authenticate with `hf auth login` if the native MiniMax-H3 VAEs require access.

## Serve

From the repository root:

```bash
vllm serve FastVideo/FastVideo-FastH3-8-Step-V2 --omni \
  --config recipes/FastVideo/fasth3-8step-8gpu.yaml
```

The [shared eight-GPU configuration](fasth3-8step-8gpu.yaml) sets parallelism,
eight sampling steps, and guidance 1. The checkpoint selects the T2VA partition,
schedule, and sparsity; cuDNN selects the kernel for the GPU.
A local checkpoint directory can replace the model ID.

## Generate

```bash
curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
  -F 'prompt=A red fox runs through snow at dawn, with pawsteps, wind and distant birds.' \
  -F 'size=1344x768' -F 'seconds=5' \
  -o output.mp4
```

No reference media is required. The output includes video and audio. The
release's nine sigma nodes represent eight transformer forwards; do not
change the step count, guidance, flow shifts, or ratio-derived top-k.

## Validation scope

V2 loader, sampling, and sparse-kernel regressions are covered; V2 full-model
serving and performance remain unverified. The shared deployment has been
exercised with Omni Ref2VA, which is a separate workload.

The command follows the checkpoint's `main` branch. Loader validation used
revision `3da2ddfe1954d9cda4c05b643dc0f26007a655c5`; add
`--revision <commit>` to select that version. This revision has no
`provenance.json`. The adapter uses native tiled/parallel VAEs from MiniMax-H3
revision `9bfb6693f2cf6de171db46d1aa586f67d773a1da`, honoring an explicit
provenance pin when supplied. This is a compatibility choice; matching latent
normalization metadata does not establish VAE weight or output equivalence.
