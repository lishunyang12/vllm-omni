# FastH3 Omni on RTX 5090

Generate video with stereo audio from text and a reference image on
one RTX 5090 with 32 GB GPU memory.

| Setting | Value |
| --- | --- |
| Model | `FastVideo/FastVideo-FastH3-Omni-8-Step-V1` |
| Quantization | Online FP8 DiT and text encoder |
| Attention | FlashInfer BF16 VSA |
| Placement | Layerwise offload of DiT and text encoder |
| Video VAE | Released NVFP4 `fast-8L` |
| Output | 832×480 or 1344×768; five-second requests; 24 fps and 32 kHz stereo audio |
| Configuration | [fasth3-omni-rtx5090.yaml](fasth3-omni-rtx5090.yaml) |

## Install and serve

Complete the [shared installation steps](<FastH3 RefOmni Single Card.md#install>),
including the CUDA environment, `ffmpeg`, and model access. Run from the
repository root:

```bash
uv pip install -e '.[vsa]'

CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --omni \
  --revision 88d512c5e7cc38dda499fbb65688ce58611f9c80 \
  --config recipes/FastVideo/fasth3-omni-rtx5090.yaml
```

Then follow [Generate video and audio](<FastH3 RefOmni Single Card.md#generate-video-and-audio>)
to submit your reference image and prompt.

## Memory and decoder settings

Keep `diffusion-offload-config` set to `mode: layer` with
`components: [dit, text_encoder]`, and keep `vae-use-tiling: true`.
Offload uses host RAM to reduce GPU weight residency.

Use `fast-8L` for faster decoding, or follow
[Choose the video decoder](<FastH3 RefOmni Single Card.md#choose-the-video-decoder>)
to select `quality-36L`. Restart the server after editing the YAML.
