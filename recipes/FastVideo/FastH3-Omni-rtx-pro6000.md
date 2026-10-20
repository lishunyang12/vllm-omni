# FastH3 Omni on RTX PRO 6000 Blackwell

Generate video with stereo audio from text and a reference image on
one RTX PRO 6000 **Blackwell** with 96 GB GPU memory.

| Setting | Value |
| --- | --- |
| Model | `FastVideo/FastVideo-FastH3-Omni-8-Step-V1` |
| Quantization | Online FP8 DiT and text encoder |
| Attention | cuDNN BF16 VSA |
| Placement | Resident DiT; text-encoder layer offload |
| Video VAE | Released NVFP4 `fast-8L` |
| Output | 832×480 or 1344×768; five-second requests; 24 fps and 32 kHz stereo audio |
| Configuration | [fasth3-omni-rtx-pro6000.yaml](fasth3-omni-rtx-pro6000.yaml) |

## Install and serve

Complete the [shared installation steps](<FastH3 RefOmni Single Card.md#install>),
including the CUDA environment, `ffmpeg`, and model access. Run from the
repository root:

```bash
uv pip install -e '.[cudnn-bsa]'

CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --omni \
  --revision 88d512c5e7cc38dda499fbb65688ce58611f9c80 \
  --config recipes/FastVideo/fasth3-omni-rtx-pro6000.yaml
```

Then follow [Generate video and audio](<FastH3 RefOmni Single Card.md#generate-video-and-audio>)
to submit your reference image and prompt.

## Memory and decoder settings

Keep `diffusion-offload-config` set to `mode: layer` with
`components: [text_encoder]`. This retains the DiT on the GPU while
offloading text-encoder layers. Keep `vae-use-tiling: true`.

Use `fast-8L` for faster decoding, or follow
[Choose the video decoder](<FastH3 RefOmni Single Card.md#choose-the-video-decoder>)
to select `quality-36L`. Restart the server after editing the YAML.
