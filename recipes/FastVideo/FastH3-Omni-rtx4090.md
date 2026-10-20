# FastH3 Omni on RTX 4090

Generate video with stereo audio from text and a reference image on
one RTX 4090 with 24 GB GPU memory.

| Setting | Value |
| --- | --- |
| Model | `FastVideo/FastVideo-FastH3-Omni-8-Step-V1` |
| Quantization | Online FP8 DiT and text encoder |
| Attention | Triton BF16 VSA |
| Placement | Layerwise offload with rotating buffers; staged video/audio VAEs |
| Video VAE | Native FP32 encoding and FP16 decoding |
| Output | 832×480 or 1344×768; five-second requests; 24 fps and 32 kHz stereo audio |
| Configuration | [fasth3-omni-rtx4090.yaml](fasth3-omni-rtx4090.yaml) |

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
  --config recipes/FastVideo/fasth3-omni-rtx4090.yaml
```

Then follow [Generate video and audio](<FastH3 RefOmni Single Card.md#generate-video-and-audio>)
to submit your reference image and prompt.

## Memory settings

Keep `enable-distributed-layerwise-offload: true`,
`dlo-no-use-allgather: true`, and `vae-use-tiling: true`. These settings
rotate layer weights through GPU buffers and stage the native VAEs during
encoding and decoding. Host RAM holds offloaded weights.

Use the native VAE on the 4090, which has no native NVFP4 execution.
Start at `832x480`; use `1344x768` for 768P output.
