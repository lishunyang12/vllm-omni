# FastH3 Omni on DGX Spark

Generate video with stereo audio from text and a reference image on
DGX Spark with its GB10 GPU and 128 GB unified memory.

| Setting | Value |
| --- | --- |
| Model | `FastVideo/FastVideo-FastH3-Omni-8-Step-V1` |
| Quantization | Online FP8 DiT and text encoder |
| Attention | FlashInfer BF16 VSA |
| Placement | Resident in unified memory |
| Video VAE | Released NVFP4 `fast-8L` |
| Output | 832×480 or 1344×768; five-second requests; 24 fps and 32 kHz stereo audio |
| Configuration | [fasth3-omni-spark.yaml](fasth3-omni-spark.yaml) |

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
  --config recipes/FastVideo/fasth3-omni-spark.yaml
```

Then follow [Generate video and audio](<FastH3 RefOmni Single Card.md#generate-video-and-audio>)
to submit your reference image and prompt.

## Memory and decoder settings

Use a native Linux `aarch64` environment. Keep the resident configuration
and `vae-use-tiling: true`. CPU and GPU share the 128 GB memory pool, so
CPU offload does not add memory capacity. Close other large workloads to
leave room for model loading and generation.

Use `fast-8L` for faster decoding, or follow
[Choose the video decoder](<FastH3 RefOmni Single Card.md#choose-the-video-decoder>)
to select `quality-36L`. Restart the server after editing the YAML.
