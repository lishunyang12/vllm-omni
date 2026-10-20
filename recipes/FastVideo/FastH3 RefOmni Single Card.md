# FastH3 RefOmni Single Card

[FastH3 family](README.md) · [Omni multi-GPU serving](FastH3-Omni.md)

Run FastH3 reference-conditioned video and audio generation in vLLM-Omni on
one RTX 4090, RTX 5090, DGX Spark, or RTX PRO 6000 Blackwell. Supply a text
prompt and a reference image to generate an MP4 with stereo audio.

These recipes use
[FastVideo/FastVideo-FastH3-Omni-8-Step-V1](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1)
with online FP8 quantization of the DiT and text encoder, BF16 attention,
eight inference steps, and guidance scale 1.

## Choose your hardware profile

| Device | Memory | Attention | Memory placement | Video decoder | Configuration |
| --- | --- | --- | --- | --- | --- |
| RTX 4090 | 24 GB | Triton BF16 | DiT/text layer offload; stage VAEs when needed | Native, FP16 decode | [rtx4090 YAML](fasth3-omni-rtx4090.yaml) |
| RTX 5090 | 32 GB | FlashInfer BF16 | DiT/text layer offload | NVFP4 8L | [rtx5090 YAML](fasth3-omni-rtx5090.yaml) |
| DGX Spark (GB10) | 128 GB unified | FlashInfer BF16 | Resident in unified memory | NVFP4 8L | [spark YAML](fasth3-omni-spark.yaml) |
| RTX PRO 6000 Blackwell | 96 GB | cuDNN BF16 | Resident DiT; text-encoder layer offload | NVFP4 8L | [rtx-pro6000 YAML](fasth3-omni-rtx-pro6000.yaml) |

Each YAML selects one GPU, eager execution, and tiled VAE decoding.
Use the 96 GB **Blackwell** model for the RTX PRO 6000 profile.

## Install

Use Linux and Python 3.12. Follow the
[CUDA installation guide](../../docs/getting_started/installation/gpu.md)
to install compatible vLLM and PyTorch packages, then run the commands below
from the vLLM-Omni repository root in that environment.
On DGX Spark, use the native `aarch64` environment.

Install the optional kernels for your device:

```bash
# RTX 4090, RTX 5090, or DGX Spark
uv pip install -e '.[vsa]'
```

```bash
# RTX PRO 6000 Blackwell
uv pip install -e '.[cudnn-bsa]'
```

The Blackwell profiles also use FlashInfer for the NVFP4 VAE decoder.
See the [attention provider requirements](../../docs/user_guide/diffusion/attention_backends/fastvideo_vsa.md)
for the corresponding CUDA packages.

Install `ffmpeg` using your system package manager. For Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

If the release-pinned MiniMax-H3 components require access, accept their
access terms on Hugging Face and authenticate:

```bash
hf auth login
```

The first start downloads the checkpoint and its pinned text encoder and
VAEs. Loading quantizes eligible weights to FP8; allow host RAM and disk
space for the source weights as well as the runtime model. Subsequent starts
reuse the Hugging Face cache.

## Start the server

Set `FASTH3_PROFILE` to your device:

```bash
FASTH3_PROFILE=rtx4090  # rtx4090 | rtx5090 | spark | rtx-pro6000

CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --omni \
  --revision 88d512c5e7cc38dda499fbb65688ce58611f9c80 \
  --config "recipes/FastVideo/fasth3-omni-${FASTH3_PROFILE}.yaml"
```

Keep this terminal running. The server listens on port 8000; use `--port`
to change it. A local checkpoint directory can replace the model ID.
Wait for startup to finish before sending a request.

## Generate video and audio

In a second terminal, save your reference image as `reference.png` and write
the scene, motion, and desired sound in `prompt.txt`. For example:

```bash
cat > prompt.txt <<'EOF'
The person in the reference image stands in a sunlit garden, smiles,
and waves at the camera. The camera slowly moves closer. Birds chirp
in the background as the person says, "Welcome to the garden!"
EOF
```

Generate a 480P clip:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
  -F 'prompt=<prompt.txt' \
  -F 'input_reference=@reference.png' \
  -F 'size=832x480' \
  -F 'seconds=5' \
  -F 'seed=1234' \
  -o output.mp4
```

Open `output.mp4` in a media player. The response includes video and 32 kHz
stereo audio. A five-second request maps to 124 frames at 24 fps.
For 768P, change `size=832x480` to `size=1344x768`.

For multiple images, video, or audio references, see the
[H3 reference API](../MiniMaxAI/MiniMax-H3.md).

## Choose the video decoder

The RTX 4090 profile uses the native VAE with FP32 encoding and FP16
decoding. Keep this decoder on the 4090.

The RTX 5090, DGX Spark, and RTX PRO 6000 Blackwell profiles use the released
[NVFP4 VAE decoder](https://huggingface.co/FastVideo/FastH3-NVFP4-VAE-Decoder),
with `fast-8L` selected by default. For the deeper decoder, set
`additional-config.video_vae_decoder.subfolder` to `quality-36L` in your
YAML and restart the server. Switch it back to `fast-8L` for faster decoding.

## Memory and startup tips

Keep the memory placement in your selected YAML. On DGX Spark, CPU and GPU
share memory; CPU offload does not provide a separate memory pool.

- **If memory is tight:** close other GPU workloads, use `fast-8L` on
  Blackwell, and start with `832x480`. Keep `vae-use-tiling: true`.
- **If startup is slow:** allow the initial downloads and FP8 conversion
  to finish. The profiles allow up to one hour for initialization.

## Related recipes

See [FastH3 Omni](FastH3-Omni.md) for multi-GPU serving. Upstream references:
[FastVideo #1980](https://github.com/hao-ai-lab/FastVideo/pull/1980) and
[FastVideo #1919](https://github.com/hao-ai-lab/FastVideo/pull/1919).
