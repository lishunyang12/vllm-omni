# VoxCPM2 Offline Inference

VoxCPM2 is a 2B-parameter tokenizer-free diffusion AR TTS model with 48kHz output, 30+ languages, voice design, and voice cloning.

## Quick Start

```bash
# Zero-shot synthesis
python examples/offline_inference/voxcpm2/end2end.py \
    --model openbmb/VoxCPM2 \
    --text "Hello, this is a VoxCPM2 demo."

# Voice design (describe the voice you want)
python examples/offline_inference/voxcpm2/end2end.py \
    --model openbmb/VoxCPM2 \
    --text "Hello, this is a voice design demo." \
    --control-instruction "A warm female voice with happy tone"

# Voice cloning (no transcript needed in v2)
python examples/offline_inference/voxcpm2/end2end.py \
    --model openbmb/VoxCPM2 \
    --text "Hello, this is a voice cloning demo." \
    --reference-audio path/to/reference.wav

# Streaming mode
python examples/offline_inference/voxcpm2/end2end.py \
    --model openbmb/VoxCPM2 \
    --text "Hello, streaming demo." \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm2.yaml
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cfg-value` | 2.0 | Classifier-free guidance (1.0-3.0 recommended) |
| `--inference-timesteps` | 10 | Diffusion steps (4-30 recommended) |
| `--temperature` | 1.0 | Sampling temperature |
| `--top-p` | 1.0 | Nucleus sampling threshold |
| `--control-instruction` | None | Voice design description |
| `--reference-audio` | None | Voice cloning reference |
