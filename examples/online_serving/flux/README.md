# FLUX.1-schnell / FLUX.1-dev

This example demonstrates how to deploy FLUX.1-schnell (or FLUX.1-dev) model for online image generation service using vLLM-Omni.

## Model Variants

| Model | Steps | Guidance | Description |
|-------|-------|----------|-------------|
| FLUX.1-schnell | 4 | 0.0 | Distilled, optimized for fast generation |
| FLUX.1-dev | 28 | 3.5 | Full model, higher quality |

## Start Server

### Basic Start

```bash
# FLUX.1-schnell (default)
vllm serve black-forest-labs/FLUX.1-schnell --omni --port 8091

# FLUX.1-dev
vllm serve black-forest-labs/FLUX.1-dev --omni --port 8091
```

### Start with Script

```bash
# FLUX.1-schnell (default)
bash run_server.sh

# FLUX.1-dev
MODEL=black-forest-labs/FLUX.1-dev bash run_server.sh
```

## API Calls

### Method 1: Using curl

```bash
# Basic text-to-image generation
bash run_curl_text_to_image.sh

# Or execute directly
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape painting",
    "size": "1024x1024",
    "num_inference_steps": 4,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > output.png
```

### Method 2: Using Python Client

```bash
python openai_chat_client.py --prompt "A beautiful landscape painting" --output output.png
```

## Request Format

### Images Generation API

```json
{
  "prompt": "A beautiful landscape painting",
  "size": "1024x1024",
  "num_inference_steps": 4,
  "seed": 42
}
```

### Chat Completions API

```json
{
  "messages": [
    {"role": "user", "content": "A beautiful landscape painting"}
  ],
  "extra_body": {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 4,
    "seed": 42
  }
}
```

## Generation Parameters

| Parameter | Type | Default (schnell) | Default (dev) | Description |
|-----------|------|:-----------------:|:-------------:|-------------|
| `height` | int | 1024 | 1024 | Image height in pixels |
| `width` | int | 1024 | 1024 | Image width in pixels |
| `size` | str | None | None | Image size (e.g., "1024x1024") |
| `num_inference_steps` | int | 4 | 28 | Number of denoising steps |
| `guidance_scale` | float | 0.0 | 3.5 | Classifier-free guidance scale |
| `seed` | int | None | None | Random seed (reproducible) |
| `negative_prompt` | str | None | None | Negative prompt |
| `num_outputs_per_prompt` | int | 1 | 1 | Number of images to generate |

## File Description

| File | Description |
|------|-------------|
| `run_server.sh` | Server startup script |
| `run_curl_text_to_image.sh` | curl example |
| `openai_chat_client.py` | Python client |
