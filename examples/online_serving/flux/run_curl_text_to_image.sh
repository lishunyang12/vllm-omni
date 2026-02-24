#!/bin/bash
# FLUX.1-schnell text-to-image curl example
#
# FLUX.1-schnell uses fewer steps (4) and no guidance (guidance_scale=0).
# FLUX.1-dev uses more steps (28) and guidance (guidance_scale=3.5).

curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a dragon laying over the spine of the Green Mountains of Vermont",
    "size": "1024x1024",
    "num_inference_steps": 4,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > flux_output.png
