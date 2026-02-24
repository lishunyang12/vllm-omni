#!/bin/bash
# FLUX.1-schnell online serving startup script
#
# FLUX.1-schnell is a distilled variant of FLUX.1 optimized for fast generation.
# It shares the same FluxPipeline architecture as FLUX.1-dev but typically uses
# fewer inference steps (4 vs 28) and does not require guidance (guidance_scale=0).
#
# Usage:
#   bash run_server.sh                                    # FLUX.1-schnell (default)
#   MODEL=black-forest-labs/FLUX.1-dev bash run_server.sh # FLUX.1-dev

MODEL="${MODEL:-black-forest-labs/FLUX.1-schnell}"
PORT="${PORT:-8091}"

echo "Starting FLUX server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT"
