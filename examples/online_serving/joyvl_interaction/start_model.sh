#!/bin/bash
# Serve the JoyVL interaction model as a plain VLM (NOT --omni: it is a
# standard Qwen3-VL autoregressive model, not an omni/diffusion model).
set -euo pipefail

MODEL="${MODEL:-ydydy/JoyAI-VL-Interaction-Preview}"
SERVED_NAME="${SERVED_NAME:-JoyAI-VL-Interaction-Preview}"
PORT="${PORT:-8061}"
GPU="${GPU:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

CUDA_VISIBLE_DEVICES="${GPU}" vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt '{"image":32,"video":1}'
