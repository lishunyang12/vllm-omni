#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Serve MiniMax-H3 FL2VA with global online FP8 on one GPU and no offload.

set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:?Set MODEL_ROOT to the local MiniMax-H3 repository}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8091}"
RUN_DIR="${RUN_DIR:-$PWD/h3_single_gpu_fp8_no_offload}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT%/}/FL2VA}"

if [[ ! -f "${MODEL_PATH}/model_index.json" ]]; then
  echo "MiniMax-H3 FL2VA partition not found: ${MODEL_PATH}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is required" >&2
  exit 2
fi
if ! command -v "${VLLM_BIN}" >/dev/null; then
  echo "vLLM executable not found: ${VLLM_BIN}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"

nvidia-smi -i "${GPU_ID}" \
  --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader | tee "${RUN_DIR}/gpu.txt"
if git rev-parse HEAD 2>/dev/null | tee "${RUN_DIR}/commit.txt"; then
  :
else
  printf 'unknown\n' | tee "${RUN_DIR}/commit.txt"
fi

nvidia-smi -i "${GPU_ID}" \
  --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop-ms=200 >"${RUN_DIR}/gpu.csv" &
GPU_MONITOR_PID=$!

summarize_memory() {
  if kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
  wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  if [[ ! -s "${RUN_DIR}/gpu.csv" ]]; then
    echo "GPU sampler produced no data" >&2
    return
  fi
  awk -F, '
    {
      total=$4; used=$5
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", total)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", used)
      if (total > max_total) max_total=total
      if (used > max_used) max_used=used
    }
    END {
      printf "memory_total_mib=%.0f\npeak_used_mib=%.0f\nheadroom_mib=%.0f\n", \
        max_total, max_used, max_total-max_used
    }
  ' "${RUN_DIR}/gpu.csv" | tee "${RUN_DIR}/memory_summary.txt"
}
trap summarize_memory EXIT

echo "Starting MiniMax-H3 single-GPU global-FP8 server"
echo "Model: ${MODEL_PATH}"
echo "GPU: ${GPU_ID}"
echo "Port: ${PORT}"
echo "Artifacts: ${RUN_DIR}"
echo "Offload: disabled"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
"${VLLM_BIN}" serve "${MODEL_PATH}" \
  --omni \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 1 \
  --tensor-parallel-size 1 \
  --usp 1 \
  --ring 1 \
  --text-encoder-tp-size 1 \
  --vae-patch-parallel-size 1 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --quantization fp8 \
  --enforce-eager \
  --enable-diffusion-pipeline-profiler \
  --diffusion-attention-backend CUDNN_ATTN \
  2>&1 | tee "${RUN_DIR}/server.log"
