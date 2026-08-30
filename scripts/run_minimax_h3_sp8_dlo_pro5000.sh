#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

DEFAULT_ROOT=/lustre/raplab/client/sylarl/minimax-h3-native
ROOT=${MINIMAX_H3_ROOT:-$DEFAULT_ROOT}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
VENV=${VLLM_OMNI_VENV:-"$REPO/../.venv-vllm028"}
MODEL=${MINIMAX_H3_MODEL:-"$ROOT/MiniMax-H3/FL2VA"}
PORT=${VLLM_OMNI_PORT:-8091}
GPU_ORDER=0,4,1,5,2,6,3,7

require_command() {
  if ! command -v "$1" >/dev/null; then
    echo "ERROR: required command not found: $1" >&2
    exit 2
  fi
}

for command_name in nvidia-smi numactl nohup setsid ss; do
  require_command "$command_name"
done

cd "$REPO"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: checkout is dirty: $REPO" >&2
  git status --short >&2
  exit 2
fi

VLLM="$VENV/bin/vllm"
PYTHON="$VENV/bin/python"

[[ -x "$VLLM" ]] || {
  echo "ERROR: vllm executable not found: $VLLM" >&2
  exit 2
}
[[ -x "$PYTHON" ]] || {
  echo "ERROR: Python executable not found: $PYTHON" >&2
  exit 2
}
[[ -f "$MODEL/model_index.json" ]] || {
  echo "ERROR: MiniMax-H3 FL2VA model not found: $MODEL" >&2
  exit 2
}

AVAILABLE=$(nvidia-smi --query-gpu=index --format=csv,noheader)
if [[ $(wc -l <<<"$AVAILABLE") -ne 8 ]]; then
  echo "ERROR: this profile requires exactly eight visible GPUs" >&2
  printf '%s\n' "$AVAILABLE" >&2
  exit 3
fi

ACTIVE_PIDS=$(nvidia-smi \
  --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {print $1}')
if [[ -n "$ACTIVE_PIDS" ]]; then
  echo "ERROR: selected GPUs have active compute processes" >&2
  printf '%s\n' "$ACTIVE_PIDS" >&2
  exit 3
fi

if [[ -n "$(ss -ltnH "sport = :$PORT")" ]]; then
  echo "ERROR: port $PORT is already listening" >&2
  exit 3
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_ROOT="$ROOT/results/dlo-sp8-main-$STAMP"
SERVER_LOG="$RESULT_ROOT/server.log"
mkdir -p "$RESULT_ROOT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_ORDER
export PYTHONPATH=$REPO
export PYTHONNOUSERSITE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export HF_HOME="$ROOT/hf-cache"
export XDG_CACHE_HOME="$ROOT/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/torchinductor-cache"
export TRITON_CACHE_DIR="$ROOT/triton-cache"

ARGS=(
  serve "$MODEL"
  --omni
  --host 127.0.0.1
  --port "$PORT"
  --trust-remote-code
  --task-type fl2va
  --num-gpus 8
  --data-parallel-size 1
  --tensor-parallel-size 1
  --usp 8
  --ring 1
  --text-encoder-tp-size 8
  --vae-patch-parallel-size 8
  --vae-parallel-mode tile
  --vae-use-tiling
  --enable-distributed-layerwise-offload
  --dlo-use-allgather
  --dlo-resident-layers 0
  --enforce-eager
  --diffusion-attention-backend CUDNN_ATTN
)

{
  echo "utc_start=$STAMP"
  echo "repo=$REPO"
  echo "commit=$(git rev-parse HEAD)"
  echo "model=$MODEL"
  echo "physical_gpu_order=$GPU_ORDER"
  echo "parallelism=DP1_TP1_SP8_RING1_TE8_VAE8"
  echo "dlo_use_allgather=true"
  echo "dlo_resident_layers=0"
  "$PYTHON" --version
  "$VLLM" --version
  "$PYTHON" -m pip show vllm vllm-omni
  nvidia-smi
  nvidia-smi topo -m
  numactl --hardware
} >"$RESULT_ROOT/environment.txt" 2>&1

"$PYTHON" -m pip freeze >"$RESULT_ROOT/packages.txt"
printf '%q ' "$VLLM" "${ARGS[@]}" >"$RESULT_ROOT/command.txt"
printf '\n' >>"$RESULT_ROOT/command.txt"

CMD=(numactl --interleave=0,1 "$VLLM" "${ARGS[@]}")
nohup setsid "${CMD[@]}" >"$SERVER_LOG" 2>&1 </dev/null &
SERVER_PID=$!
echo "$SERVER_PID" >"$RESULT_ROOT/server.pid"

sleep 3
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "ERROR: server exited during initial startup" >&2
  tail -n 100 "$SERVER_LOG" >&2
  exit 4
fi

SERVER_PGID=$(ps -o pgid= -p "$SERVER_PID" | tr -d ' ')
if [[ ! "$SERVER_PGID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: failed to resolve server process group" >&2
  exit 4
fi

echo "$SERVER_PGID" >"$RESULT_ROOT/server.pgid"
ps -o pid,pgid,sid,stat,etime,cmd -p "$SERVER_PID" \
  >"$RESULT_ROOT/server.process"

echo "RESULT_ROOT=$RESULT_ROOT"
echo "SERVER_PID=$SERVER_PID"
echo "SERVER_PGID=$SERVER_PGID"
echo "SERVER_LOG=$SERVER_LOG"
echo "TAIL_COMMAND=tail -f $SERVER_LOG"
echo "STOP_COMMAND=kill -TERM -- -$SERVER_PGID"
