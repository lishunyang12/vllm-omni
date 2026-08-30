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
STEPS=${MINIMAX_H3_STEPS:-2}
GPU_ORDER=0,4,1,5,2,6,3,7

for command_name in nvidia-smi numactl nohup setsid; do
  if ! command -v "$command_name" >/dev/null; then
    echo "ERROR: required command not found: $command_name" >&2
    exit 2
  fi
done

cd "$REPO"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: checkout is dirty: $REPO" >&2
  git status --short >&2
  exit 2
fi

PYTHON="$VENV/bin/python"
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
  echo "ERROR: this profile requires exactly eight GPUs" >&2
  exit 3
fi

ACTIVE_PIDS=$(nvidia-smi \
  --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {print $1}')
if [[ -n "$ACTIVE_PIDS" ]]; then
  echo "ERROR: GPUs have active compute processes" >&2
  printf '%s\n' "$ACTIVE_PIDS" >&2
  exit 3
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_ROOT="$ROOT/results/dlo-sp8-offline-$STAMP"
RUN_LOG="$RESULT_ROOT/run.log"
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

CMD=(
  numactl --interleave=0,1
  "$PYTHON"
  "$REPO/examples/offline_inference/minimax_h3/dlo_lifecycle.py"
  --model "$MODEL"
  --mode dlo
  --dp-size 1
  --sp-size 8
  --steps "$STEPS"
  --seed 0
  --duration 5.0
  --width 1344
  --height 768
  --output "$RESULT_ROOT/summary.json"
  --video-output "$RESULT_ROOT/smoke.mp4"
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
  echo "steps=$STEPS"
  "$PYTHON" --version
  "$PYTHON" -m pip show vllm vllm-omni
  nvidia-smi
  nvidia-smi topo -m
  numactl --hardware
} >"$RESULT_ROOT/environment.txt" 2>&1

"$PYTHON" -m pip freeze >"$RESULT_ROOT/packages.txt"
printf '%q ' "${CMD[@]}" >"$RESULT_ROOT/command.txt"
printf '\n' >>"$RESULT_ROOT/command.txt"

nohup setsid "${CMD[@]}" >"$RUN_LOG" 2>&1 </dev/null &
RUN_PID=$!
echo "$RUN_PID" >"$RESULT_ROOT/run.pid"

sleep 3
if ! kill -0 "$RUN_PID" 2>/dev/null; then
  echo "ERROR: offline inference exited during startup" >&2
  tail -n 100 "$RUN_LOG" >&2
  exit 4
fi

RUN_PGID=$(ps -o pgid= -p "$RUN_PID" | tr -d ' ')
if [[ ! "$RUN_PGID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: failed to resolve inference process group" >&2
  exit 4
fi
echo "$RUN_PGID" >"$RESULT_ROOT/run.pgid"
ps -o pid,pgid,sid,stat,etime,cmd -p "$RUN_PID" \
  >"$RESULT_ROOT/run.process"

echo "RESULT_ROOT=$RESULT_ROOT"
echo "RUN_PID=$RUN_PID"
echo "RUN_PGID=$RUN_PGID"
echo "RUN_LOG=$RUN_LOG"
echo "TAIL_COMMAND=tail -f $RUN_LOG"
echo "STOP_COMMAND=kill -TERM -- -$RUN_PGID"
