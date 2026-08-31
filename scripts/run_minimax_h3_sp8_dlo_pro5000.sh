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
PROMPT_FILE="$REPO/examples/offline_inference/minimax_h3/prompts/starship_hyperspace_10s.txt"
STEPS=${MINIMAX_H3_STEPS:-2}
DURATION=${MINIMAX_H3_DURATION:-10.0}
REPETITIONS=${MINIMAX_H3_REPETITIONS:-1}
RESIDENT_LAYERS=${MINIMAX_H3_DLO_RESIDENT_LAYERS:-35}
ULYSSES_A2A_PERMUTE=${MINIMAX_H3_ULYSSES_A2A_PERMUTE:-0}
GPU_ORDER=${MINIMAX_H3_GPU_ORDER:-0,4,1,5,2,6,3,7}
IFS=',' read -r -a SELECTED_GPUS <<<"$GPU_ORDER"
SP_SIZE=${#SELECTED_GPUS[@]}

if [[ "$ULYSSES_A2A_PERMUTE" != 0 && "$ULYSSES_A2A_PERMUTE" != 1 ]]; then
  echo "ERROR: MINIMAX_H3_ULYSSES_A2A_PERMUTE must be 0 or 1" >&2
  exit 2
fi
if [[ "$SP_SIZE" != 4 && "$SP_SIZE" != 8 ]]; then
  echo "ERROR: MINIMAX_H3_GPU_ORDER must select exactly 4 or 8 GPUs" >&2
  exit 2
fi
if [[ "$ULYSSES_A2A_PERMUTE" == 1 ]]; then
  ULYSSES_TRANSPORT=lsa
else
  ULYSSES_TRANSPORT=regular
fi

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

CUDA_TOOLKIT=unused
TORCH_CUDA_VERSION=$(
  "$PYTHON" -c 'import torch; print(torch.version.cuda or "")'
)
if [[ "$ULYSSES_A2A_PERMUTE" == 1 ]]; then
  [[ -n "$TORCH_CUDA_VERSION" ]] || {
    echo "ERROR: LSA requires a CUDA-enabled PyTorch build" >&2
    exit 2
  }

  CUDA_CANDIDATES=()
  if [[ -n "${MINIMAX_H3_CUDA_HOME:-}" ]]; then
    CUDA_CANDIDATES+=("$MINIMAX_H3_CUDA_HOME")
  else
    [[ -n "${CUDA_HOME:-}" ]] && CUDA_CANDIDATES+=("$CUDA_HOME")
    for candidate in "$VENV"/lib/python*/site-packages/nvidia/cu13; do
      CUDA_CANDIDATES+=("$candidate")
    done
    CUDA_CANDIDATES+=("/usr/local/cuda-$TORCH_CUDA_VERSION" "/usr/local/cuda")
  fi

  for candidate in "${CUDA_CANDIDATES[@]}"; do
    [[ -x "$candidate/bin/nvcc" ]] || continue
    NVCC_VERSION=$(
      "$candidate/bin/nvcc" --version |
        awk '/release/ {sub(/.*release /, ""); sub(/,.*/, ""); print; exit}'
    )
    if [[ "$NVCC_VERSION" == "$TORCH_CUDA_VERSION" ]]; then
      CUDA_TOOLKIT=$(cd -- "$candidate" && pwd -P)
      break
    fi
  done

  if [[ "$CUDA_TOOLKIT" == unused ]]; then
    echo "ERROR: LSA JIT requires nvcc $TORCH_CUDA_VERSION to match PyTorch; no matching CUDA toolkit found" >&2
    echo "ERROR: set MINIMAX_H3_CUDA_HOME to the matching toolkit root" >&2
    exit 2
  fi
  export CUDA_HOME=$CUDA_TOOLKIT
  export PATH="$CUDA_TOOLKIT/bin:$PATH"
  if [[ -d "$CUDA_TOOLKIT/lib" ]]; then
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib:${LD_LIBRARY_PATH:-}"
  elif [[ -d "$CUDA_TOOLKIT/lib64" ]]; then
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64:${LD_LIBRARY_PATH:-}"
  fi
fi

[[ -f "$MODEL/model_index.json" ]] || {
  echo "ERROR: MiniMax-H3 FL2VA model not found: $MODEL" >&2
  exit 2
}
[[ -s "$PROMPT_FILE" ]] || {
  echo "ERROR: prompt file not found or empty: $PROMPT_FILE" >&2
  exit 2
}

GPU_STATE=$(nvidia-smi \
  --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits)
for GPU_ID in "${SELECTED_GPUS[@]}"; do
  MEMORY_USED=$(awk -F',' -v gpu="$GPU_ID" '$1 + 0 == gpu {gsub(/ /, "", $2); print $2}' <<<"$GPU_STATE")
  GPU_UTIL=$(awk -F',' -v gpu="$GPU_ID" '$1 + 0 == gpu {gsub(/ /, "", $3); print $3}' <<<"$GPU_STATE")
  if [[ -z "$MEMORY_USED" || -z "$GPU_UTIL" ]]; then
    echo "ERROR: selected GPU $GPU_ID is not visible" >&2
    exit 3
  fi
  if (( MEMORY_USED > 2048 || GPU_UTIL > 10 )); then
    echo "ERROR: selected GPU $GPU_ID is busy: memory=${MEMORY_USED} MiB, utilization=${GPU_UTIL}%" >&2
    exit 3
  fi
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_ROOT="$ROOT/results/dlo-sp$SP_SIZE-$ULYSSES_TRANSPORT-offline-$STAMP"
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
  --sp-size "$SP_SIZE"
  --steps "$STEPS"
  --repetitions "$REPETITIONS"
  --seed 0
  --resident-layers "$RESIDENT_LAYERS"
  --duration "$DURATION"
  --width 1344
  --height 768
  --prompt-file "$PROMPT_FILE"
  --output "$RESULT_ROOT/summary.json"
  --video-output "$RESULT_ROOT/smoke.mp4"
)
if [[ "$ULYSSES_A2A_PERMUTE" == 1 ]]; then
  CMD+=(--ulysses-a2a-permute)
fi

cp "$PROMPT_FILE" "$RESULT_ROOT/prompt.txt"

{
  echo "utc_start=$STAMP"
  echo "repo=$REPO"
  echo "commit=$(git rev-parse HEAD)"
  echo "model=$MODEL"
  echo "prompt_file=$PROMPT_FILE"
  echo "duration=$DURATION"
  echo "physical_gpu_order=$GPU_ORDER"
  echo "parallelism=DP1_TP1_SP${SP_SIZE}_RING1_TE${SP_SIZE}_VAE${SP_SIZE}"
  echo "ulysses_mode=strict"
  echo "ulysses_a2a_permute=$ULYSSES_A2A_PERMUTE"
  echo "ulysses_transport=$ULYSSES_TRANSPORT"
  echo "torch_cuda=$TORCH_CUDA_VERSION"
  echo "cuda_toolkit=$CUDA_TOOLKIT"
  echo "dlo_use_allgather=true"
  echo "dlo_resident_layers=$RESIDENT_LAYERS"
  echo "steps=$STEPS"
  echo "repetitions=$REPETITIONS"
  "$PYTHON" --version
  if [[ "$ULYSSES_A2A_PERMUTE" == 1 ]]; then
    "$CUDA_TOOLKIT/bin/nvcc" --version
  fi
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
