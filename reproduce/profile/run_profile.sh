#!/bin/bash
set -Eeuo pipefail
work=${1:-/work}
export HF_HOME="$work/hf-cache" UV_CACHE_DIR="$work/uv-cache" XDG_CACHE_HOME="$work/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$work/torchinductor-cache" TRITON_CACHE_DIR="$work/triton-cache" TORCH_HOME="$work/torch-cache"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export PYTHONPATH="$work/profile:$work/source/head${PYTHONPATH:+:$PYTHONPATH}"
export H3_FORWARD_WARMUP=3 H3_FORWARD_REPEATS=3
cd "$work/source/head"
extra=()
if python3 -c 'import torch,sys;sys.exit(0 if torch.cuda.get_device_properties(0).total_memory < 110*1024**3 else 1)'; then
 extra=(--cpu-offload)
fi
task=${H3_PROFILE_TASK:-ref2va}
backend=${H3_PROFILE_BACKEND:-SOL_ATTN}
export H3_PROFILE_MODE=${H3_PROFILE_MODE:-events}
export H3_FORWARD_RESULT="$work/profile/results/$task-$backend-${H3_PROFILE_TAG:-$H3_PROFILE_MODE}"
mkdir -p "$H3_FORWARD_RESULT"
python3 "$work/profile/profile_case.py" --work "$work" --task "$task" --backend "$backend" "${extra[@]}" > "$H3_FORWARD_RESULT/run.log" 2>&1
