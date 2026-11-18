#!/bin/bash
set -Eeuo pipefail
work=${1:-/work}
export HF_HOME="$work/hf-cache" UV_CACHE_DIR="$work/uv-cache" XDG_CACHE_HOME="$work/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$work/torchinductor-cache" TRITON_CACHE_DIR="$work/triton-cache" TORCH_HOME="$work/torch-cache"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
variant=${H3_VARIANT:?Set reference, fused, or dense}
source_dir="$work/source/head"
backend=SOL_ATTN
if test "$variant" = fused; then source_dir="$work/source/fused"; fi
if test "$variant" = precise; then source_dir="$work/source/precise"; fi
if test "$variant" = dense; then backend=CUDNN_ATTN; fi
export PYTHONPATH="$work/shadow:$source_dir"
export H3_FORWARD_WARMUP=0 H3_FORWARD_REPEATS=3
cd "$source_dir"
extra=()
if python3 -c 'import torch,sys;sys.exit(0 if torch.cuda.get_device_properties(0).total_memory < 110*1024**3 else 1)'; then
 extra=(--cpu-offload)
fi
export H3_FORWARD_RESULT="$work/fused/results/${H3_RUN_ID:-ref2va-$variant}"
mkdir -p "$H3_FORWARD_RESULT"
python3 "$work/shadow/benchmark_case.py" --work "$work" --task ref2va --backend "$backend" "${extra[@]}" > "$H3_FORWARD_RESULT/run.log" 2>&1
