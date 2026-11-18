#!/bin/bash
set -Eeuo pipefail
work=/work
export HF_HOME="$work/hf-cache" UV_CACHE_DIR="$work/uv-cache" XDG_CACHE_HOME="$work/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$work/torchinductor-cache" TRITON_CACHE_DIR="$work/triton-cache" TORCH_HOME="$work/torch-cache"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=600
export PYTHONPATH="$work/bundle:$work/source/head${PYTHONPATH:+:$PYTHONPATH}"
export H3_FORWARD_WARMUP=1 H3_FORWARD_REPEATS=3 CUDA_LAUNCH_BLOCKING=1
export H3_FORWARD_RESULT="$work/results/ref2va-SOL_ATTN-blocking-diagnostic"
mkdir -p "$H3_FORWARD_RESULT"
cd "$work/source/head"
timeout --kill-after=15s 300s python3 "$work/bundle/benchmark_case.py" --work "$work" --task ref2va --backend SOL_ATTN > "$H3_FORWARD_RESULT/run.log" 2>&1
