#!/bin/bash
set -Eeuo pipefail
work=${1:-/work}
export HF_HOME="$work/hf-cache" HF_HUB_OFFLINE=1 TORCH_HOME="$work/torch-cache"
export XDG_CACHE_HOME="$work/xdg-cache" TRITON_CACHE_DIR="$work/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$work/torchinductor-cache" OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export PYTHONPATH="$work/study:$work/source/sana/techniques/sparse_backends:$work/source/native"
export H3_SOURCE_COMMIT=3da353c45224f5863808da320d694be045c16ef8
export H3_UPSTREAM_COMMIT=088897e0c6a9cb970f00b43ab2177c3be7336b36
export H3_GPU_COUNT=${H3_GPU_COUNT:-1} H3_VARIANT=capture
export H3_FORWARD_RESULT="$work/results/qkv-${H3_RUN_ID:-capture}"
mkdir -p "$H3_FORWARD_RESULT"
cp "$work/native/input-manifest.json" "$work/study/input-manifest.json"
python3 "$work/study/make_capture_case.py" "$work/native/benchmark_case.py" "$work/study/capture_case.py"
cd "$work/source/native"
extra=()
if [[ ${H3_CPU_OFFLOAD:-1} == 1 ]]; then extra=(--cpu-offload); fi
python3 "$work/study/capture_case.py" --work "$work" --task ref2va --backend SOL_ATTN "${extra[@]}" > "$H3_FORWARD_RESULT/run.log" 2>&1
cat "$H3_FORWARD_RESULT/summary.json"
