#!/bin/bash
set -Eeuo pipefail
export CUTE_DSL_DISABLE_FILE_CACHING=1
work=${1:-/work}
python3 /work/scripts/require_contiguous_validation.py "$work"
artifact=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HF_HOME="$work/hf-cache" HF_HUB_OFFLINE=1 TORCH_HOME="$work/torch-cache"
export XDG_CACHE_HOME="$work/xdg-cache" TRITON_CACHE_DIR="$work/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$work/torchinductor-cache" OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600
export H3_SOL_SOURCE_DIR=${H3_SOL_SOURCE_DIR:-$work/source/${H3_DESIGN_VARIANT:?}}
export PYTHONPATH="$artifact:$H3_SOL_SOURCE_DIR/techniques/sparse_backends:$work/source/native"
export H3_SOURCE_COMMIT=3da353c45224f5863808da320d694be045c16ef8
export H3_UPSTREAM_COMMIT=${H3_UPSTREAM_COMMIT:-e3f60d5fdf15ddab41c6079c17fccc8d10d62c29}
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=3600 VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=3600
export H3_GPU_COUNT=${H3_GPU_COUNT:-2}
export H3_VARIANT=${H3_DESIGN_VARIANT:?} H3_FORWARD_WARMUP=3 H3_FORWARD_REPEATS=9
export H3_FORWARD_RESULT="$work/results/ref2va-native-${H3_RUN_ID:-1}"
mkdir -p "$H3_FORWARD_RESULT"
extra=()
if [[ ${H3_CPU_OFFLOAD:-auto} == 1 ]] || { [[ ${H3_CPU_OFFLOAD:-auto} == auto ]] && python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.get_device_properties(0).total_memory < 110*1024**3 else 1)'; }; then
  extra=(--cpu-offload)
fi
cd "$work/source/native"
python3 "$artifact/contiguous_forward_case.py" --work "$work" --task ref2va --backend SOL_ATTN "${extra[@]}" > "$H3_FORWARD_RESULT/run.log" 2>&1
cat "$H3_FORWARD_RESULT/summary.json"
