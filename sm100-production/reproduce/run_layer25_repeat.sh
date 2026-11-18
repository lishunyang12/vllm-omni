#!/bin/bash
set -Eeuo pipefail
label=${1:?}
export PYTHONPATH=/work/source/candidate/techniques/sparse_backends
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
root=/work/results/$label
mkdir -p "$root"
python3 /work/source/candidate/techniques/sparse_backends/benchmarks/benchmark_sm100.py \
  --baseline-root /work/source/baseline/techniques/sparse_backends \
  --input /work/captures/qkv-layer-25-rank-0.pt --output "$root/layer-25-repeat.json" \
  > "$root/layer-25-repeat.log" 2>&1
touch "$root/complete"
