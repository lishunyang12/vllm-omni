#!/bin/bash
set -Eeuo pipefail
label=$1
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
mkdir -p "/work/results/$label"
for variant in protected-q-contiguous; do
  export PYTHONPATH="/work/source/$variant/techniques/sparse_backends"
  bench="/work/source/$variant/techniques/sparse_backends/benchmarks/benchmark_sm100.py"
  for layer in 00 25; do
    timeout 240 python3 "$bench" --baseline-root /work/source/candidate/techniques/sparse_backends --input "/work/captures/qkv-layer-$layer-rank-0.pt" --output "/work/results/$label/$variant-layer-$layer.json" > "/work/results/$label/$variant-layer-$layer.log" 2>&1
  done
done
touch "/work/results/$label/complete"
