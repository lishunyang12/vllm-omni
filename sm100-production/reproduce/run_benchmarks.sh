#!/bin/bash
set -Eeuo pipefail
label=${1:?result label}
export PYTHONPATH=/work/source/candidate/techniques/sparse_backends
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
test "$(cat /work/results/tests-final-exit-code.txt)" = 0
root=/work/results/$label
mkdir -p "$root"
bench=/work/source/candidate/techniques/sparse_backends/benchmarks/benchmark_sm100.py
base=/work/source/baseline/techniques/sparse_backends
python3 "$bench" --baseline-root "$base" --output "$root/matrix.json" > "$root/matrix.log" 2>&1
for layer in 00 12 25 49; do
  python3 "$bench" --baseline-root "$base" --input "/work/captures/qkv-layer-$layer-rank-0.pt" \
    --output "$root/layer-$layer.json" > "$root/layer-$layer.log" 2>&1
done
python3 "$bench" --baseline-root "$base" --input /work/captures/qkv-layer-00-rank-0.pt \
  --output "$root/layer-00-repeat.json" > "$root/layer-00-repeat.log" 2>&1
touch "$root/complete"
