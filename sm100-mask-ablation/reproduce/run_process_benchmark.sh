#!/bin/bash
set -Eeuo pipefail
label=${1:?}
export CUTE_DSL_DISABLE_FILE_CACHING=1 TRITON_CACHE_DIR=/work/triton-cache XDG_CACHE_HOME=/work/xdg-cache OMP_NUM_THREADS=4
mkdir -p "/work/results/$label"
for layer in 00 25; do
  python3 /work/scripts/process_benchmark.py \
    --baseline /work/source/extension-fp32 --candidate /work/source/candidate-fp32 \
    --input "/work/results/ref2va-native-sequence-450/qkv-step-10-layer-$layer-rank-0.pt" \
    --report "/work/results/$label/layer-$layer.json" \
    > "/work/results/$label/layer-$layer.log" 2>&1
done
touch "/work/results/$label/complete"
