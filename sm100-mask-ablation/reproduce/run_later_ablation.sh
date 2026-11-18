#!/bin/bash
set -Eeuo pipefail
label=$1
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
mkdir -p "/work/results/$label"
for step in 00 10 25; do
  for layer in 00 25; do
    input="/work/results/ref2va-native-sequence-450/qkv-step-$step-layer-$layer-rank-0.pt"
    python3 /work/scripts/ablation.py --input "$input" --output "/work/results/$label/step-$step-layer-$layer.json" > "/work/results/$label/step-$step-layer-$layer.log" 2>&1
  done
done
touch "/work/results/$label/complete"
