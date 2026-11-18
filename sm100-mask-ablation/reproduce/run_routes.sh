#!/bin/bash
set -Eeuo pipefail
export PYTHONPATH=/work/source/candidate/techniques/sparse_backends:/work/scripts
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
root=/work/results/routes
mkdir -p "$root"
for step in 00 10 25; do
  for layer in 00 25; do
    input="/work/results/ref2va-native-sequence-450/qkv-step-$step-layer-$layer-rank-0.pt"
    python3 /work/scripts/route_audit.py --input "$input" --output "$root/step-$step-layer-$layer.json" > "$root/step-$step-layer-$layer.log" 2>&1
  done
done
