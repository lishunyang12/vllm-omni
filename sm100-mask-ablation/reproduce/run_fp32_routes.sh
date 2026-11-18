#!/bin/bash
set -Eeuo pipefail
export CUTE_DSL_DISABLE_FILE_CACHING=1 PYTHONPATH=/work/source/candidate-fp32/techniques/sparse_backends:/work/scripts
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
root=/work/results/routes-fp32
mkdir -p "$root"
for layer in 00 25; do
  input="/work/results/ref2va-native-sequence-450/qkv-step-10-layer-$layer-rank-0.pt"
  python3 /work/scripts/route_fp32_audit.py --input "$input" --output "$root/layer-$layer.json" > "$root/layer-$layer.log" 2>&1
done
