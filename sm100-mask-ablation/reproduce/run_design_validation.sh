#!/bin/bash
set -Eeuo pipefail
export OMP_NUM_THREADS=4 XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache
mkdir -p /work/results/design-validation
for variant in protected-q p-store-x16; do
  timeout 900 python3 /work/scripts/validate_design.py --baseline /work/source/candidate --candidate "/work/source/$variant" --report "/work/results/design-validation/$variant-full.json" > "/work/results/design-validation/$variant-full.log" 2>&1
done
