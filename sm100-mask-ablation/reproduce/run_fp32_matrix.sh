#!/bin/bash
set -Eeuo pipefail
export CUTE_DSL_DISABLE_FILE_CACHING=1 XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
for variant in extension-fp32 candidate-fp32; do
  export PYTHONPATH="/work/source/$variant/techniques/sparse_backends"
  output="/work/results/fp32-matrix/$variant"
  mkdir -p "$output"
  PYTHONPYCACHEPREFIX="$output/pycache" timeout -k 10 600 python3 /work/scripts/independent_matrix.py --output "$output" > "$output/run.log" 2>&1
done
