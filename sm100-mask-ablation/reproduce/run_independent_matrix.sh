#!/bin/bash
set -Eeuo pipefail
export CUTE_DSL_DISABLE_FILE_CACHING=1 XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
for variant in extension alpha-only; do
  if [[ $variant == extension ]]; then source=/work/source/extension; else source=/work/pr594-ablation/63-alpha-only; fi
  export PYTHONPATH="$source/techniques/sparse_backends"
  output="/work/results/independent-matrix/$variant"
  mkdir -p "$output"
  timeout -k 10 600 python3 /work/scripts/independent_matrix.py --output "$output" > "$output/run.log" 2>&1
done
