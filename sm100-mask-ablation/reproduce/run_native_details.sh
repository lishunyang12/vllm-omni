#!/bin/bash
set -Eeuo pipefail
export CUTE_DSL_DISABLE_FILE_CACHING=1 XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
PYTHONPATH=/work/source/extension/techniques/sparse_backends timeout -k 10 240 python3 /work/scripts/matrix_oracle.py > /work/results/independent-matrix/oracle.log 2>&1
for variant in upstream extension candidate alpha-only; do
  if [[ $variant == alpha-only ]]; then source=/work/pr594-ablation/63-alpha-only; else source=/work/source/$variant; fi
  export PYTHONPATH="$source/techniques/sparse_backends"
  export H3_DETAILS_OUTPUT="/work/results/native-details/$variant"
  mkdir -p "$H3_DETAILS_OUTPUT"
  PYTHONPYCACHEPREFIX="$H3_DETAILS_OUTPUT/pycache" timeout -k 10 120 python3 /work/scripts/native_details.py > "$H3_DETAILS_OUTPUT/run.log" 2>&1
done
