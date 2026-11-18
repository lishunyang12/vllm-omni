#!/bin/bash
set -Eeuo pipefail
export CUTE_DSL_DISABLE_FILE_CACHING=1 TRITON_CACHE_DIR=/work/triton-cache XDG_CACHE_HOME=/work/xdg-cache OMP_NUM_THREADS=4
for variant in extension candidate; do
  export PYTHONPATH="/work/source/$variant/techniques/sparse_backends"
  export H3_PROBE_OUTPUT="/work/results/frozen-prepared/$variant"
  mkdir -p "$H3_PROBE_OUTPUT"
  PYTHONPYCACHEPREFIX="$H3_PROBE_OUTPUT/pycache" python3 /work/scripts/frozen_prepare_probe.py > "$H3_PROBE_OUTPUT/run.log" 2>&1
done
export PYTHONPATH=/work/source/candidate-fp32/techniques/sparse_backends H3_SWEEP_OUTPUT=/work/results/preprocess-fp32-sweep
python3 /work/scripts/sweep_preprocess.py > /work/results/preprocess-fp32-sweep.log 2>&1
