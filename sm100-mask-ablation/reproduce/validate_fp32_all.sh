#!/bin/bash
set -Eeuo pipefail
export CUTE_DSL_DISABLE_FILE_CACHING=1 TRITON_CACHE_DIR=/work/triton-cache XDG_CACHE_HOME=/work/xdg-cache OMP_NUM_THREADS=4
export PYTHONPATH=/work/source/candidate-fp32/techniques/sparse_backends
root=/work/results/fp32-validation
mkdir -p "$root"
set +e
python3 -m pytest -q /work/scripts/test_preprocess_precision.py --junitxml="$root/pooling.xml" > "$root/pooling.log" 2>&1
printf '%s\n' "$?" > "$root/pooling.exit"
timeout -k 10 900 python3 /work/scripts/validate_design.py --baseline /work/source/extension-fp32 --candidate /work/source/candidate-fp32 --tests-root /work/source/candidate-fp32/techniques/sparse_backends/tests --report "$root/full.json" > "$root/full.log" 2>&1
printf '%s\n' "$?" > "$root/full.exit"
