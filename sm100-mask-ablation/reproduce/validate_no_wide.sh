#!/bin/bash
set -Eeuo pipefail
export CUTE_DSL_DISABLE_FILE_CACHING=1 XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
export PYTHONPATH=/work/pr594-ablation/150-no-wide/techniques/sparse_backends
root=/work/results/pr594-fix-validation
mkdir -p "$root"
set +e
timeout -k 10 900 python3 /work/scripts/validate_design.py --baseline /work/source/extension --candidate /work/pr594-ablation/150-no-wide --tests-root /work/source/candidate/techniques/sparse_backends/tests --report "$root/no-wide-full.json" > "$root/no-wide-full.log" 2>&1
printf '%s\n' "$?" > "$root/no-wide-full.exit"
