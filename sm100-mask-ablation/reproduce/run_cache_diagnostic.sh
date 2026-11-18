#!/bin/bash
set -Eeuo pipefail
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4 CUTE_DSL_DISABLE_FILE_CACHING=1
bash /work/scripts/run_extension_probe.sh
export PYTHONPATH=/work/source/candidate/techniques/sparse_backends
python3 /work/scripts/compare_independent.py > /work/results/independent-process/comparison.log 2>&1
mkdir -p /work/results/no-cache-diagnostic
set +e
timeout -k 10 240 python3 /work/scripts/diagnose_smem_p_overlap.py --baseline /work/source/candidate --serial /work/source/smem-p-serial --overlap /work/source/smem-p-vector --report /work/results/no-cache-diagnostic/8193-none.json > /work/results/no-cache-diagnostic/8193-none.log 2>&1
printf '%s\n' "$?" > /work/results/no-cache-diagnostic/8193-none.exit
