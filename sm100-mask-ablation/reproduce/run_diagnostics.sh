#!/bin/bash
set -Eeuo pipefail
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
mkdir -p /work/results/protected-q-diagnostic
set +e
timeout -k 10 240 python3 /work/scripts/diagnose_protected_q.py --baseline /work/source/candidate --a /work/source/protected-q --c /work/source/protected-q-contiguous --modes eager --repeats 4 --report /work/results/protected-q-diagnostic/disjoint.json > /work/results/protected-q-diagnostic/disjoint.log 2>&1
rc=$?
printf '%s\n' "$rc" > /work/results/protected-q-diagnostic/disjoint.exit
set -e
set +e
mkdir -p /work/results/smem-p-diagnostic
timeout -k 10 240 python3 /work/scripts/diagnose_smem_p_overlap.py --baseline /work/source/candidate --serial /work/source/smem-p-serial --overlap /work/source/smem-p-overlap --publish-first /work/source/smem-p-overlap-publish-first --report /work/results/smem-p-diagnostic/8193-none.json > /work/results/smem-p-diagnostic/8193-none.log 2>&1
printf '%s\n' "$?" > /work/results/smem-p-diagnostic/8193-none.exit
set -e
bash /work/scripts/export_design_resources.sh
