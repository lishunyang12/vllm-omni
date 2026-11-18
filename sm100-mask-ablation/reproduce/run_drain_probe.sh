#!/bin/bash
set -Eeuo pipefail
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4 CUTE_DSL_DISABLE_FILE_CACHING=1
export PYTHONPATH=/work/source/route-pv-drain/techniques/sparse_backends
export H3_PROBE_OUTPUT=/work/results/independent-process/route-pv-drain
mkdir -p "$H3_PROBE_OUTPUT"
CUTE_DSL_KEEP=cubin,ptx CUTE_DSL_DUMP_DIR="$H3_PROBE_OUTPUT" timeout -k 10 180 python3 /work/scripts/independent_process_probe.py > "$H3_PROBE_OUTPUT/run.log" 2>&1
cat "$H3_PROBE_OUTPUT/report.json"
