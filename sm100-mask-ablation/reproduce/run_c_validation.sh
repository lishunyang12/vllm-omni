#!/bin/bash
set -Eeuo pipefail
export OMP_NUM_THREADS=4 XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache
mkdir -p /work/results/design-validation
export PYTHONPATH=/work/source/protected-q-contiguous/techniques/sparse_backends
timeout 240 python3 /work/source/protected-q-contiguous/techniques/sparse_backends/benchmarks/benchmark_sm100.py --baseline-root /work/source/candidate/techniques/sparse_backends --stress-only --output /work/results/design-validation/protected-q-contiguous-stress.json > /work/results/design-validation/protected-q-contiguous-stress.log 2>&1
timeout 900 python3 /work/scripts/validate_design.py --baseline /work/source/candidate --candidate /work/source/protected-q-contiguous --report /work/results/design-validation/protected-q-contiguous-full.json > /work/results/design-validation/protected-q-contiguous-full.log 2>&1
