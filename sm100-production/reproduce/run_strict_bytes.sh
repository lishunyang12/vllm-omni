#!/bin/bash
set -Eeuo pipefail
export PYTHONPATH=/work/source/candidate/techniques/sparse_backends:/work/source/candidate/techniques/sparse_backends/benchmarks
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
python3 /work/scripts/strict_byte_validation.py > /work/results/strict-byte-validation.log 2>&1
python3 /work/source/candidate/techniques/sparse_backends/benchmarks/benchmark_sm100.py \
  --baseline-root /work/source/baseline/techniques/sparse_backends --stress-only \
  --output /work/results/strict-byte-stress.json > /work/results/strict-byte-stress.log 2>&1
tail -n 2 /work/results/strict-byte-validation.log
cat /work/results/strict-byte-stress.log
