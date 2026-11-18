#!/bin/bash
set -Eeuo pipefail
export PYTHONPATH=/work/source/candidate/techniques/sparse_backends
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
mkdir -p /work/results
cd /work/source/candidate
python3 -m pytest -q -ra techniques/sparse_backends/tests \
  --junitxml=/work/results/tests-final.xml > /work/results/tests-final.log 2>&1
tail -n 9 /work/results/tests-final.log
python3 techniques/sparse_backends/benchmarks/benchmark_sm100.py \
  --baseline-root /work/source/baseline/techniques/sparse_backends --stress-only \
  --output /work/results/baseline-stress-final.json > /work/results/baseline-stress-final.log 2>&1
cat /work/results/baseline-stress-final.log
sha256sum techniques/sparse_backends/sol_attn/sm100/{softmax,mainloop}.py \
  techniques/sparse_backends/tests/{test_exact_block_mask,test_sm100_rescale_mask}.py \
  techniques/sparse_backends/benchmarks/benchmark_sm100.py > /work/results/final-source-sha256.txt
printf '0\n' > /work/results/tests-final-exit-code.txt
