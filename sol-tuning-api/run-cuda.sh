#!/usr/bin/env bash
set -Eeuo pipefail
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export XDG_CACHE_HOME=/work/xdg-cache TORCHINDUCTOR_CACHE_DIR=/work/torchinductor-cache TRITON_CACHE_DIR=/work/triton-cache
mkdir -p /work/sol-kernel
tar xzf /evidence/sol-kernel.tar.gz -C /work/sol-kernel
uv pip install --system --no-deps --no-build-isolation /work/sol-kernel
cp /evidence/test_sol_attn_cuda.py /work/source/tests/diffusion/attention/test_sol_attn_cuda.py
cd /work/source
set +e
python3 -m pytest -q --run-level=core_model tests/diffusion/attention/test_sol_attn_cuda.py -k 'layer_tuning or h3_simple_selection' --junitxml=/evidence/cuda.xml > /evidence/cuda.log 2>&1
result=$?
echo "$result" > /evidence/cuda.exit
tail -n 28 /evidence/cuda.log
exit "$result"
