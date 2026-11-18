#!/bin/bash
set -Eeuo pipefail
export CUTE_DSL_DISABLE_FILE_CACHING=1 XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
export PYTHONPATH=/work/source/protected-q-contiguous-fp32/techniques/sparse_backends
root=/work/results/contiguous-fp32-validation
mkdir -p "$root"
run_check() {
  local name=$1
  shift
  set +e
  timeout -k 10 600 "$@" > "$root/$name.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$root/$name.exit"
  return "$rc"
}
run_check full python3 /work/scripts/validate_design.py \
  --baseline /work/source/candidate-fp32 --candidate /work/source/protected-q-contiguous-fp32 \
  --tests-root /work/source/candidate-fp32/techniques/sparse_backends/tests --report "$root/full.json"
run_check matrix python3 /work/scripts/independent_matrix.py --output "$root/matrix"
python3 /work/scripts/require_contiguous_validation.py /work
