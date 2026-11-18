#!/bin/bash
set -Eeuo pipefail
variant=${1:?}
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
export PYTHONPATH="/work/source/$variant/techniques/sparse_backends"
root=/work/results/design-validation
mkdir -p "$root"
run_stage() {
  local name=$1 limit=$2
  shift 2
  set +e
  timeout -k 10 "$limit" "$@" > "$root/$variant-$name.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$root/$variant-$name.exit"
  return "$rc"
}
run_stage stress 240 python3 "/work/source/$variant/techniques/sparse_backends/benchmarks/benchmark_sm100.py" --baseline-root /work/source/candidate/techniques/sparse_backends --stress-only --output "$root/$variant-stress.json"
run_stage full 900 python3 /work/scripts/validate_design.py --baseline /work/source/candidate --candidate "/work/source/$variant" --report "$root/$variant-full.json"
python3 /work/scripts/require_validation.py /work "$variant"
