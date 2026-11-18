#!/bin/bash
set -Eeuo pipefail
export OMP_NUM_THREADS=4 XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache
root=/work/results/design-validation
mkdir -p "$root"
validate() {
  local variant=$1 rc=0
  export PYTHONPATH="/work/source/$variant/techniques/sparse_backends"
  if timeout -k 10 240 python3 "/work/source/$variant/techniques/sparse_backends/benchmarks/benchmark_sm100.py" \
    --baseline-root /work/source/candidate/techniques/sparse_backends --stress-only \
    --output "$root/$variant-stress.json" > "$root/$variant-stress.log" 2>&1; then
    printf '0\n' > "$root/$variant-stress.exit"
  else
    rc=$?; printf '%s\n' "$rc" > "$root/$variant-stress.exit"; return "$rc"
  fi
  if timeout -k 10 900 python3 /work/scripts/validate_design.py --baseline /work/source/candidate \
    --candidate "/work/source/$variant" --report "$root/$variant-full.json" > "$root/$variant-full.log" 2>&1; then
    printf '0\n' > "$root/$variant-full.exit"
  else
    rc=$?; printf '%s\n' "$rc" > "$root/$variant-full.exit"; return "$rc"
  fi
}
validate protected-q-contiguous || true
if validate smem-p-serial; then
  validate smem-p-overlap || true
else
  printf 'Skipped: serial SMEM-P correctness gate failed.\n' > "$root/smem-p-overlap-skipped.txt"
fi
touch "$root/next-candidates-validation-complete"
