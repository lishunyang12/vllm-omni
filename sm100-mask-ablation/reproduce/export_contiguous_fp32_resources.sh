#!/usr/bin/env bash
set -euo pipefail
baseline=${1:?Usage: $0 BASELINE_ROOT CANDIDATE_ROOT NEW_OUTPUT_DIR [INPUT]}
candidate=${2:?}
output=${3:?}
input=${4:-/work/results/ref2va-native-sequence-450/qkv-step-10-layer-00-rank-0.pt}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -- "$output"
for variant in baseline candidate; do
  source_root=$baseline
  if [[ $variant == candidate ]]; then source_root=$candidate; fi
  mkdir -- "$output/$variant"
  PYTHONPATH="$source_root/techniques/sparse_backends" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$output/$variant/pycache" \
    CUTE_DSL_KEEP=cubin,ptx CUTE_DSL_DUMP_DIR="$output/$variant" CUTE_DSL_DISABLE_FILE_CACHING=1 \
    "${PYTHON:-python3}" "$script_dir/resource_probe_fp32.py" \
      --input "$input" --source-root "$source_root" --output "$output/$variant/meta.json" \
      > "$output/$variant/build.log" 2>&1
done
"${PYTHON:-python3}" "$script_dir/collect_contiguous_resources.py" \
  --root "$output" --cuobjdump "${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
