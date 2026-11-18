#!/bin/bash
set -Eeuo pipefail
work=${1:-/work}
label=${2:-mask-750}
export PYTHONPATH="$work/source/sana/techniques/sparse_backends:$work/study"
export XDG_CACHE_HOME="$work/xdg-cache" TRITON_CACHE_DIR="$work/triton-cache" OMP_NUM_THREADS=4
root="$work/study/results/$label"
mkdir -p "$root"
python3 "$work/study/experiment.py" --validate --candidates fullpair,skip_identity_wide32_fullpair \
  --output "$root/validation.json" > "$root/validation.log" 2>&1
for layer in 00 25; do
  python3 "$work/study/experiment.py" --candidates skip_identity_wide32,fullpair,skip_identity_wide32_fullpair \
    --input "$work/results/qkv-sm100-dense/qkv-layer-$layer-rank-0.pt" \
    --output "$root/layer-$layer.json" > "$root/layer-$layer.log" 2>&1
done
python3 "$work/study/experiment.py" --candidates skip_identity_wide32,fullpair,skip_identity_wide32_fullpair \
  --input "$work/results/qkv-sm100-dense/qkv-layer-00-rank-0.pt" \
  --output "$root/layer-00-repeat.json" > "$root/layer-00-repeat.log" 2>&1
touch "$root/complete"
