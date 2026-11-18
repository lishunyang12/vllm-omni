#!/bin/bash
set -Eeuo pipefail
work=${1:-/work}
label=${2:-750mhz}
export PYTHONPATH="$work/source/sana/techniques/sparse_backends:$work/study"
export XDG_CACHE_HOME="$work/xdg-cache" TRITON_CACHE_DIR="$work/triton-cache" OMP_NUM_THREADS=4
mkdir -p "$work/study/results/$label"
python3 "$work/study/replay.py" --validate --output "$work/study/compatibility-trace.json"
for layer in 00 12 25 49; do
  python3 "$work/study/replay.py" --input "$work/results/qkv-sm100-dense/qkv-layer-$layer-rank-0.pt" \
    --output "$work/study/results/$label/layer-$layer.json" --heads 28 --trace \
    > "$work/study/results/$label/layer-$layer.log" 2>&1
done
# Independent-process repeat of layer 0; diagnostics are not timed.
python3 "$work/study/replay.py" --input "$work/results/qkv-sm100-dense/qkv-layer-00-rank-0.pt" \
  --output "$work/study/results/$label/layer-00-repeat.json" --heads 28 \
  > "$work/study/results/$label/layer-00-repeat.log" 2>&1
touch "$work/study/results/$label/complete"
