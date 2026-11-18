#!/bin/bash
set -Eeuo pipefail
label=$1
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
mkdir -p "/work/results/$label"
python3 -c 'import os,torch; print("GPU_MAPPING",os.environ.get("CUDA_VISIBLE_DEVICES"),torch.cuda.get_device_properties(0))' > "/work/results/$label/device.txt"
for layer in 00 25; do
  python3 /work/scripts/ablation.py --input "/work/captures/qkv-layer-$layer-rank-0.pt" --output "/work/results/$label/layer-$layer.json" > "/work/results/$label/layer-$layer.log" 2>&1
done
touch "/work/results/$label/complete"
