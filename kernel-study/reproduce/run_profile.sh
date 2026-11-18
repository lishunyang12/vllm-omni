#!/bin/bash
set -Eeuo pipefail
work=${1:-/work}
variant=${2:?dense/native/warp}
layer=${3:-00}
[[ $variant =~ ^(dense|native|warp)$ && $layer =~ ^[0-9][0-9]$ ]]
export PYTHONPATH="$work/source/sana/techniques/sparse_backends:$work/study"
export XDG_CACHE_HOME="$work/xdg-cache" TRITON_CACHE_DIR="$work/triton-cache" OMP_NUM_THREADS=4
ncu="$work/study/tools/nsight_compute-linux-sbsa-2026.4.0.10-archive/ncu"
root="$work/study/ncu"
mkdir -p "$root"
case "$variant" in
  native) kernel='regex:_sol_attn_sm100_bf16_kernel' ;;
  warp) kernel='regex:.*kernel.*' ;;
  dense) kernel='regex:.*(fmha|flash|cudnn).*' ;;
esac
"$ncu" --nvtx --nvtx-include 'H3_SOL_PROFILE/' --clock-control none \
  --kernel-name "$kernel" --kernel-name-base demangled --launch-count 1 \
  --section SpeedOfLight --section ComputeWorkloadAnalysis \
  --section MemoryWorkloadAnalysis --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  --force-overwrite -o "$root/$variant-layer-$layer" \
  python3 "$work/study/replay.py" --input "$work/results/qkv-sm100-dense/qkv-layer-$layer-rank-0.pt" \
  --heads 28 --profile-only "$variant" --output "$root/$variant-layer-$layer-meta.json" \
  > "$root/$variant-layer-$layer.log" 2>&1
report="$root/$variant-layer-$layer.ncu-repz"
if [[ ! -f $report ]]; then report="$root/$variant-layer-$layer.ncu-rep"; fi
"$ncu" --import "$report" --csv --page raw > "$root/$variant-layer-$layer.csv"
"$ncu" --import "$report" --page details > "$root/$variant-layer-$layer-details.txt"
