#!/bin/bash
set -Eeuo pipefail
label=${1:?}
export PYTHONPATH=/work/source/candidate/techniques/sparse_backends:/work/source/candidate/techniques/sparse_backends/benchmarks
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
root=/work/results/$label
mkdir -p "$root"
ncu=/work/tools/nsight_compute-linux-x86_64-2026.4.0.10-archive/ncu
"$ncu" --version > "$root/version.txt"
for variant in baseline candidate; do
  "$ncu" --replay-mode application --nvtx --nvtx-include 'H3_SOL_PROFILE/' --clock-control none \
    --kernel-name 'regex:_sol_attn_sm100_bf16_kernel' --kernel-name-base demangled --launch-count 1 \
    --section LaunchStats --section Occupancy \
    --force-overwrite -o "$root/$variant" python3 /work/scripts/profile_candidate.py \
    --variant "$variant" --output "$root/$variant-meta.json" > "$root/$variant.log" 2>&1
  report="$root/$variant.ncu-repz"
  if [[ ! -f $report ]]; then report="$root/$variant.ncu-rep"; fi
  "$ncu" --import "$report" --csv --page raw > "$root/$variant.csv"
  "$ncu" --import "$report" --page details > "$root/$variant-details.txt"
done
touch "$root/complete"
