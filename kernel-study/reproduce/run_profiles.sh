#!/bin/bash
set -Eeuo pipefail
work=${1:-/work}
label=${2:-750mhz-profile}
for variant in native dense warp; do
  bash "$work/study/run_profile.sh" "$work" "$variant" 00
done
mkdir -p "$work/study/results/$label"
touch "$work/study/results/$label/complete"
