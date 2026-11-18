#!/bin/bash
set -Eeuo pipefail
work=$1 container=$2 indices=$3 mhz=$4 label=$5
[[ $indices =~ ^[0-9]+(,[0-9]+)*$ && $mhz =~ ^[0-9]+$ && $label =~ ^[a-zA-Z0-9_-]+$ ]]
[[ ${SLURM_STEP_GPUS:?} == "$indices" ]]
evidence="$work/results/$label-control"
mkdir -p "$evidence"
changed=0 observer=
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [[ -n $observer ]]; then kill "$observer" 2>/dev/null || true; wait "$observer" 2>/dev/null || true; fi
  if [[ $changed == 1 ]]; then
    sudo -n /usr/bin/nvidia-smi -i "$indices" -rgc > "$evidence/restore.log" 2>&1 || rc=90
  fi
  nvidia-smi -i "$indices" -q -d CLOCK,POWER > "$evidence/after.txt" 2>&1 || rc=91
  printf '%s\n' "$rc" > "$evidence/exit-code.txt"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
nvidia-smi -i "$indices" -q -d CLOCK,POWER > "$evidence/before.txt"
changed=1
sudo -n /usr/bin/nvidia-smi -i "$indices" -lgc "$mhz,$mhz" > "$evidence/set-clock.log" 2>&1
python3 "$work/scripts/gpu_telemetry.py" --indices "$indices" \
  --stop-file "$work/results/$label/complete" --output "$evidence/telemetry.csv" --seconds 4200 &
observer=$!
export ENROOT_CACHE_PATH=/var/lib/enroot-cache/group-$(id -g)
export ENROOT_DATA_PATH=/tmp/enroot-data/user-$(id -u)
env -u SLURM_CPUS_PER_TASK srun --jobid="${SLURM_JOB_ID:?}" --overlap --ntasks=1 --cpus-per-task=8 \
  --cpu-bind=none --immediate=10 --container-name="$container" --container-remap-root \
  --container-mounts="$work:/work" bash /work/scripts/run_layer25_repeat.sh "$label"
