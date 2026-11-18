#!/bin/bash
set -Eeuo pipefail
work=$1
container=$2
gpu_count=$3
run_id=$4
indices=$5
clock=${6:-default}
mounts=${7:-$work:/work}
cpu_mode=${8:-auto}
inner_cpus=${9:-12}
[[ $indices =~ ^[0-9]+(,[0-9]+)*$ ]]
[[ $run_id =~ ^[a-zA-Z0-9_-]+$ ]]
[[ $clock == default || $clock =~ ^[0-9]+$ ]]
[[ ${SLURM_STEP_GPUS:?Allocation GPU indices required} == "$indices" ]]
evidence="$work/evidence/$run_id-control"
mkdir -p "$evidence"
changed=0
observer_pid=
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [[ -n $observer_pid ]]; then
    kill "$observer_pid" 2>/dev/null || true
    wait "$observer_pid" 2>/dev/null || true
  fi
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
printf 'physical_gpu_indices=%s\nrequested_clock_mhz=%s\ncpu_offload=%s\n' "$indices" "$clock" "$cpu_mode" > "$evidence/condition.txt"
if [[ $clock != default ]]; then
  # Only the allocated GPUs; site-delegated nvidia-smi permission, no limit increase.
  changed=1
  sudo -n /usr/bin/nvidia-smi -i "$indices" -lgc "$clock,$clock" > "$evidence/set-clock.log" 2>&1
fi
python3 "$work/scripts/gpu_telemetry.py" --indices "$indices" \
  --stop-file "$work/results/ref2va-native-$run_id/summary.json" \
  --output "$evidence/telemetry.csv" --seconds 7200 &
observer_pid=$!
export ENROOT_CACHE_PATH=/var/lib/enroot-cache/group-$(id -g)
export ENROOT_DATA_PATH=/tmp/enroot-data/user-$(id -u)
env -u SLURM_CPUS_PER_TASK srun --jobid="${SLURM_JOB_ID:?}" --overlap --ntasks=1 --cpus-per-task="$inner_cpus" \
  --cpu-bind=none --immediate=10 --container-name="$container" \
  --container-mounts="$mounts" --container-remap-root \
  env H3_GPU_COUNT="$gpu_count" H3_RUN_ID="$run_id" H3_CPU_OFFLOAD="$cpu_mode" \
  H3_GPU_FREQ_REQUEST_MHZ="$clock" H3_PHYSICAL_GPU_INDICES="$indices" \
  bash /work/scripts/run_contiguous_forward.sh /work
