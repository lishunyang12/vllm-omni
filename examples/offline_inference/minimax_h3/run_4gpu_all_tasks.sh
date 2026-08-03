#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

# One-command MiniMax-H3 coverage for the four task paths currently supported
# by vLLM-Omni. The two checkpoint partitions run in separate Python processes.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$(dirname -- "${REPO_ROOT}")}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/MiniMax-H3}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/results/minimax-h3-all-tasks-$(date -u +%Y%m%dT%H%M%SZ)}"
RUNNER="${SCRIPT_DIR}/all_tasks_4gpu.py"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1344}"
DURATION_SECONDS="${DURATION_SECONDS:-5.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SEED_BASE="${SEED_BASE:-1101}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"
MAX_PREFLIGHT_MEMORY_MIB="${MAX_PREFLIGHT_MEMORY_MIB:-2048}"
MAX_PREFLIGHT_GPU_UTIL="${MAX_PREFLIGHT_GPU_UTIL:-10}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-70000}"

if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "${WORK_ROOT}/bin/python" ]]; then
  PYTHON="${WORK_ROOT}/bin/python"
elif [[ -x "${WORK_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${WORK_ROOT}/.venv/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON="$(command -v python || command -v python3 || true)"
fi

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "No Python interpreter found. Set PYTHON=/path/to/venv/bin/python." >&2
  exit 1
fi

for required_file in \
  "${RUNNER}" \
  "${MODEL_ROOT}/FL2VA/model_index.json" \
  "${MODEL_ROOT}/Ref2VA/model_index.json"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file is missing: ${required_file}" >&2
    echo "Both FL2VA and Ref2VA checkpoints are required for the all-task run." >&2
    exit 1
  fi
done

for command_name in nvidia-smi ffmpeg ffprobe; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "Required command is missing: ${command_name}" >&2
    exit 1
  fi
done

IFS=',' read -r -a SELECTED_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#SELECTED_GPUS[@]}" -ne 4 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly four physical GPU indices." >&2
  exit 1
fi

GPU_STATE="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits)"
echo "Selected physical GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "${GPU_STATE}"
for gpu_index in "${SELECTED_GPUS[@]}"; do
  used_mib="$(awk -F',' -v wanted="${gpu_index}" '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
      if ($1 == wanted) print $4
    }
  ' <<< "${GPU_STATE}")"
  total_mib="$(awk -F',' -v wanted="${gpu_index}" '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3)
      if ($1 == wanted) print $3
    }
  ' <<< "${GPU_STATE}")"
  gpu_util="$(awk -F',' -v wanted="${gpu_index}" '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5)
      if ($1 == wanted) print $5
    }
  ' <<< "${GPU_STATE}")"
  if [[ -z "${used_mib}" ]]; then
    echo "GPU ${gpu_index} was not found by nvidia-smi." >&2
    exit 1
  fi
  if (( total_mib < MIN_GPU_MEMORY_MIB )); then
    echo "GPU ${gpu_index} has ${total_mib} MiB; this resident recipe requires" >&2
    echo "the 72 GB RTX PRO 5000 variant or a larger GPU." >&2
    exit 1
  fi
  if (( used_mib > MAX_PREFLIGHT_MEMORY_MIB )); then
    echo "GPU ${gpu_index} already uses ${used_mib} MiB; refusing to interfere." >&2
    exit 1
  fi
  if (( gpu_util > MAX_PREFLIGHT_GPU_UTIL )); then
    echo "GPU ${gpu_index} is ${gpu_util}% busy; refusing to interfere." >&2
    exit 1
  fi
done

mkdir -p \
  "${OUTPUT_DIR}" \
  "${WORK_ROOT}/hf-cache" \
  "${WORK_ROOT}/torchinductor-cache" \
  "${WORK_ROOT}/triton-cache" \
  "${WORK_ROOT}/xdg-cache/torch/kernels"

if [[ "${INSTALL_EDITABLE}" == "1" ]]; then
  "${PYTHON}" -m pip install --no-deps -e "${REPO_ROOT}"
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${WORK_ROOT}/hf-cache"
export TORCHINDUCTOR_CACHE_DIR="${WORK_ROOT}/torchinductor-cache"
export TRITON_CACHE_DIR="${WORK_ROOT}/triton-cache"
export XDG_CACHE_HOME="${WORK_ROOT}/xdg-cache"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export VLLM_OMNI_USE_QUACK_FP8=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

COMMON_ARGS=(
  --model-root "${MODEL_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --duration "${DURATION_SECONDS}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --seed-base "${SEED_BASE}"
)
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  COMMON_ARGS+=(--enforce-eager)
fi

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
  --format=csv \
  -lms 500 > "${OUTPUT_DIR}/nvidia-smi.csv" &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[1/2] Loading FL2VA and running T2VA + first-frame FL2VA"
"${PYTHON}" "${RUNNER}" --partition fl2va "${COMMON_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/run.log"

echo "[2/2] Loading Ref2VA and running image+audio + two-video Ref2VA"
"${PYTHON}" "${RUNNER}" --partition ref2va "${COMMON_ARGS[@]}" \
  2>&1 | tee -a "${OUTPUT_DIR}/run.log"

cleanup
trap - EXIT INT TERM

MEDIA_FILES=(
  "${OUTPUT_DIR}/01_t2va.mp4"
  "${OUTPUT_DIR}/02_fl2va_first_frame.mp4"
  "${OUTPUT_DIR}/03_ref2va_image_audio.mp4"
  "${OUTPUT_DIR}/04_ref2va_two_videos.mp4"
)
for media_path in "${MEDIA_FILES[@]}"; do
  if [[ ! -s "${media_path}" ]]; then
    echo "Expected output is missing or empty: ${media_path}" >&2
    exit 1
  fi
  ffprobe -v error \
    -show_entries stream=index,codec_name,codec_type,width,height,r_frame_rate,sample_rate,channels,duration \
    -show_entries format=duration,size \
    -of json \
    "${media_path}" > "${media_path%.mp4}.ffprobe.json"
done
sha256sum "${MEDIA_FILES[@]}" > "${OUTPUT_DIR}/artifact_sha256.txt"

awk -F',' -v selected="${CUDA_VISIBLE_DEVICES}" '
  BEGIN {
    count=split(selected, selected_gpu, ",")
    for (i=1; i<=count; i++) wanted[selected_gpu[i]]=1
  }
  NR > 1 {
    gpu=$2
    memory=$3
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", gpu)
    gsub(/[^0-9.]/, "", memory)
    if (wanted[gpu] && memory + 0 > peak[gpu] + 0) peak[gpu]=memory + 0
  }
  END {
    print "physical_gpu,peak_memory_mib"
    for (i=1; i<=count; i++) {
      gpu=selected_gpu[i]
      print gpu "," peak[gpu]
    }
  }
' "${OUTPUT_DIR}/nvidia-smi.csv" > "${OUTPUT_DIR}/gpu_peak_memory.csv"

echo "Completed all four MiniMax-H3 task paths."
echo "Outputs: ${OUTPUT_DIR}"
echo "Summary: ${OUTPUT_DIR}/summary.json"
echo "GPU peaks: ${OUTPUT_DIR}/gpu_peak_memory.csv"
