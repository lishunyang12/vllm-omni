#!/usr/bin/env bash
# Run every MiniMax-H3 task supported by vLLM-Omni on two 24/32 GB GPUs.
set -Eeuo pipefail

RUN_ROOT=${RUN_ROOT:-"$(pwd)"}
MODEL_ROOT=${MODEL_ROOT:-"${RUN_ROOT}/MiniMax-H3"}
OUTPUT_DIR=${OUTPUT_DIR:-"${RUN_ROOT}/results/minimax-h3-2gpu-$(date -u +%Y%m%dT%H%M%SZ)"}
GPU_IDS=${GPU_IDS:-0,1}
PROFILE=${PROFILE:-auto}
PORT=${PORT:-8091}
ASSET_PORT=${ASSET_PORT:-8092}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
DURATION_SECONDS=${DURATION_SECONDS:-5.0}
SERVER_START_TIMEOUT_SECONDS=${SERVER_START_TIMEOUT_SECONDS:-1800}
DLO_RESIDENT_LAYERS=${DLO_RESIDENT_LAYERS:-auto}

SERVER_PID=""
ASSET_SERVER_PID=""
MONITOR_PID=""

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

stop_pid() {
  local pid=${1:-}
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  stop_pid "${SERVER_PID}"
  stop_pid "${ASSET_SERVER_PID}"
  stop_pid "${MONITOR_PID}"
}
trap cleanup EXIT INT TERM

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 1
  }
}

for command_name in vllm curl ffmpeg ffprobe nvidia-smi python; do
  require_command "${command_name}"
done

IFS=',' read -r -a gpu_id_array <<<"${GPU_IDS}"
if [[ ${#gpu_id_array[@]} -ne 2 ]]; then
  echo "GPU_IDS must contain exactly two comma-separated GPU indexes" >&2
  exit 1
fi

for partition in FL2VA Ref2VA; do
  test -f "${MODEL_ROOT}/${partition}/model_index.json" || {
    echo "Missing ${MODEL_ROOT}/${partition}/model_index.json" >&2
    exit 1
  }
done

mkdir -p "${OUTPUT_DIR}/assets" "${OUTPUT_DIR}/logs"

min_vram_mib=999999
for gpu_id in "${gpu_id_array[@]}"; do
  active_pids=$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')
  if [[ -n "${active_pids}" && ${ALLOW_BUSY_GPUS:-0} != 1 ]]; then
    echo "GPU ${gpu_id} is busy with process(es): ${active_pids//$'\n'/, }" >&2
    echo "Choose two idle GPUs or set ALLOW_BUSY_GPUS=1 at your own risk." >&2
    exit 1
  fi
  gpu_vram_mib=$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
  if (( gpu_vram_mib < min_vram_mib )); then
    min_vram_mib=${gpu_vram_mib}
  fi
done

if [[ "${PROFILE}" == auto ]]; then
  if (( min_vram_mib >= 30000 )); then
    PROFILE=rtx5090
  else
    PROFILE=rtx4090
  fi
fi

case "${PROFILE}" in
  rtx5090)
    WIDTH=${WIDTH:-1344}
    HEIGHT=${HEIGHT:-768}
    if [[ "${DLO_RESIDENT_LAYERS}" == auto ]]; then
      DLO_RESIDENT_LAYERS=20
    fi
    ;;
  rtx4090)
    WIDTH=${WIDTH:-1024}
    HEIGHT=${HEIGHT:-576}
    if [[ "${DLO_RESIDENT_LAYERS}" == auto ]]; then
      DLO_RESIDENT_LAYERS=12
    fi
    ;;
  *)
    echo "PROFILE must be auto, rtx5090, or rtx4090" >&2
    exit 1
    ;;
esac

available_ram_gib=$(awk '/MemAvailable:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)
if (( available_ram_gib < 200 )) && [[ ${ALLOW_LOW_HOST_RAM:-0} != 1 ]]; then
  echo "Only ${available_ram_gib} GiB host RAM is available; at least 200 GiB is required." >&2
  echo "Use a 384 GiB-class host, or set ALLOW_LOW_HOST_RAM=1 at your own risk." >&2
  exit 1
fi

[[ "${DLO_RESIDENT_LAYERS}" =~ ^[0-9]+$ ]] || {
  echo "DLO_RESIDENT_LAYERS must be a non-negative integer or auto" >&2
  exit 1
}

log "profile=${PROFILE}, shape=${WIDTH}x${HEIGHT}, GPUs=${GPU_IDS}, steps=${NUM_INFERENCE_STEPS}, resident_layers=${DLO_RESIDENT_LAYERS}"
log "output=${OUTPUT_DIR}"

(
  while true; do
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    nvidia-smi -i "${GPU_IDS}" \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits | \
      awk -v ts="${timestamp}" -F ', ' '{print ts "," $1 "," $2 "," $3}'
    sleep 1
  done
) >"${OUTPUT_DIR}/gpu-memory.csv" &
MONITOR_PID=$!

wait_for_server() {
  local deadline=$((SECONDS + SERVER_START_TIMEOUT_SECONDS))
  until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "vLLM server exited during startup" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for vLLM server" >&2
      return 1
    fi
    sleep 5
  done
}

start_server() {
  local partition=$1
  local model=${MODEL_ROOT}/${partition}
  local server_log=${OUTPUT_DIR}/logs/${partition,,}-server.log

  stop_pid "${SERVER_PID}"
  SERVER_PID=""
  log "starting ${partition} server; runtime logs remain visible in this terminal"

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
  PYTHONUNBUFFERED=1 \
  vllm serve "${model}" \
    --omni \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --trust-remote-code \
    --num-gpus 2 \
    --tensor-parallel-size 2 \
    --usp 1 \
    --ring 1 \
    --text-encoder-tp-size 2 \
    --vae-patch-parallel-size 2 \
    --vae-parallel-mode tile \
    --vae-use-tiling \
    --enable-distributed-layerwise-offload \
    --dlo-no-use-allgather \
    --dlo-resident-layers "${DLO_RESIDENT_LAYERS}" \
    --enforce-eager \
    --diffusion-attention-backend CUDNN_ATTN \
    > >(tee "${server_log}") 2>&1 &
  SERVER_PID=$!
  wait_for_server
  log "${partition} server ready (pid=${SERVER_PID})"
}

validate_mp4() {
  local path=$1
  local video_codec audio_codec fps sample_rate channels
  video_codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "${path}")
  audio_codec=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "${path}")
  fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "${path}")
  sample_rate=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "${path}")
  channels=$(ffprobe -v error -select_streams a:0 -show_entries stream=channels -of csv=p=0 "${path}")

  [[ "${video_codec}" == h264 && "${audio_codec}" == aac && "${fps}" == 24/1 \
    && "${sample_rate}" == 32000 && "${channels}" == 2 ]] || {
    echo "Invalid media streams in ${path}: video=${video_codec}, audio=${audio_codec}, fps=${fps}, rate=${sample_rate}, channels=${channels}" >&2
    return 1
  }
  ffmpeg -v error -i "${path}" -map 0:v:0 -map 0:a:0 -f null -
  log "validated ${path}"
}

submit_video() {
  local output=$1
  shift
  local partial=${output}.part
  rm -f "${partial}"
  time curl --fail-with-body -sS -X POST \
    "http://127.0.0.1:${PORT}/v1/videos/sync" \
    "$@" \
    -o "${partial}"
  mv "${partial}" "${output}"
  validate_mp4 "${output}"
}

start_server FL2VA

submit_video "${OUTPUT_DIR}/t2va.mp4" \
  -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
  -F "width=${WIDTH}" \
  -F "height=${HEIGHT}" \
  -F 'fps=24' \
  -F "num_inference_steps=${NUM_INFERENCE_STEPS}" \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F "extra_params={\"task\":\"t2va\",\"duration\":${DURATION_SECONDS},\"audio_flow_shift\":3.0}"

ffmpeg -y -v error -i "${OUTPUT_DIR}/t2va.mp4" \
  -frames:v 1 "${OUTPUT_DIR}/assets/first-frame.png"
ffmpeg -y -v error -i "${OUTPUT_DIR}/t2va.mp4" \
  -vn -ac 2 -ar 32000 "${OUTPUT_DIR}/assets/reference.wav"

submit_video "${OUTPUT_DIR}/fl2va.mp4" \
  -F 'prompt=The supplied frame continues naturally as the cats march across the bedroom playing tiny brass instruments, with synchronized music and room ambience.' \
  -F 'fps=24' \
  -F "num_inference_steps=${NUM_INFERENCE_STEPS}" \
  -F 'flow_shift=12' \
  -F 'seed=2101' \
  -F "extra_params={\"task\":\"fl2va\",\"duration\":${DURATION_SECONDS},\"audio_flow_shift\":3.0}" \
  -F "input_reference=@${OUTPUT_DIR}/assets/first-frame.png;type=image/png"

# Fail before an expensive Ref2VA request if TorchCodec cannot see shared
# FFmpeg libraries.  A static imageio-ffmpeg executable is not sufficient.
REF_AUDIO=${OUTPUT_DIR}/assets/reference.wav python -c \
  'import os, torchaudio; audio, rate = torchaudio.load(os.environ["REF_AUDIO"]); assert audio.numel() and rate == 32000'

python -m http.server "${ASSET_PORT}" \
  --bind 127.0.0.1 \
  --directory "${OUTPUT_DIR}/assets" \
  > >(tee "${OUTPUT_DIR}/logs/reference-http-server.log") 2>&1 &
ASSET_SERVER_PID=$!
until curl -fsI "http://127.0.0.1:${ASSET_PORT}/reference.wav" >/dev/null; do
  sleep 1
done

start_server Ref2VA

submit_video "${OUTPUT_DIR}/ref2va-image-audio.mp4" \
  -F 'prompt=Use Picture 1 as the visual subject and Audio 1 as the sound reference, with coherent natural motion.' \
  -F "width=${WIDTH}" \
  -F "height=${HEIGHT}" \
  -F 'fps=24' \
  -F "num_inference_steps=${NUM_INFERENCE_STEPS}" \
  -F 'flow_shift=12' \
  -F 'seed=3101' \
  -F "extra_params={\"task\":\"ref2va\",\"duration\":${DURATION_SECONDS},\"audio_flow_shift\":3.0}" \
  -F "input_reference=@${OUTPUT_DIR}/assets/first-frame.png;type=image/png" \
  -F "audio_reference={\"audio_url\":\"http://127.0.0.1:${ASSET_PORT}/reference.wav\"}"

submit_video "${OUTPUT_DIR}/ref2va-two-video.mp4" \
  -F 'prompt=Combine Video 1 and Video 2 into one coherent scene, preserving the subject motion and synchronized sound.' \
  -F "width=${WIDTH}" \
  -F "height=${HEIGHT}" \
  -F 'fps=24' \
  -F "num_inference_steps=${NUM_INFERENCE_STEPS}" \
  -F 'flow_shift=12' \
  -F 'seed=4101' \
  -F "extra_params={\"task\":\"ref2va\",\"duration\":${DURATION_SECONDS},\"audio_flow_shift\":3.0}" \
  -F "input_references=@${OUTPUT_DIR}/t2va.mp4;type=video/mp4" \
  -F "input_references=@${OUTPUT_DIR}/fl2va.mp4;type=video/mp4"

log "all MiniMax-H3 tasks completed successfully"
ls -lh "${OUTPUT_DIR}"/*.mp4
