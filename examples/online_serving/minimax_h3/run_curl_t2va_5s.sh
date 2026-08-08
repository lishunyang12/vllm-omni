#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Send and validate the five-second MiniMax-H3 T2VA capacity-test request.

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8091}"
RUN_DIR="${RUN_DIR:-$PWD/h3_single_gpu_fp8_no_offload}"
OUTPUT_PATH="${OUTPUT_PATH:-${RUN_DIR}/t2va.mp4}"
PROMPT="${PROMPT:-In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.}"

for required_command in curl ffprobe; do
  if ! command -v "${required_command}" >/dev/null; then
    echo "${required_command} is required" >&2
    exit 2
  fi
done
if [[ ! -x /usr/bin/time ]]; then
  echo "/usr/bin/time is required to record client latency" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"
curl --fail --silent "${BASE_URL}/health" >/dev/null

/usr/bin/time -f 'client_wall_time_s=%e' -o "${RUN_DIR}/client_time.txt" \
curl --fail-with-body --silent --show-error --max-time 1800 \
  -X POST "${BASE_URL}/v1/videos/sync" \
  --form-string "prompt=${PROMPT}" \
  -F width=1344 \
  -F height=768 \
  -F fps=24 \
  -F num_inference_steps=50 \
  -F flow_shift=12 \
  -F seed=1101 \
  -F aspect_ratio=16:9 \
  -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
  -o "${OUTPUT_PATH}"

ffprobe -v error \
  -show_entries format=duration,size \
  -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels \
  -of json "${OUTPUT_PATH}" | tee "${RUN_DIR}/media.json"

video_codec="$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "${OUTPUT_PATH}")"
audio_codec="$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "${OUTPUT_PATH}")"
audio_rate="$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 "${OUTPUT_PATH}")"
audio_channels="$(ffprobe -v error -select_streams a:0 -show_entries stream=channels -of default=noprint_wrappers=1:nokey=1 "${OUTPUT_PATH}")"

if [[ "${video_codec}" != "h264" ]]; then
  echo "Expected H.264 video, got: ${video_codec:-missing}" >&2
  exit 1
fi
if [[ "${audio_codec}" != "aac" || "${audio_rate}" != "32000" || "${audio_channels}" != "2" ]]; then
  echo "Expected 32 kHz stereo AAC audio; got codec=${audio_codec:-missing}, rate=${audio_rate:-missing}, channels=${audio_channels:-missing}" >&2
  exit 1
fi

{
  printf 'video_codec=%s\n' "${video_codec}"
  printf 'audio_codec=%s\n' "${audio_codec}"
  printf 'audio_rate_hz=%s\n' "${audio_rate}"
  printf 'audio_channels=%s\n' "${audio_channels}"
} | tee "${RUN_DIR}/validation.txt"

echo "Validated output: ${OUTPUT_PATH}"
echo "Stop the server with Ctrl-C to finalize ${RUN_DIR}/memory_summary.txt"
