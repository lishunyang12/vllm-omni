#!/bin/bash
# One-shot JoyVL demo: model -> interaction orchestrator -> JD webui.
#
# The model is the only fixed component; every external module is pluggable and
# configured here via env (unset = disabled), matching JoyVL's design:
#
#   TTS_URL                       voice out  (point at our tts_bridge, or any OpenAI-Realtime-style TTS WS)
#   ASR_URL                       voice in   (point at our asr_bridge, or any compatible ASR WS)
#   BACKGROUND_CODEX_API_URL      delegation background brain (webui agent)
#   ENABLE_MEMORY (default 1)     3-tier memory; SUMMARIZER_BACKEND_URL/MODEL pick the summarizer (default: reuse main)
#
# Usage:  bash start_all.sh
#         TTS_URL=ws://host/v1/tts ASR_URL=ws://host/v1/asr GPU=0 bash start_all.sh
set -o pipefail

VENV="${VENV:-/home/zjy/code/zjy-vllm-omni/.venv}"
WEBUI_DIR="${WEBUI_DIR:-/tmp/je/joyvl-interaction/joyvl-interaction-webui}"
MODEL="${MODEL:-ydydy/JoyAI-VL-Interaction-Preview}"
GPU="${GPU:-0}"
MODEL_PORT="${MODEL_PORT:-8061}"
ORCH_PORT="${ORCH_PORT:-8070}"
WEBUI_PORT="${WEBUI_PORT:-8999}"
WEBUI_USERNAME="${WEBUI_USERNAME:-vllm_omni_maintainer}"
WEBUI_PASSWORD="${WEBUI_PASSWORD:-JoyVL@day0}"
ENABLE_MEMORY="${ENABLE_MEMORY:-1}"
TTS_URL="${TTS_URL:-}"
ASR_URL="${ASR_URL:-}"
BACKGROUND_CODEX_API_URL="${BACKGROUND_CODEX_API_URL:-}"

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG=/tmp/joyvl_demo
mkdir -p "$LOG"

kill_port() {
  local p
  p=$(ss -tlnp 2>/dev/null | grep ":$1 " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  [ -n "${p:-}" ] && kill "$p" 2>/dev/null
  return 0
}
wait_http() {
  for _ in $(seq 1 150); do
    curl -sk -m2 "$1" -o /dev/null 2>/dev/null && return 0
    sleep 2
  done
  echo "  ! timeout waiting for $1"
  return 1
}

echo "Resetting ports ${MODEL_PORT} ${ORCH_PORT} ${WEBUI_PORT}…"
for p in "$MODEL_PORT" "$ORCH_PORT" "$WEBUI_PORT"; do kill_port "$p"; done
sleep 3

source "$VENV/bin/activate"

# detach into a new session so servers outlive this launcher
spawn() { local log="$1"; shift; setsid "$@" > "$log" 2>&1 < /dev/null & }

echo "[1/3] model on GPU ${GPU}…"
spawn "$LOG/model.log" env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
  vllm serve "$MODEL" --served-model-name JoyAI-VL-Interaction-Preview --port "$MODEL_PORT" \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":32,"video":1}'
wait_http "http://127.0.0.1:${MODEL_PORT}/health" && echo "  up :${MODEL_PORT}"

echo "[2/3] orchestrator (memory=${ENABLE_MEMORY})…"
ORCH_ARGS=(--port "$ORCH_PORT" --main-backend-url "http://127.0.0.1:${MODEL_PORT}/v1"
           --main-model JoyAI-VL-Interaction-Preview --no-delegation)
[ "$ENABLE_MEMORY" = "0" ] && ORCH_ARGS+=(--no-memory)
[ -n "${SUMMARIZER_BACKEND_URL:-}" ] && ORCH_ARGS+=(--summarizer-backend-url "$SUMMARIZER_BACKEND_URL")
[ -n "${SUMMARIZER_MODEL:-}" ] && ORCH_ARGS+=(--summarizer-model "$SUMMARIZER_MODEL")
spawn "$LOG/orch.log" env PYTHONPATH="$CODE" python -m vllm_omni.interaction.server "${ORCH_ARGS[@]}"
wait_http "http://127.0.0.1:${ORCH_PORT}/health" && echo "  up :${ORCH_PORT}"

echo "[3/3] JD webui…"
spawn "$LOG/webui.log" env WEBUI_USERNAME="$WEBUI_USERNAME" WEBUI_PASSWORD="$WEBUI_PASSWORD" \
  TTS_URL="$TTS_URL" ASR_URL="$ASR_URL" BACKGROUND_CODEX_API_URL="$BACKGROUND_CODEX_API_URL" \
  bash -c "cd '$WEBUI_DIR' && exec bash scripts/start_server.sh --api-base 'http://127.0.0.1:${ORCH_PORT}/v1'"
for _ in $(seq 1 60); do grep -q "Server startup complete" "$LOG/webui.log" 2>/dev/null && break; sleep 1; done

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
echo "=== JoyVL demo ready ==="
echo "  webui  : https://${IP}:${WEBUI_PORT}   (user ${WEBUI_USERNAME} / pass ${WEBUI_PASSWORD})"
echo "  memory : $([ "$ENABLE_MEMORY" = 0 ] && echo off || echo on)"
echo "  TTS    : ${TTS_URL:-<unset>}"
echo "  ASR    : ${ASR_URL:-<unset>}"
echo "  agent  : ${BACKGROUND_CODEX_API_URL:-<unset>}"
echo "  logs   : ${LOG}/{model,orch,webui}.log"
