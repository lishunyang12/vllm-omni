#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# End-to-end attention-backend comparison on LTX-2.0 (T2V + audio) for the
# #3079 PR. LTX-2 has a richer attention topology than the other test models
# (six per-block attention modules: self-video, self-audio, cross-video,
# cross-audio, audio->video, video->audio), and it stresses the audio
# attention path that triggers cuDNN's symbolic-head_dim fallback.
#
# Uses 2-GPU CFG parallel (one CFG branch per GPU) — needs both GPUs free.
# Each backend run takes ~5 minutes; expect ~15-20 minutes total.
#
# Usage (from repo root):
#   bash benchmarks/diffusion/bench_e2e_ltx2.sh
#   STEPS=40 bash benchmarks/diffusion/bench_e2e_ltx2.sh   # quicker variant

set -euo pipefail

MODEL="${MODEL:-Lightricks/LTX-2}"
PROMPT="${PROMPT:-A warm morning kitchen. The camera opens on a tight close-up of a woman and a man in their 30s standing across from each other at the counter, mugs in hand. The woman says low and serious: \"We said one. Just one.\" The man exhales, glancing guiltily at the floor, and mutters, \"It got lonely.\" The camera slowly pans right to reveal every surface of the kitchen covered in identical potted houseplants. He is still holding one more behind his back. Tone: deadpan, affectionate, and quietly tragic.}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
FRAMES="${FRAMES:-97}"
STEPS="${STEPS:-500}"
GUIDANCE="${GUIDANCE:-4.0}"
FRAME_RATE="${FRAME_RATE:-48}"
FPS="${FPS:-48}"
SEED="${SEED:-42}"
CFG_PARALLEL_SIZE="${CFG_PARALLEL_SIZE:-2}"

BACKENDS=(
  TORCH_SDPA
  CUDNN_ATTN
  FLASHINFER_ATTN
)

OUT_DIR="${OUT_DIR:-bench_ltx2_out}"
mkdir -p "$OUT_DIR"

declare -A TOTALS
declare -A PER_STEP
declare -A PEAK_MEM

for BACKEND in "${BACKENDS[@]}"; do
  LOG="$OUT_DIR/ltx2_${BACKEND}.log"
  VID="$OUT_DIR/ltx2_${BACKEND}.mp4"
  echo "======================================================================"
  echo "Running $BACKEND (CFG parallel ${CFG_PARALLEL_SIZE}) ..."
  echo "======================================================================"
  DIFFUSION_ATTENTION_BACKEND="$BACKEND" \
    python examples/offline_inference/text_to_video/text_to_video.py \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --num-frames "$FRAMES" \
      --guidance-scale "$GUIDANCE" \
      --frame-rate "$FRAME_RATE" \
      --num-inference-steps "$STEPS" \
      --fps "$FPS" \
      --seed "$SEED" \
      --cfg-parallel-size "$CFG_PARALLEL_SIZE" \
      --output "$VID" 2>&1 | tee "$LOG"

  TOTAL=$(grep -oE "Total generation time: [0-9.]+" "$LOG" | awk '{print $4}' | tail -1)
  STEP=$(grep -oE "[0-9]+\.[0-9]+s/it" "$LOG" | tail -1 | sed 's/s\/it//')
  if [[ -z "$STEP" ]]; then
    RATE=$(grep -oE "[0-9]+\.[0-9]+it/s" "$LOG" | tail -1 | sed 's/it\/s//')
    if [[ -n "$RATE" ]]; then
      STEP=$(awk -v r="$RATE" 'BEGIN { printf "%.3f", 1.0/r }')
    fi
  fi
  MEM=$(grep -oE "Peak GPU memory.*reserved" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+ GB" | head -1)

  TOTALS[$BACKEND]="${TOTAL:-?}"
  PER_STEP[$BACKEND]="${STEP:-?}"
  PEAK_MEM[$BACKEND]="${MEM:-?}"
done

echo
echo "======================================================================"
echo "LTX-2.0 e2e ranking (${HEIGHT}x${WIDTH}, ${FRAMES} frames, ${STEPS} steps, CFG=${CFG_PARALLEL_SIZE}, seed ${SEED})"
echo "======================================================================"
printf "%-18s %14s %14s %18s\n" "backend" "total (s)" "s/step" "peak VRAM"
printf "%-18s %14s %14s %18s\n" "------------------" "--------------" "--------------" "------------------"
for BACKEND in "${BACKENDS[@]}"; do
  printf "%-18s %14s %14s %18s\n" \
    "$BACKEND" "${TOTALS[$BACKEND]}" "${PER_STEP[$BACKEND]}" "${PEAK_MEM[$BACKEND]}"
done

echo
echo "Note: CUDNN_ATTN logs 'cuDNN SDPA rejected this shape; falling back...' on"
echo "LTX-2's audio-attention path (symbolic head_dim under torch.compile)."
echo "Expected and harmless — the fallback guard catches it."
