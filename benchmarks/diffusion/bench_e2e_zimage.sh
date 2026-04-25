#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# End-to-end attention-backend comparison on Z-Image-Turbo for the #3079 PR.
# Z-Image-Turbo is a few-step distilled model (~2B params) that fits on a
# single GPU and runs in seconds — useful "small DiT" data point alongside
# Qwen-Image / FLUX.2-dev.
#
# Usage (from repo root):
#   bash benchmarks/diffusion/bench_e2e_zimage.sh
#   STEPS=8 SEED=42 bash benchmarks/diffusion/bench_e2e_zimage.sh
#
# Native config: 8 steps, CFG disabled (cfg_scale=1.0). The model is trained
# for that specific schedule — bumping steps doesn't improve quality.

set -euo pipefail

MODEL="${MODEL:-Tongyi-MAI/Z-Image-Turbo}"
PROMPT="${PROMPT:-A warm morning kitchen close-up of a woman and a man in their 30s standing across from each other at the counter, both holding mugs, cinematic, deadpan, golden morning light, shallow depth of field.}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${STEPS:-8}"
CFG_SCALE="${CFG_SCALE:-1.0}"
SEED="${SEED:-42}"

BACKENDS=(
  TORCH_SDPA
  CUDNN_ATTN
  FLASHINFER_ATTN
)

OUT_DIR="${OUT_DIR:-bench_zimage_out}"
mkdir -p "$OUT_DIR"

declare -A TOTALS
declare -A PER_STEP
declare -A PEAK_MEM

for BACKEND in "${BACKENDS[@]}"; do
  LOG="$OUT_DIR/zimage_${BACKEND}.log"
  IMG="$OUT_DIR/zimage_${BACKEND}.png"
  echo "======================================================================"
  echo "Running $BACKEND ..."
  echo "======================================================================"
  CUDA_VISIBLE_DEVICES=0 DIFFUSION_ATTENTION_BACKEND="$BACKEND" \
    python examples/offline_inference/text_to_image/text_to_image.py \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --num-inference-steps "$STEPS" \
      --cfg-scale "$CFG_SCALE" \
      --seed "$SEED" \
      --output "$IMG" 2>&1 | tee "$LOG"

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
echo "Z-Image-Turbo e2e ranking (${HEIGHT}x${WIDTH}, ${STEPS} steps, seed ${SEED})"
echo "======================================================================"
printf "%-18s %14s %14s %18s\n" "backend" "total (s)" "s/step" "peak VRAM"
printf "%-18s %14s %14s %18s\n" "------------------" "--------------" "--------------" "------------------"
for BACKEND in "${BACKENDS[@]}"; do
  printf "%-18s %14s %14s %18s\n" \
    "$BACKEND" "${TOTALS[$BACKEND]}" "${PER_STEP[$BACKEND]}" "${PEAK_MEM[$BACKEND]}"
done
