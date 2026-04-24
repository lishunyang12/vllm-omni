#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# End-to-end attention-backend comparison for the #3079 PR. Runs the same
# HunyuanVideo-1.5 generation under each BF16 backend and prints a ranking.
# Use this to pick the auto-route winner for Blackwell.
#
# Usage (from repo root):
#   bash benchmarks/diffusion/bench_e2e_attention.sh
#   MODEL=... FRAMES=33 STEPS=50 bash benchmarks/diffusion/bench_e2e_attention.sh
#
# Uses seed 42 so frame-level diffs are reproducible between backends. Captures
# "Total generation time" and "per-step" lines from each run.

set -euo pipefail

MODEL="${MODEL:-hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v}"
PROMPT="${PROMPT:-A cat walks through a sunlit garden, flowers swaying gently in the breeze.}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
FRAMES="${FRAMES:-33}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-6.0}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
FPS="${FPS:-24}"
SEED="${SEED:-42}"

BACKENDS=(
  TORCH_SDPA
  FLASH_ATTN
  CUDNN_ATTN
  FLASHINFER_ATTN
)

OUT_DIR="${OUT_DIR:-bench_e2e_out}"
mkdir -p "$OUT_DIR"

declare -A TOTALS
declare -A PER_STEP
declare -A PEAK_MEM

for BACKEND in "${BACKENDS[@]}"; do
  LOG="$OUT_DIR/hv15_${BACKEND}.log"
  VID="$OUT_DIR/hv15_${BACKEND}.mp4"
  echo "======================================================================"
  echo "Running $BACKEND ..."
  echo "======================================================================"
  DIFFUSION_ATTENTION_BACKEND="$BACKEND" \
    python examples/offline_inference/text_to_video/text_to_video.py \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --num-frames "$FRAMES" \
      --guidance-scale "$GUIDANCE" \
      --flow-shift "$FLOW_SHIFT" \
      --num-inference-steps "$STEPS" \
      --fps "$FPS" \
      --seed "$SEED" \
      --output "$VID" 2>&1 | tee "$LOG"

  # Parse "Total generation time: NNN.NN seconds"
  TOTAL=$(grep -oE "Total generation time: [0-9.]+" "$LOG" | awk '{print $4}' | tail -1)
  # Parse tqdm step rate "NN.NNs/it" or "N.Ns/it" from the progress line
  STEP=$(grep -oE "[0-9]+\.[0-9]+s/it" "$LOG" | tail -1 | sed 's/s\/it//')
  # Parse peak reserved memory
  MEM=$(grep -oE "Peak GPU memory.*reserved" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+ GB" | head -1)

  TOTALS[$BACKEND]="${TOTAL:-?}"
  PER_STEP[$BACKEND]="${STEP:-?}"
  PEAK_MEM[$BACKEND]="${MEM:-?}"
done

echo
echo "======================================================================"
echo "E2E ranking (HunyuanVideo-1.5 ${HEIGHT}x${WIDTH}, ${FRAMES} frames, ${STEPS} steps, seed ${SEED})"
echo "======================================================================"
printf "%-18s %14s %14s %18s\n" "backend" "total (s)" "s/step" "peak VRAM"
printf "%-18s %14s %14s %18s\n" "------------------" "--------------" "--------------" "------------------"
for BACKEND in "${BACKENDS[@]}"; do
  printf "%-18s %14s %14s %18s\n" \
    "$BACKEND" "${TOTALS[$BACKEND]}" "${PER_STEP[$BACKEND]}" "${PEAK_MEM[$BACKEND]}"
done

echo
echo "Videos saved under $OUT_DIR/. Compare frame-level parity with:"
echo "  ffmpeg -i $OUT_DIR/hv15_TORCH_SDPA.mp4 -i $OUT_DIR/hv15_FLASHINFER_ATTN.mp4 \\"
echo "         -filter_complex \"psnr\" -f null -"
