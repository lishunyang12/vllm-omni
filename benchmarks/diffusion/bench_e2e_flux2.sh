#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# End-to-end attention-backend comparison on FLUX.2-dev for the #3079 PR.
# FLUX.2-dev (32B DiT + ~48 GB Qwen2.5-VL text encoder) needs TP=2 to fit on
# 2x96 GB Blackwell — single-card OOMs even at 96 GB. This script runs the
# three BF16 backends in sequence with TP=2 and prints a ranking.
#
# Usage (from repo root, with both GPUs free):
#   bash benchmarks/diffusion/bench_e2e_flux2.sh
#   STEPS=28 SEED=7 bash benchmarks/diffusion/bench_e2e_flux2.sh
#
# Requires HuggingFace access to black-forest-labs/FLUX.2-dev. If you haven't
# accepted the license, the first run fails with 403; see
#   https://huggingface.co/black-forest-labs/FLUX.2-dev
# and run `huggingface-cli login` once with a Read token.

set -euo pipefail

MODEL="${MODEL:-black-forest-labs/FLUX.2-dev}"
PROMPT="${PROMPT:-A warm morning kitchen close-up of a woman and a man in their 30s standing across from each other at the counter, both holding mugs, cinematic, deadpan, golden morning light, shallow depth of field.}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
SEED="${SEED:-42}"
TP="${TP:-2}"

BACKENDS=(
  TORCH_SDPA
  CUDNN_ATTN
  FLASHINFER_ATTN
)

OUT_DIR="${OUT_DIR:-bench_flux2_out}"
mkdir -p "$OUT_DIR"

declare -A TOTALS
declare -A PER_STEP
declare -A PEAK_MEM

for BACKEND in "${BACKENDS[@]}"; do
  LOG="$OUT_DIR/flux2_${BACKEND}.log"
  IMG="$OUT_DIR/flux2_${BACKEND}.png"
  echo "======================================================================"
  echo "Running $BACKEND (TP=$TP) ..."
  echo "======================================================================"
  CUDA_VISIBLE_DEVICES=0,1 DIFFUSION_ATTENTION_BACKEND="$BACKEND" \
    python examples/offline_inference/text_to_image/text_to_image.py \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --num-inference-steps "$STEPS" \
      --cfg-scale "$CFG_SCALE" \
      --seed "$SEED" \
      --tensor-parallel-size "$TP" \
      --output "$IMG" 2>&1 | tee "$LOG"

  # Parse "Total generation time: NNN.NN seconds"
  TOTAL=$(grep -oE "Total generation time: [0-9.]+" "$LOG" | awk '{print $4}' | tail -1)
  # Parse tqdm step rate from the inference progress line
  STEP=$(grep -oE "[0-9]+\.[0-9]+s/it" "$LOG" | tail -1 | sed 's/s\/it//')
  if [[ -z "$STEP" ]]; then
    # Steps faster than a second print as it/s instead of s/it
    RATE=$(grep -oE "[0-9]+\.[0-9]+it/s" "$LOG" | tail -1 | sed 's/it\/s//')
    if [[ -n "$RATE" ]]; then
      STEP=$(awk -v r="$RATE" 'BEGIN { printf "%.3f", 1.0/r }')
    fi
  fi
  # Peak reserved memory across both GPUs (take last occurrence)
  MEM=$(grep -oE "Peak GPU memory.*reserved" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+ GB" | head -1)

  TOTALS[$BACKEND]="${TOTAL:-?}"
  PER_STEP[$BACKEND]="${STEP:-?}"
  PEAK_MEM[$BACKEND]="${MEM:-?}"
done

echo
echo "======================================================================"
echo "FLUX.2-dev e2e ranking (${HEIGHT}x${WIDTH}, ${STEPS} steps, TP=${TP}, seed ${SEED})"
echo "======================================================================"
printf "%-18s %14s %14s %18s\n" "backend" "total (s)" "s/step" "peak VRAM"
printf "%-18s %14s %14s %18s\n" "------------------" "--------------" "--------------" "------------------"
for BACKEND in "${BACKENDS[@]}"; do
  printf "%-18s %14s %14s %18s\n" \
    "$BACKEND" "${TOTALS[$BACKEND]}" "${PER_STEP[$BACKEND]}" "${PEAK_MEM[$BACKEND]}"
done

echo
echo "Images saved under $OUT_DIR/. To compare frame parity:"
echo "  python -c \"from PIL import Image; import numpy as np; \\"
echo "    a=np.asarray(Image.open('$OUT_DIR/flux2_TORCH_SDPA.png')); \\"
echo "    b=np.asarray(Image.open('$OUT_DIR/flux2_FLASHINFER_ATTN.png')); \\"
echo "    print('mean abs diff:', np.abs(a.astype(int)-b.astype(int)).mean())\""
