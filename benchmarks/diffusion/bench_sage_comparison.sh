#!/bin/bash
# Benchmark: HunyuanVideo 1.5 480p — BF16 baseline vs SageAttention
# Resolution: 480×832, 33 frames
set -e

MODEL="hunyuanvideo-community/HunyuanVideo-1.5-480p_t2v"
PROMPT="A serene lakeside sunrise with mist over the water."
SCRIPT="examples/offline_inference/text_to_video/text_to_video.py"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace}"

COMMON_ARGS="--model $MODEL \
  --height 480 --width 832 --num-frames 33 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --seed 42 \
  --vae-use-tiling \
  --enforce-eager"

echo "============================================"
echo "=== 1/2: BF16 + FlashAttention (baseline)==="
echo "============================================"
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
  python $SCRIPT $COMMON_ARGS \
  --output "$OUTPUT_DIR/output_flash_attn.mp4"

echo ""
echo "============================================"
echo "=== 2/2: BF16 + SageAttention            ==="
echo "============================================"
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN \
  python $SCRIPT $COMMON_ARGS \
  --output "$OUTPUT_DIR/output_sage_attn.mp4"

echo ""
echo "=== Done. Compare: output_flash_attn.mp4 vs output_sage_attn.mp4 ==="
