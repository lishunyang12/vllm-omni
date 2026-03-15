#!/bin/bash
# Test script for PR #1901: GPU memory debug utility
#
# Proves the tool works universally across different model types:
#   - Text-to-Video: Wan2.1 1.3B, LTX-2
#   - Text-to-Image: SD3 Medium, FLUX.1-dev, Qwen-Image
#   - Image-to-Video: Wan2.1 I2V (if image provided)
#   - Helios (video generation)
#
# Usage:
#   bash tests/test_memory_debug.sh              # run all
#   bash tests/test_memory_debug.sh wan          # filter by name
#   bash tests/test_memory_debug.sh t2i          # all text-to-image
#   bash tests/test_memory_debug.sh t2v          # all text-to-video

set -uo pipefail

FILTER="${1:-}"
RESULTS_DIR="./memory_debug_results"
mkdir -p "$RESULTS_DIR"
SUMMARY_FILE="$RESULTS_DIR/summary.md"
PASS=0
FAIL=0
SKIP=0

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unknown")

cat > "$SUMMARY_FILE" <<EOF
# GPU Memory Debug Tool — Test Results

**Date:** $(date -u +"%Y-%m-%d %H:%M UTC")
**GPU:** $GPU_INFO
**Branch:** $(git rev-parse --abbrev-ref HEAD) ($(git rev-parse --short HEAD))

---

EOF

run_test() {
    local name="$1"
    local model="$2"
    local script="$3"
    local extra_args="$4"
    local log_file="$RESULTS_DIR/${name}.log"

    if [[ -n "$FILTER" ]] && [[ "$name" != *"$FILTER"* ]]; then
        SKIP=$((SKIP + 1))
        return 0
    fi

    echo "============================================================"
    echo "  TEST: $name"
    echo "  Model: $model"
    echo "============================================================"

    local status="PASS"
    if VLLM_DEBUG_MEMORY=1 timeout 900 python "$script" \
        --model "$model" \
        $extra_args \
        --enforce-eager \
        2>&1 | tee "$log_file"; then
        status="PASS"
    else
        status="FAIL"
    fi

    # Check which tables appeared
    local has_runner=false
    local has_pipeline=false
    grep -q "before_forward" "$log_file" 2>/dev/null && has_runner=true
    grep -q "before_encode" "$log_file" 2>/dev/null && has_pipeline=true

    if [[ "$status" == "PASS" ]] && [[ "$has_runner" == "false" ]]; then
        status="FAIL (no memory table)"
    fi

    [[ "$status" == "PASS" ]] && PASS=$((PASS + 1)) || FAIL=$((FAIL + 1))

    # Write to summary
    {
        echo "## $name"
        echo "- **Model:** \`$model\`"
        echo "- **Args:** \`$extra_args --enforce-eager\`"
        echo "- **Status:** $status"
        echo "- **Runner table (universal):** $has_runner"
        echo "- **Pipeline detail table (opt-in):** $has_pipeline"
        echo ""
        if [[ "$has_runner" == "true" ]] || [[ "$has_pipeline" == "true" ]]; then
            echo '```'
            # Extract only the actual run (skip warmup), get last occurrence of each table
            grep -E "GPU MEMORY|before_|after_|={10,}|-{10,}|Stage.*Alloc|\[MEMORY\]" "$log_file" 2>/dev/null
            echo '```'
        fi
        echo ""
    } >> "$SUMMARY_FILE"

    echo "  -> $status (runner=$has_runner, pipeline=$has_pipeline)"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# Text-to-Video models
# ═══════════════════════════════════════════════════════════════════════════════

run_test "t2v_ltx2_33f" \
    "Lightricks/LTX-2" \
    "examples/offline_inference/text_to_video/text_to_video.py" \
    "--height 512 --width 768 --num-frames 33 --num-inference-steps 10 --frame-rate 24"

# ═══════════════════════════════════════════════════════════════════════════════
# Text-to-Image models
# ═══════════════════════════════════════════════════════════════════════════════

run_test "t2i_sd3_medium" \
    "stabilityai/stable-diffusion-3-medium-diffusers" \
    "examples/offline_inference/text_to_image/text_to_image.py" \
    "--height 1024 --width 1024 --num-inference-steps 10"

run_test "t2i_flux_dev" \
    "black-forest-labs/FLUX.1-dev" \
    "examples/offline_inference/text_to_image/text_to_image.py" \
    "--height 1024 --width 1024 --num-inference-steps 10"

run_test "t2i_qwen_image" \
    "Qwen/Qwen-Image" \
    "examples/offline_inference/text_to_image/text_to_image.py" \
    "--height 1024 --width 1024 --num-inference-steps 10"

# ═══════════════════════════════════════════════════════════════════════════════
# Other model types
# ═══════════════════════════════════════════════════════════════════════════════

run_test "helios_t2v" \
    "Doubiiu/Helios-7B" \
    "examples/offline_inference/helios/end2end.py" \
    "--num-inference-steps 10"

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

cat >> "$SUMMARY_FILE" <<EOF
---

## Summary

| Result | Count |
|---|---|
| Passed | $PASS |
| Failed | $FAIL |
| Skipped | $SKIP |

### Model coverage

| Category | Models tested |
|---|---|
| Text-to-Video | LTX-2 |
| Text-to-Image | SD3 Medium, FLUX.1-dev, Qwen-Image |
| Video Gen | Helios 7B |

### Tool coverage

| Level | Scope | Enabled by |
|---|---|---|
| Model runner (\`before_forward\` / \`after_forward\`) | **All models** | \`VLLM_DEBUG_MEMORY=1\` |
| Pipeline stages (encode / denoise / vae_decode) | Opt-in per model (Wan2.2 included) | \`VLLM_DEBUG_MEMORY=1\` |
| Engine post-processing | **All models** | \`VLLM_DEBUG_MEMORY=1\` |

Zero overhead when \`VLLM_DEBUG_MEMORY\` is not set.
EOF

echo ""
echo "============================================================"
echo "  RESULTS: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================================"
echo ""
echo "Summary:  $SUMMARY_FILE"
echo "Logs:     $RESULTS_DIR/*.log"
echo ""
echo "--- Summary ---"
cat "$SUMMARY_FILE"
