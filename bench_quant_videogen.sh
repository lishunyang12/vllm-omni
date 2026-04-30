#!/usr/bin/env bash
# Benchmark the 6 ModelOpt FP8 checkpoints produced by quant_videogen.sh.
#
# Index → GPU: even → cuda:0, odd → cuda:1. Each pair (i, i+1) runs in
# parallel; pairs run sequentially.
#
#   0  Wan-AI/Wan2.2-I2V-A14B-Diffusers              per-tensor   GPU 0
#   1  Wan-AI/Wan2.2-I2V-A14B-Diffusers              per-block    GPU 1
#   2  HunyuanVideo-1.5-Diffusers-720p_t2v           per-tensor   GPU 0
#   3  HunyuanVideo-1.5-Diffusers-720p_t2v           per-block    GPU 1
#   4  Wan-AI/Wan2.1-VACE-14B-diffusers              per-tensor   GPU 0
#   5  Wan-AI/Wan2.1-VACE-14B-diffusers              per-block    GPU 1
#
# For each config the bench script (benchmarks/diffusion/quantization_quality.py)
# loads the BF16 baseline, generates the prompts, unloads, then loads the FP8
# checkpoint and re-generates the same prompts with the same seed; LPIPS,
# Speedup, Throughput, Model VRAM, Peak VRAM are written into <output>/results.md.
#
# Re-runnability: a config is SKIPPED if its output dir already exists.
# Set FORCE=1 to redo all six.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/quant_checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/quant_bench_results}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

# Image dirs for i2v / R2V tasks. Override via env if your layout differs.
WAN_I2V_IMAGES="${WAN_I2V_IMAGES:-/vllm-omni/images}"
VACE_REF_IMAGES="${VACE_REF_IMAGES:-/vllm-omni/images}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

BENCH_SCRIPT="${REPO_ROOT}/benchmarks/diffusion/quantization_quality.py"

# Same prompt set across all configs so LPIPS rows can be compared meaningfully.
PROMPTS=(
    "An astronaut in a white spacesuit riding a horse across the lunar surface, gray dust kicked up by the horse's hooves, Earth visible in the black sky, lunar lander in the distance, cinematic wide shot. Make sure the astronaut is really moving!"
    "A skateboarder in a purple bomber jacket doing a kickflip in a foggy urban plaza, overcast morning light, slow motion, european architecture in the background."
)

# Wan2.2 I2V degenerates to a static first-frame copy without an anti-static
# negative prompt — Wan's official one explicitly negates "static / still /
# still frame". HunyuanVideo and Wan2.1 VACE produce normal motion with an
# empty negative, so this is only passed to the Wan I2V configs below.
WAN_I2V_NEGATIVE_PROMPT="vibrant colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, still frame, cluttered background, three legs, many people in the background, walking backwards"

# ============================================================
# Pre-flight
# ============================================================
preflight() {
    local missing=0
    [[ -f "${BENCH_SCRIPT}" ]] || { echo "[preflight] FAIL — bench script not found: ${BENCH_SCRIPT}" >&2; missing=1; }
    [[ -d "${CKPT_DIR}" ]] || { echo "[preflight] FAIL — checkpoint dir not found: ${CKPT_DIR}" >&2; missing=1; }
    for ref in "${WAN_I2V_IMAGES}" "${VACE_REF_IMAGES}"; do
        if [[ ! -e "${ref}" ]]; then
            echo "[preflight] FAIL — image dir not found: ${ref}" >&2
            echo "             set WAN_I2V_IMAGES / VACE_REF_IMAGES env vars" >&2
            missing=1
        fi
    done
    local n_gpus
    n_gpus=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
    if [[ "${n_gpus}" -lt 2 ]]; then
        echo "[preflight] FAIL — need ≥ 2 GPUs (even/odd split), detected ${n_gpus}." >&2
        missing=1
    fi
    [[ "${missing}" -eq 0 ]]
}
preflight || exit 1

# ============================================================
# Helpers
# ============================================================
ts() { date +%H:%M:%S; }

# run_bench <label> <gpu> <bench_args...>
# Skips if ${OUTPUT_ROOT}/<label> already exists (unless FORCE=1).
# Always passes --output-dir; the caller must NOT include it in <bench_args>.
run_bench() {
    local label="$1" gpu="$2"
    shift 2
    local out_dir="${OUTPUT_ROOT}/${label}"
    local logfile="${LOG_DIR}/${label}.log"

    if [[ -d "${out_dir}" && "${FORCE:-0}" != "1" ]]; then
        echo "[$(ts)] [${label}] SKIP — ${out_dir} exists. (FORCE=1 to redo.)"
        return 0
    fi

    echo "[$(ts)] [${label}] START — GPU ${gpu}, log=${logfile}"
    if CUDA_VISIBLE_DEVICES="${gpu}" python "${BENCH_SCRIPT}" "$@" \
            --output-dir "${out_dir}" > "${logfile}" 2>&1; then
        echo "[$(ts)] [${label}] DONE — ${out_dir}/results.md"
        return 0
    else
        local rc=$?
        echo "[$(ts)] [${label}] FAILED (rc=${rc}) — see ${logfile}" >&2
        return ${rc}
    fi
}

# ============================================================
# Per-config bench commands. Index in name encodes GPU (0,2,4 → 0; 1,3,5 → 1).
# Each function intentionally repeats its full arg list (rather than sharing
# via variables) so it's straightforward to tweak one config without
# affecting others.
# ============================================================

bench_0_wan22_i2v_per_tensor() {  # GPU 0
    run_bench "wan22-i2v-a14b-fp8-per-tensor" "0" \
        --use-offline-quant \
        --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
        --model-quant-checkpoint "${CKPT_DIR}/wan22-i2v-a14b-fp8-per-tensor" \
        --task i2v \
        --quantization fp8 \
        --prompts "${PROMPTS[@]}" \
        --negative-prompt "${WAN_I2V_NEGATIVE_PROMPT}" \
        --images "${WAN_I2V_IMAGES}" \
        --height 720 --width 1280 \
        --num-frames 81 --num-inference-steps 50 --seed 42 \
        --vae-use-tiling
}

bench_1_wan22_i2v_per_block() {   # GPU 1
    run_bench "wan22-i2v-a14b-fp8-per-block" "1" \
        --use-offline-quant \
        --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
        --model-quant-checkpoint "${CKPT_DIR}/wan22-i2v-a14b-fp8-per-block" \
        --task i2v \
        --quantization fp8 \
        --prompts "${PROMPTS[@]}" \
        --negative-prompt "${WAN_I2V_NEGATIVE_PROMPT}" \
        --images "${WAN_I2V_IMAGES}" \
        --height 720 --width 1280 \
        --num-frames 81 --num-inference-steps 50 --seed 42 \
        --vae-use-tiling
}

bench_2_hv15_720p_t2v_per_tensor() {  # GPU 0
    run_bench "hv15-720p-t2v-fp8-per-tensor" "0" \
        --use-offline-quant \
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v \
        --model-quant-checkpoint "${CKPT_DIR}/hv15-720p-t2v-fp8-per-tensor" \
        --task t2v \
        --quantization fp8 \
        --prompts "${PROMPTS[@]}" \
        --height 720 --width 1280 \
        --guidance-scale 6.0 \
        --num-frames 49 --num-inference-steps 30 --seed 42 \
        --vae-use-tiling
}

bench_3_hv15_720p_t2v_per_block() {   # GPU 1
    run_bench "hv15-720p-t2v-fp8-per-block" "1" \
        --use-offline-quant \
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v \
        --model-quant-checkpoint "${CKPT_DIR}/hv15-720p-t2v-fp8-per-block" \
        --task t2v \
        --quantization fp8 \
        --prompts "${PROMPTS[@]}" \
        --height 720 --width 1280 \
        --guidance-scale 6.0 \
        --num-frames 49 --num-inference-steps 30 --seed 42 \
        --vae-use-tiling
}

bench_4_vace_14b_r2v_per_tensor() {   # GPU 0
    # VACE R2V: pass --task i2v so the bench script loads --images and routes
    # via multi_modal_data; vllm-omni's VACE pipeline maps that to its
    # reference_images kwarg. Use 480p (VACE native res) — 720p inflates
    # latents past the comfortable single-GPU range for 14B + VAE.
    run_bench "wan21-vace-14b-r2v-fp8-per-tensor" "0" \
        --use-offline-quant \
        --model Wan-AI/Wan2.1-VACE-14B-diffusers \
        --model-quant-checkpoint "${CKPT_DIR}/wan21-vace-14b-r2v-fp8-per-tensor" \
        --task i2v \
        --quantization fp8 \
        --prompts "${PROMPTS[@]}" \
        --images "${VACE_REF_IMAGES}" \
        --height 480 --width 832 \
        --num-frames 49 --num-inference-steps 30 --seed 42 \
        --vae-use-tiling
}

bench_5_vace_14b_r2v_per_block() {    # GPU 1
    run_bench "wan21-vace-14b-r2v-fp8-per-block" "1" \
        --use-offline-quant \
        --model Wan-AI/Wan2.1-VACE-14B-diffusers \
        --model-quant-checkpoint "${CKPT_DIR}/wan21-vace-14b-r2v-fp8-per-block" \
        --task i2v \
        --quantization fp8 \
        --prompts "${PROMPTS[@]}" \
        --images "${VACE_REF_IMAGES}" \
        --height 480 --width 832 \
        --num-frames 49 --num-inference-steps 30 --seed 42 \
        --vae-use-tiling
}

# ============================================================
# Pair runner
# ============================================================
# `wait <pid>` propagates the underlying process's exit code, so we capture
# both branches' rc to surface failures without one taking the other down.
run_pair() {
    local desc="$1" fn_a="$2" fn_b="$3"
    echo "================================================================"
    echo "Pair: ${desc}"
    echo "================================================================"
    "${fn_a}" &
    local pid_a=$!
    "${fn_b}" &
    local pid_b=$!

    local rc_a=0 rc_b=0
    wait "${pid_a}" || rc_a=$?
    wait "${pid_b}" || rc_b=$?

    if [[ "${rc_a}" -ne 0 || "${rc_b}" -ne 0 ]]; then
        echo "[$(ts)] Pair '${desc}' had failures: ${fn_a}=${rc_a}, ${fn_b}=${rc_b}" >&2
        return 1
    fi
}

# ============================================================
# Main
# ============================================================
echo "[$(ts)] Checkpoint dir: ${CKPT_DIR}"
echo "[$(ts)] Output dir:     ${OUTPUT_ROOT}"
echo "[$(ts)] Log dir:        ${LOG_DIR}"
echo "[$(ts)] FORCE=${FORCE:-0}  (set FORCE=1 to redo existing benches)"
echo

run_pair "Wan2.2 I2V A14B  {per-tensor (GPU0) || per-block (GPU1)}" \
    bench_0_wan22_i2v_per_tensor bench_1_wan22_i2v_per_block

run_pair "HV-1.5 720p T2V   {per-tensor (GPU0) || per-block (GPU1)}" \
    bench_2_hv15_720p_t2v_per_tensor bench_3_hv15_720p_t2v_per_block

run_pair "Wan2.1 VACE 14B   {per-tensor (GPU0) || per-block (GPU1)}" \
    bench_4_vace_14b_r2v_per_tensor bench_5_vace_14b_r2v_per_block

echo
echo "================================================================"
echo "All benchmarks done. results.md per config:"
echo "================================================================"
for d in "${OUTPUT_ROOT}"/*/; do
    if [[ -f "${d}results.md" ]]; then
        echo "  ${d}results.md"
    fi
done