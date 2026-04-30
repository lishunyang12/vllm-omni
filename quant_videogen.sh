#!/usr/bin/env bash
# Auto-generate ModelOpt FP8 checkpoints for the quantization-quality benchmark.
#
# 3 models × 2 strategies (per-tensor + per-block 128x128) = 6 checkpoints:
#   1. Wan-AI/Wan2.2-I2V-A14B-Diffusers       (dual-transformer MoE, 2 GPUs)
#   2. hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v  (1 GPU)
#   3. Wan-AI/Wan2.1-VACE-14B-diffusers       (single transformer, R2V calib mix, 1 GPU)
#
# Schedule:
#   Stage 1 (sequential, both GPUs cuda:0+cuda:1):
#       Wan2.2 I2V A14B per-tensor → per-block
#   Stage 2 (parallel; runs after stage 1 finishes):
#       GPU 0:  HV-1.5 720p T2V per-tensor → per-block
#       GPU 1:  Wan2.1 VACE 14B R2V per-tensor → per-block
#
# Output directory layout (configurable via $OUTPUT_DIR, default ./quant_checkpoints):
#   ${OUTPUT_DIR}/wan22-i2v-a14b-fp8-per-tensor/
#   ${OUTPUT_DIR}/wan22-i2v-a14b-fp8-per-block/
#   ${OUTPUT_DIR}/hv15-720p-t2v-fp8-per-tensor/
#   ${OUTPUT_DIR}/hv15-720p-t2v-fp8-per-block/
#   ${OUTPUT_DIR}/wan21-vace-14b-r2v-fp8-per-tensor/
#   ${OUTPUT_DIR}/wan21-vace-14b-r2v-fp8-per-block/
#   ${OUTPUT_DIR}/logs/<task>.log         # one log per (model, strategy) pair
#
# Re-runnability: tasks are SKIPPED if the output dir already exists.
# Set FORCE=1 to overwrite all existing checkpoints.
#
# Required: $WAN_I2V_REF_IMAGES (Wan2.2 I2V) and $VACE_REF_IMAGES (VACE R2V)
# must point to directories of jpg/jpeg/png/webp files (or a single file).

set -euo pipefail

# ============================================================
# Configuration
# ============================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/quant_checkpoints}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

# Reference image directories. Override via env vars; default matches the
# existing /vllm-omni convention used in earlier ad-hoc commands.
WAN_I2V_REF_IMAGES="${WAN_I2V_REF_IMAGES:-/vllm-omni/reference-images}"
VACE_REF_IMAGES="${VACE_REF_IMAGES:-/vllm-omni/reference-images}"

# Model HF ids
WAN_I2V_A14B_MODEL="Wan-AI/Wan2.2-I2V-A14B-Diffusers"
HV15_720P_T2V_MODEL="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"
VACE_14B_MODEL="Wan-AI/Wan2.1-VACE-14B-diffusers"

# Producer scripts
WAN_PRODUCER="${REPO_ROOT}/examples/quantization/quantize_wan2_2_modelopt_fp8.py"
HV15_PRODUCER="${REPO_ROOT}/examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py"
VACE_PRODUCER="${REPO_ROOT}/examples/quantization/quantize_wan2_2_vace_modelopt_fp8.py"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ============================================================
# Pre-flight checks
# ============================================================
preflight() {
    local missing=0
    for path in "${WAN_PRODUCER}" "${HV15_PRODUCER}" "${VACE_PRODUCER}"; do
        if [[ ! -f "${path}" ]]; then
            echo "[preflight] FAIL — producer not found: ${path}" >&2
            missing=1
        fi
    done
    for ref in "${WAN_I2V_REF_IMAGES}" "${VACE_REF_IMAGES}"; do
        if [[ ! -e "${ref}" ]]; then
            echo "[preflight] FAIL — reference image path not found: ${ref}" >&2
            echo "             set WAN_I2V_REF_IMAGES / VACE_REF_IMAGES env vars" >&2
            missing=1
        fi
    done
    local n_gpus
    n_gpus=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
    if [[ "${n_gpus}" -lt 2 ]]; then
        echo "[preflight] FAIL — need at least 2 GPUs, detected ${n_gpus}." >&2
        missing=1
    fi
    [[ "${missing}" -eq 0 ]]
}
preflight || exit 1

# ============================================================
# Helpers
# ============================================================
ts() { date +%H:%M:%S; }

# run_quant <label> <cuda_devices> <output_dir> <producer_args...>
# - Skips if <output_dir> already exists (unless FORCE=1).
# - Logs stdout+stderr to ${LOG_DIR}/<label>.log
# - Returns producer's exit code (so the caller can decide on failure).
run_quant() {
    local label="$1" cuda_devs="$2" out_dir="$3"
    shift 3
    local logfile="${LOG_DIR}/${label}.log"

    if [[ -d "${out_dir}" && "${FORCE:-0}" != "1" ]]; then
        echo "[$(ts)] [${label}] SKIP — ${out_dir} exists. (FORCE=1 to redo.)"
        return 0
    fi

    echo "[$(ts)] [${label}] START — CUDA_VISIBLE_DEVICES=${cuda_devs}, log=${logfile}"
    if CUDA_VISIBLE_DEVICES="${cuda_devs}" python "$@" --output "${out_dir}" --overwrite \
            > "${logfile}" 2>&1; then
        echo "[$(ts)] [${label}] DONE — ${out_dir}"
        return 0
    else
        local rc=$?
        echo "[$(ts)] [${label}] FAILED (rc=${rc}) — see ${logfile}" >&2
        return ${rc}
    fi
}

# ============================================================
# Stage 1: Wan2.2 I2V A14B (dual-transformer, NEEDS BOTH GPUs)
# ============================================================
stage1_wan22_i2v() {
    echo "================================================================"
    echo "Stage 1: Wan2.2 I2V A14B (dual-transformer, both GPUs)"
    echo "================================================================"

    run_quant "wan22_i2v_a14b_per_tensor" "0,1" \
        "${OUTPUT_DIR}/wan22-i2v-a14b-fp8-per-tensor" \
        "${WAN_PRODUCER}" \
        --model "${WAN_I2V_A14B_MODEL}" \
        --is-i2v --reference-images "${WAN_I2V_REF_IMAGES}" \
        --calib-boundary-ratio 0.5

    run_quant "wan22_i2v_a14b_per_block" "0,1" \
        "${OUTPUT_DIR}/wan22-i2v-a14b-fp8-per-block" \
        "${WAN_PRODUCER}" \
        --model "${WAN_I2V_A14B_MODEL}" \
        --is-i2v --reference-images "${WAN_I2V_REF_IMAGES}" \
        --calib-boundary-ratio 0.5 \
        --weight-block-size 128,128
}

# ============================================================
# Stage 2: HV-1.5 720p T2V (GPU 0) || Wan2.1 VACE 14B R2V (GPU 1)
# ============================================================
stage2_hv15_branch() {
    run_quant "hv15_720p_t2v_per_tensor" "0" \
        "${OUTPUT_DIR}/hv15-720p-t2v-fp8-per-tensor" \
        "${HV15_PRODUCER}" \
        --model "${HV15_720P_T2V_MODEL}" \
        --variant t2v \
        --height 720 --width 1280

    run_quant "hv15_720p_t2v_per_block" "0" \
        "${OUTPUT_DIR}/hv15-720p-t2v-fp8-per-block" \
        "${HV15_PRODUCER}" \
        --model "${HV15_720P_T2V_MODEL}" \
        --height 720 --width 1280 \
        --variant t2v \
        --weight-block-size 128,128
}

stage2_vace_branch() {
    # Wan2.1-VACE-14B is single-transformer, so --calib-boundary-ratio is a no-op.
    # --reference-images switches calibration to mix T2V + R2V samples (R2V is
    # what we care about for the benchmark).
    run_quant "wan21_vace_14b_r2v_per_tensor" "1" \
        "${OUTPUT_DIR}/wan21-vace-14b-r2v-fp8-per-tensor" \
        "${VACE_PRODUCER}" \
        --model "${VACE_14B_MODEL}" \
        --reference-images "${VACE_REF_IMAGES}" \
        --height 480 --width 832 \

    run_quant "wan21_vace_14b_r2v_per_block" "1" \
        "${OUTPUT_DIR}/wan21-vace-14b-r2v-fp8-per-block" \
        "${VACE_PRODUCER}" \
        --model "${VACE_14B_MODEL}" \
        --reference-images "${VACE_REF_IMAGES}" \
        --height 480 --width 832 \
        --weight-block-size 128,128
}

stage2_parallel() {
    echo "================================================================"
    echo "Stage 2: HV-1.5 720p T2V (GPU 0) || Wan2.1 VACE 14B R2V (GPU 1)"
    echo "================================================================"

    stage2_hv15_branch &
    local hv_pid=$!
    stage2_vace_branch &
    local vace_pid=$!

    # `wait <pid>` returns that process's exit status; we want both branches'
    # status surfaced so a failure in one is reported but doesn't take down
    # the other (we still wait on both before exiting).
    local hv_rc=0 vace_rc=0
    wait "${hv_pid}" || hv_rc=$?
    wait "${vace_pid}" || vace_rc=$?

    if [[ "${hv_rc}" -ne 0 || "${vace_rc}" -ne 0 ]]; then
        echo "[$(ts)] Stage 2 had failures: hv15_rc=${hv_rc}, vace_rc=${vace_rc}" >&2
        return 1
    fi
}

# ============================================================
# Main
# ============================================================
echo "[$(ts)] Output dir: ${OUTPUT_DIR}"
echo "[$(ts)] Log dir:    ${LOG_DIR}"
echo "[$(ts)] FORCE=${FORCE:-0}  (set FORCE=1 to redo existing checkpoints)"
echo

stage1_wan22_i2v
stage2_parallel

echo
echo "================================================================"
echo "All checkpoints under ${OUTPUT_DIR}:"
echo "================================================================"
ls -d "${OUTPUT_DIR}"/*-fp8-* 2>/dev/null || echo "(none yet)"
