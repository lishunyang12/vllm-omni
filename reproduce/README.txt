Reproduce PR #10390 quality validation
====================================

Requirements: four Rubin GPUs, compatible host driver, Docker with NVIDIA
Container Toolkit, and approximately 600 GB of writable storage for weights,
caches, lossless frames and viewing videos. The pinned image is the same image
used in the H3 runtime work. No SGLang comparison is made by this script.

From the directory containing prepare.sh and matrix.json:

  mkdir -p /path/to/validation-data
  docker run --rm --gpus all --ipc=host \
    -v "$PWD:/validation:ro" \
    -v /path/to/validation-data:/work \
    --entrypoint bash \
    vllm/vllm-openai:cu134-nightly-1648fefab91204f3b15bf8062217316f07f2c4bb \
    /validation/prepare.sh /work

Outputs:
  /path/to/validation-data/report.html     Before/after images and playable videos
  /path/to/validation-data/metrics.csv     Summary quality metrics
  /path/to/validation-data/metrics.json    Definitions, scores, unavailable pairs
  /path/to/validation-data/outputs/        Lossless PNG frames and per-case metadata
  /path/to/validation-data/logs/           Full generation logs (including failures)
  /path/to/validation-data/evidence/       Revisions, runtime, hardware, run results

The default matrix contains H3-Image (4096x4096, 6 steps), Qwen-Image
(1024x1024, 50 steps), Wan2.2-TI2V-5B (1280x704, 121 frames, 50 steps),
and MiniMax-H3 FL2VA text-to-video/audio (1344x768, 124 frames, 50 steps).
Each uses the fixed prompt, seed and sampling parameters recorded in matrix.json.
All modes use eager execution, four GPUs, DiT TP1/Ulysses4; H3 encoder TP4,
other encoders TP1. Quantization is applied only to the diffusion transformer.
H3 video enables trust_remote_code for its native VAE implementation at the
pinned checkpoint revision and uses its native 17n+5 frame grid (124 frames).
The models retain their built-in exclusions for unsupported or sensitive layers.

Comparisons:
  base_bf16 -> pr_bf16: isolates the implementation changes from quantization.
  pr_bf16 -> pr_mxfp8: measures the change from online MXFP8.
  pr_bf16 -> pr_fp8_block: measures existing fp8_per_block online quantization.
The base cannot run this PR's CUDA MXFP8 adapter; baseline-before-MXFP8 means BF16.

SSIM and LPIPS are averaged over frames at native output resolution. PSNR uses
pooled RGB pixel MSE across all frames. The script rejects mismatched frame
counts or dimensions; it does not resize, truncate, or evaluate encoded MP4s.
SSIM uses Gaussian weighting (sigma 1.5) and population covariance. LPIPS is
AlexNet v0.1 on RGB in [-1,1]. MP4s are H.264 viewing copies; source PNGs are
lossless. Viewing MP4s contain video only; the H3 comparison does not save or
score audio.

This is one matched example per model, not a broad quality-equivalence claim.
Generation time includes the first request and possible compilation/autotuning;
it must not be interpreted as warmed steady-state performance or a P50 result.
Failed cases stay explicit. Rerunning reproduce.sh skips completed cases and
archives failed attempts before retrying them. For a new seed/prompt/config,
use a NEW /work directory so results cannot be mixed with existing cases.

Reference presentation: https://github.com/vllm-project/vllm-omni/pull/1764
This uses paired BF16/quantized outputs as in that PR. Its image LPIPS helper
resized to 256x256; this script scores native resolution, so absolute LPIPS
numbers should not be compared directly between the two reports.
Implementation PR: https://github.com/vllm-project/vllm-omni/pull/10390

Optional blockwise FP8 operator check, after prepare.sh has installed Omni:

  uv pip install --system pytest
  PYTHONPATH=/work/source/candidate python3 -m pytest -q -s \
    /validation/test_fp8_block_runtime.py

This checks BF16/FP16 engine dtypes, 2D/3D strided inputs, bias, fused QKV and
MLP loading, and ignored layers against unquantized torch linear operations.
It uses small synthetic tensors; the generated-media comparisons remain the
model-level quality evidence.

Optional export integrity check in the prepared container:

  python3 /validation/verify_artifacts.py /work

This verifies every PNG pixel hash, score aggregation and complete MP4 decoding.
