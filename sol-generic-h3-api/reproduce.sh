#!/usr/bin/env bash
# Run inside vllm/vllm-openai:cu134-nightly-1648fefab91204f3b15bf8062217316f07f2c4bb.
# Requires git, uv, network access, and two scheduler-assigned CUDA GPUs.
set -Eeuo pipefail

evidence_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work=${SOL_REPRO_WORK:-/work/sol-generic-h3}
omni_revision=842f3f235def1146a516ef5ac146e3b6a1b170f1
kernel_revision=e3f60d5fdf15ddab41c6079c17fccc8d10d62c29
mkdir -p "$work"
work=$(cd -- "$work" && pwd)
if [[ -e "$work/source" || -e "$work/sana" ]]; then
    echo 'Use an empty SOL_REPRO_WORK directory to preserve source provenance.' >&2
    exit 1
fi
mkdir -p "$work/results"

git init -q "$work/source"
git -C "$work/source" fetch --depth=1 https://github.com/lishunyang12/vllm-omni.git "$omni_revision"
git -C "$work/source" checkout --detach FETCH_HEAD
git init -q "$work/sana"
git -C "$work/sana" fetch --depth=1 https://github.com/lishunyang12/Sana.git "$kernel_revision"
git -C "$work/sana" checkout --detach FETCH_HEAD

export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="$work/hf-cache" UV_CACHE_DIR="$work/uv-cache" XDG_CACHE_HOME="$work/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$work/torchinductor-cache" TRITON_CACHE_DIR="$work/triton-cache"
# These constraints preserve the core packages from the tested fixed image.
VLLM_OMNI_VERSION_OVERRIDE=0.33.0+g842f3f2.solapi VLLM_OMNI_TARGET_DEVICE=cuda \
uv pip install --system --prerelease=allow \
    --constraint "$evidence_dir/runtime-constraints.txt" \
    --override "$evidence_dir/runtime-overrides.txt" \
    --constraint "$evidence_dir/validated-constraints.txt" \
    -e "$work/source" \
    'nvidia-cudnn-frontend==1.31.0' 'diffusers==0.40.0' 'transformers==5.19.0' \
    'accelerate==1.12.0' 'av==19.0.2' 'lpips==0.1.4' 'scikit-image==0.25.2' \
    'huggingface-hub==1.33.0' 'numpy==2.2.6' 'pillow==13.0.0' \
    'pytest==9.2.0' 'pytest-asyncio==2.0.0a1' 'pytest-xdist==3.8.0' 'pytest-mock==3.17.0'
uv pip install --system --no-deps --no-build-isolation "$work/sana/techniques/sparse_backends"
uv pip freeze > "$work/results/installed-environment.txt"
python3 - <<'PY' > "$work/results/environment.txt"
from importlib.metadata import version
import sys
import torch

print(sys.version)
for name in ("torch", "vllm", "nvidia-cutlass-dsl", "cuda-python", "pytest", "sol-attn"):
    print(f"{name}=={version(name)}")
print("Visible GPUs:", torch.accelerator.device_count())
print("GPU:", torch.cuda.get_device_name())
PY

cd "$work/source"
set +e
python3 -m pytest -q --run-level=core_model \
    tests/diffusion/attention/test_sol_attn_backend.py \
    tests/diffusion/attention/test_attention_config.py \
    tests/diffusion/models/minimax_h3/test_minimax_h3_contract.py \
    tests/diffusion/models/minimax_h3/test_minimax_h3_parallel.py \
    tests/diffusion/attention/test_attention_schedule_candidates.py \
    tests/diffusion/attention/test_impl_overrides.py \
    tests/diffusion/models/wan2_2/test_wan22_scheduled_attention_compile.py \
    --junitxml="$work/results/cpu.xml" > "$work/results/cpu.log" 2>&1
cpu_result=$?
echo "$cpu_result" > "$work/results/cpu.exit"
tail -n 20 "$work/results/cpu.log"
python3 -m pytest -q --run-level=core_model \
    tests/diffusion/attention/test_sol_attn_cuda.py \
    --junitxml="$work/results/cuda.xml" > "$work/results/cuda.log" 2>&1
cuda_result=$?
echo "$cuda_result" > "$work/results/cuda.exit"
tail -n 25 "$work/results/cuda.log"
exit $((cpu_result || cuda_result))
