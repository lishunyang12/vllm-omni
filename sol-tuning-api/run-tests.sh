#!/usr/bin/env bash
set -Eeuo pipefail
work=/work
mkdir -p "$work/source" "$work/evidence"
tar xzf /evidence/source.tar.gz -C "$work/source"
trap 'cp -a "$work/evidence/." /evidence/; echo $? > /evidence/copy.exit' EXIT
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="$work/hf-cache" UV_CACHE_DIR="$work/uv-cache" XDG_CACHE_HOME="$work/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$work/torchinductor-cache" TRITON_CACHE_DIR="$work/triton-cache"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv > "$work/evidence/hardware.csv"
python3 - "$work" <<'PY'
from importlib.metadata import distributions,version
from pathlib import Path
import sys
w=Path(sys.argv[1]);keep={'vllm','torch','torchvision','torchaudio','triton','pytorch-triton','nixl','apache-tvm-ffi','tilelang','quack-kernels','humming-kernels','tokenspeed-mla'}
lines=[]
for d in distributions():
 n=d.metadata['Name'].lower().replace('_','-')
 if n!='nvidia-cudnn-frontend' and (n in keep or n.startswith(('nvidia-','cuda-','flashinfer-'))):lines.append(f'{n}=={d.version}')
(w/'evidence/runtime-constraints.txt').write_text('\n'.join(sorted(lines))+'\n')
(w/'evidence/runtime-overrides.txt').write_text('\n'.join(f'{n}=={version(n)}' for n in ['triton','nvidia-nccl-cu13'])+'\n')
PY
VLLM_OMNI_VERSION_OVERRIDE=0.33.0+gca5cc9f.solapi VLLM_OMNI_TARGET_DEVICE=cuda \
uv pip install --system --prerelease=allow --constraint "$work/evidence/runtime-constraints.txt" \
 --override "$work/evidence/runtime-overrides.txt" --constraint "/evidence/validated-constraints.txt" \
 -e "$work/source" 'nvidia-cudnn-frontend==1.31.0' 'diffusers==0.40.0' 'transformers==5.19.0' \
 'accelerate==1.12.0' 'av==19.0.2' 'lpips==0.1.4' 'scikit-image==0.25.2' \
 'huggingface-hub==1.33.0' 'numpy==2.2.6' 'pillow==13.0.0' 'pytest==9.2.0' 'pytest-asyncio==2.0.0a1' 'pytest-xdist==3.8.0' 'pytest-mock==3.17.0'

uv pip freeze > "$work/evidence/installed-environment.txt"
cd "$work/source"
set +e
python3 -m pytest -q --run-level=core_model \
 tests/diffusion/attention/test_sol_attn_backend.py \
 tests/diffusion/attention/test_attention_config.py \
 tests/diffusion/attention/test_attention_schedule_candidates.py \
 tests/diffusion/attention/test_impl_overrides.py \
 tests/diffusion/models/wan2_2/test_wan22_scheduled_attention_compile.py \
 --junitxml="$work/evidence/tests.xml" > "$work/evidence/tests.log" 2>&1
result=$?
echo "$result" > "$work/evidence/tests.exit"
tail -n 35 "$work/evidence/tests.log"
exit "$result"
