#!/bin/bash
set -Eeuo pipefail
work=${1:-/work}
mkdir -p "$work/evidence" "$work/source/base" "$work/source/head" "$work/results" "$work/assets"
export HF_HOME="$work/hf-cache" UV_CACHE_DIR="$work/uv-cache" XDG_CACHE_HOME="$work/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$work/torchinductor-cache" TRITON_CACHE_DIR="$work/triton-cache" TORCH_HOME="$work/torch-cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DOWNLOAD_TIMEOUT=600
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
nvidia-smi --query-gpu=index,uuid,name,compute_cap,memory.total,driver_version --format=csv > "$work/evidence/hardware.csv"
nvidia-smi topo -m > "$work/evidence/topology.txt"
curl -fL --retry 3 https://codeload.github.com/lishunyang12/vllm-omni/tar.gz/6f5a99768f668bff08e40b3219e308743157d36c | tar xz --strip-components=1 -C "$work/source/head"
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
VLLM_OMNI_VERSION_OVERRIDE=0.33.0+g6f5a9976.validation VLLM_OMNI_TARGET_DEVICE=cuda \
uv pip install --system --prerelease=allow --constraint "$work/evidence/runtime-constraints.txt" \
 --override "$work/evidence/runtime-overrides.txt" --constraint "$work/bundle/validated-constraints.txt" \
 -e "$work/source/head" 'nvidia-cudnn-frontend==1.31.0' 'diffusers==0.40.0' 'transformers==5.19.0' \
 'accelerate==1.12.0' 'av==19.0.2' 'lpips==0.1.4' 'scikit-image==0.25.2' \
 'huggingface-hub==1.33.0' 'numpy==2.2.6' 'pillow==13.0.0' 'pytest==9.2.0' 'pytest-asyncio==2.0.0a1' 'pytest-xdist==3.8.0' 'pytest-mock==3.17.0'
mkdir -p "$work/source/sana"
curl -fL --retry 3 https://codeload.github.com/NVlabs/Sana/tar.gz/f4035e0a976d828ed41c7b0b799e9712b47cd60d | tar xz --strip-components=1 -C "$work/source/sana"
uv pip install --system --no-deps "$work/source/sana/techniques/sparse_backends"
uv pip freeze > "$work/evidence/installed-environment.txt"

python3 "$work/bundle/download_models.py" "$work" > "$work/evidence/model-download.log" 2>&1
bash "$work/bundle/run_forward.sh" "$work"
