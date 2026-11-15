#!/usr/bin/env bash
# Execute inside the exact pinned image. All caches and outputs stay in $work.
set -euo pipefail
work=${1:-/work}
validation_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$work/evidence" "$work/source/base" "$work/source/candidate"
for file in generate_case.py run_matrix.py score_outputs.py matrix.json reproduce.sh download_models.py test_fp8_block_runtime.py; do
  if [[ "$validation_dir" != "$work" ]]; then cp "$validation_dir/$file" "$work/$file"; fi
done
export HF_HOME="$work/hf-cache" UV_CACHE_DIR="$work/uv-cache" XDG_CACHE_HOME="$work/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$work/torchinductor-cache" TRITON_CACHE_DIR="$work/triton-cache"
export TORCH_HOME="$work/torch-cache" HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DOWNLOAD_TIMEOUT=600
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv > "$work/evidence/hardware.csv"
nvidia-smi topo -m > "$work/evidence/topology.txt"
curl -fL --retry 3 https://codeload.github.com/vllm-project/vllm-omni/tar.gz/eb2077899989a07d0e7acb9cda3217734fdc8909 | tar xz --strip-components=1 -C "$work/source/base"
curl -fL --retry 3 https://codeload.github.com/lishunyang12/vllm-omni/tar.gz/a6a35c86b13db636a07173755b241544dff4c082 | tar xz --strip-components=1 -C "$work/source/candidate"
python3 - "$work" <<'PY'
from importlib.metadata import distributions, version
from pathlib import Path
import sys
work = Path(sys.argv[1])
keep={'vllm','torch','torchvision','torchaudio','triton','pytorch-triton','nixl','apache-tvm-ffi','tilelang','quack-kernels','humming-kernels','tokenspeed-mla'}
lines=[]
for dist in distributions():
    name=dist.metadata['Name'].lower().replace('_','-')
    if name in keep or name.startswith(('nvidia-','cuda-','flashinfer-')):
        lines.append(f'{name}=={dist.version}')
(work/'evidence/runtime-constraints.txt').write_text('\n'.join(sorted(lines))+'\n')
(work/'evidence/runtime-overrides.txt').write_text('\n'.join(f'{name}=={version(name)}' for name in ['triton','nvidia-nccl-cu13'])+'\n')
PY
VLLM_OMNI_VERSION_OVERRIDE=0.33.0+ga6a35c86.quality VLLM_OMNI_TARGET_DEVICE=cuda \
 uv pip install --system --prerelease=allow --constraint "$work/evidence/runtime-constraints.txt" \
 --override "$work/evidence/runtime-overrides.txt" \
 --constraint "$validation_dir/validated-constraints.txt" -e "$work/source/candidate" \
 'diffusers==0.40.0' 'transformers==5.19.0' 'accelerate==1.12.0' 'av==19.0.2' \
 'lpips==0.1.4' 'scikit-image==0.25.2' 'huggingface-hub==1.33.0' \
 'numpy==2.2.6' 'pillow==13.0.0' 'pytest==9.2.0'
uv pip freeze > "$work/evidence/installed-environment.txt"
python3 "$work/download_models.py" "$work"
bash "$work/reproduce.sh" "$work"
