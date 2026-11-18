#!/bin/bash
set -Eeuo pipefail
work=${1:-/work}
mkdir -p "$work/source/sana" "$work/study/environment"
curl -fL --retry 2 https://codeload.github.com/lishunyang12/Sana/tar.gz/088897e0c6a9cb970f00b43ab2177c3be7336b36 \
  | tar xz --strip-components=1 -C "$work/source/sana"
uv pip install --system --no-deps "$work/source/sana/techniques/sparse_backends" 'numpy==2.2.6'
uv pip freeze > "$work/study/environment/installed.txt"
nvidia-smi -q > "$work/study/environment/gpu.txt"
