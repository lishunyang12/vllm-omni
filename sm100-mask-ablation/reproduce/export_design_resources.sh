#!/bin/bash
set -Eeuo pipefail
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
root=/work/results/design-resources
mkdir -p "$root"
for variant in candidate smem-p-serial; do
  export PYTHONPATH="/work/source/$variant/techniques/sparse_backends:/work/scripts"
  mkdir -p "$root/$variant"
  CUTE_DSL_KEEP=cubin,ptx CUTE_DSL_DUMP_DIR="$root/$variant" CUTE_DSL_DISABLE_FILE_CACHING=1 \
    python3 /work/scripts/resource_probe.py --input /work/captures/qkv-layer-00-rank-0.pt \
    --output "$root/$variant-meta.json" > "$root/$variant-build.log" 2>&1
  python3 - "$root" "$variant" <<'PY'
from pathlib import Path
import subprocess,sys
root=Path(sys.argv[1]); variant=sys.argv[2]
files=sorted((root/variant).rglob('*.cubin'))
assert files, 'Expected fresh compiler cubin artifacts'
reports=[]
for path in files:
    result=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(path)],check=True,capture_output=True,text=True)
    reports.append(f'FILE {path.name}\n{result.stdout}\n')
(root/f'{variant}-resources.txt').write_text('\n'.join(reports))
PY
done
