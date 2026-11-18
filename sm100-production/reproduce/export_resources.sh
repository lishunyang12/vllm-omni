#!/bin/bash
set -Eeuo pipefail
export PYTHONPATH=/work/source/candidate/techniques/sparse_backends:/work/source/candidate/techniques/sparse_backends/benchmarks
export XDG_CACHE_HOME=/work/xdg-cache TRITON_CACHE_DIR=/work/triton-cache OMP_NUM_THREADS=4
root=/work/results/static-resources
mkdir -p "$root"
for variant in baseline candidate; do
  mkdir -p "$root/$variant"
  CUTE_DSL_KEEP=cubin,ptx CUTE_DSL_DUMP_DIR="$root/$variant" CUTE_DSL_DISABLE_FILE_CACHING=1 \
    python3 /work/scripts/profile_candidate.py --variant "$variant" \
      --output "$root/$variant-meta.json" > "$root/$variant-build.log" 2>&1
  python3 - "$root" "$variant" <<'PY'
from pathlib import Path
import subprocess,sys
root=Path(sys.argv[1]); variant=sys.argv[2]
files=sorted((root/variant).rglob('*.cubin'))
assert files, 'Expected compiler cubin artifacts'
reports=[]
for p in files:
    r=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(p)],check=True,capture_output=True,text=True)
    reports.append(f'FILE {p.name}\n{r.stdout}\n')
    s=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(p)],check=False,capture_output=True,text=True)
    (p.with_suffix('.sass')).write_text(s.stdout)
    if s.returncode: (p.with_suffix('.sass-error.txt')).write_text(s.stderr)
(root/f'{variant}-resources.txt').write_text('\n'.join(reports))
PY
done
