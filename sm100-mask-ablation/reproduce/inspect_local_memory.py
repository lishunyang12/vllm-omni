"""Static SASS inspection; counts are not runtime instruction frequencies."""
from pathlib import Path
import re
import subprocess

root = Path('/work/results/design-resources')
for name in ('candidate', 'smem-p-serial'):
    path = next((root/name).glob('*.cubin'))
    result = subprocess.run(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(path)], check=False, capture_output=True, text=True)
    if result.returncode:
        (root/f'{name}-sass-unavailable.txt').write_text(result.stderr)
        print(name, 'disassembly unavailable; see saved error')
        continue
    lines = [line for line in result.stdout.splitlines() if re.search(r'\b(?:LDL|STL)(?:\.|\s)', line)]
    (root/f'{name}-local-memory-sass.txt').write_text('Static local-memory instructions (not dynamic execution counts)\n'+'\n'.join(lines)+'\n')
    print(name, 'static LDL/STL count', len(lines))
