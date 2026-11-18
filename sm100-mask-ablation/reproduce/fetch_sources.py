"""Fetch public pinned sources, then materialize isolated design candidates."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

work = Path(sys.argv[1]).resolve()
bundle = Path(__file__).resolve().parent
pins = json.loads((bundle / 'pins.json').read_text())
work.mkdir(parents=True, exist_ok=True)
(work / 'pins.json').write_text(json.dumps(pins, indent=2) + '\n')
for name, revision in pins.items():
    target = work / 'source' / name
    if target.exists():
        raise RuntimeError(f'Refusing to overwrite existing source: {target}')
    url = f'https://codeload.github.com/lishunyang12/Sana/tar.gz/{revision}'
    with urllib.request.urlopen(url) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()))
    target.mkdir(parents=True)
    for member in archive.getmembers():
        relative = Path(member.name).parts[1:]
        if relative[:2] != ('techniques', 'sparse_backends') or not member.isfile():
            continue
        if '..' in relative:
            raise RuntimeError('Unsafe archive path')
        path = target.joinpath(*relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.extractfile(member).read())
for name in ('protected-q', 'p-store-x16', 'protected-q-contiguous', 'smem-p-serial', 'smem-p-overlap', 'smem-p-overlap-publish-first', 'smem-p-vector'):
    target = work / 'source' / name
    shutil.copytree(work / 'source' / 'candidate', target)
    patch = bundle / 'patches' / f'{name}.patch'
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=target, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=target, check=True)
print('Pinned sources and experimental candidates prepared:', work)
