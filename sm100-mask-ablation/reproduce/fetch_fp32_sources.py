"""Prepare immutable-source SOL comparisons with the same explicit FP32 pooling fix."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

root = Path(sys.argv[1]).resolve()
bundle = Path(__file__).resolve().parent
pins = {'extension': '63b4e65e9db1dfea45cc7d394902145c2f520821',
        'candidate': '150114b6f0079bcd479c6c87bd0b15174581d984'}
names = list(pins) + ['extension-fp32', 'candidate-fp32', 'protected-q-contiguous-fp32']
assert all(not (root / 'source' / name).exists() for name in names), 'Refusing to overwrite existing sources'
for name, revision in pins.items():
    destination = root / 'source' / name
    destination.mkdir(parents=True)
    with tempfile.TemporaryFile() as stream:
        url = f'https://codeload.github.com/NVlabs/Sana/tar.gz/{revision}'
        with urllib.request.urlopen(url, timeout=120) as response:
            shutil.copyfileobj(response, stream)
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode='r:gz') as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                assert not member.name.startswith('/') and '..' not in parts
                relative = parts[1:]
                if relative[:2] != ('techniques', 'sparse_backends') or not member.isfile():
                    continue
                path = destination.joinpath(*relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    fixed = root / 'source' / (name + '-fp32')
    shutil.copytree(destination, fixed)
    patch = bundle / 'patches/pool-fp32.patch'
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=fixed, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=fixed, check=True)
candidate = root / 'source/protected-q-contiguous-fp32'
shutil.copytree(root / 'source/candidate-fp32', candidate)
patch = bundle / 'patches/protected-q-contiguous-fp32.patch'
subprocess.run(['git', 'apply', '--check', str(patch)], cwd=candidate, check=True)
subprocess.run(['git', 'apply', str(patch)], cwd=candidate, check=True)
expected = json.loads((bundle / 'fp32-source-manifest.json').read_text())
for name, files in expected.items():
    package = root / 'source' / name / 'techniques/sparse_backends/sol_attn'
    for path, digest in files.items():
        assert hashlib.sha256((package / path).read_bytes()).hexdigest() == digest, (name, path)
(root / 'fp32-source-check.json').write_text(json.dumps({'pins': pins, 'verified': True, 'files': {n: len(v) for n, v in expected.items()}}, indent=2) + '\n')
print('Prepared and verified source hashes:', root / 'fp32-source-check.json')
