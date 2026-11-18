"""Reject contiguous-load timing unless source-matched full and independent gates pass."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
directory = root / 'results/contiguous-fp32-validation'
full = json.loads((directory / 'full.json').read_text())
assert full['complete'] and len(full['cases']) == 73 and all(c['passed'] for c in full['cases'])
for name in ('full', 'matrix'):
    assert (directory / (name + '.exit')).read_text().strip() == '0'
for side, source in [('baseline', 'candidate-fp32'), ('candidate', 'protected-q-contiguous-fp32')]:
    package = root / 'source' / source / 'techniques/sparse_backends/sol_attn'
    for name, digest in full['source_sha256'][side].items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest, (source, name)
old = json.loads((root / 'results/fp32-matrix/candidate-fp32/report.json').read_text())
new = json.loads((directory / 'matrix/report.json').read_text())
assert old['complete'] and new['complete'] and len(old['cases']) == len(new['cases']) == 29
comparison = dict(cases=29, input_mismatch=[], prepared_mismatch=[], output_mismatch=[])
for a, b in zip(old['cases'], new['cases'], strict=True):
    assert a['name'] == b['name'] and b['repeat_stable'] and b['input_immutable']
    for field, key in [('input_sha256', 'input_mismatch'), ('prepared_sha256', 'prepared_mismatch'), ('output_sha256', 'output_mismatch')]:
        if a[field] != b[field]:
            comparison[key].append(a['name'])
    if b.get('mode') == 'graph':
        assert b['no_recompile']
(directory / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
assert not any(comparison[k] for k in ('input_mismatch', 'prepared_mismatch', 'output_mismatch')), comparison
print('Contiguous candidate: source-matched73checks and29independent cases passed')
