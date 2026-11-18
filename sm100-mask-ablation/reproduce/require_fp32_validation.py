"""Require the complete matched-source correctness record before corrected timings."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
report = json.loads((root / 'results/fp32-validation/full.json').read_text())
assert report['complete'] and len(report['cases']) == 73
assert all(c['passed'] for c in report['cases'])
for side, source in [('baseline', 'extension-fp32'), ('candidate', 'candidate-fp32')]:
    package = root / 'source' / source / 'techniques/sparse_backends/sol_attn'
    for name, digest in report['source_sha256'][side].items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest, (source, name)
for name in ('pooling.exit', 'full.exit'):
    assert (root / 'results/fp32-validation' / name).read_text().strip() == '0'
comparison = json.loads((root / 'results/fp32-matrix/comparison.json').read_text())
assert comparison['cases'] == 29
assert not any(comparison[k] for k in ('input_mismatch', 'prepared_mismatch', 'output_mismatch'))
print('Matched source hashes and complete FP32 correctness evidence verified')
