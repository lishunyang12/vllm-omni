"""Refuse experimental performance runs without matching full correctness evidence."""
import hashlib
import json
from pathlib import Path
import sys

raise RuntimeError('Performance promotion is blocked by the independent #594 oracle failure; see CORRECTNESS_BLOCKER.txt')

work = Path(sys.argv[1])
variant = sys.argv[2]
report = json.loads((work/'results/design-validation'/f'{variant}-full.json').read_text())
assert report['complete'] and len(report['cases']) == 73, 'Incomplete/failed full correctness suite'
for side, source in [('baseline', 'candidate'), ('candidate', variant)]:
    root = work/'source'/source/'techniques/sparse_backends/sol_attn'
    for name, expected in report['source_sha256'][side].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, f'Source changed: {source}/{name}'
stress = json.loads((work/'results/design-validation'/f'{variant}-stress.json').read_text())
assert len(stress['cases']) == 12 and all(c['bitwise_equal'] for c in stress['cases'])
for side, source in [('baseline', 'candidate'), ('candidate', variant)]:
    root = work/'source'/source/'techniques/sparse_backends/sol_attn'
    for name, expected in stress[f'{side}_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, f'Stress source changed: {source}/{name}'
print('Matched 12 + 73 correctness checks:', variant)
