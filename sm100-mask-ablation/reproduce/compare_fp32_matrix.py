"""Check independently produced FP32-pooling matrix input, prepared and output hashes."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path)
a = p.parse_args()
old = json.loads((a.root / 'extension-fp32/report.json').read_text())
new = json.loads((a.root / 'candidate-fp32/report.json').read_text())
assert old['complete'] and new['complete'] and len(old['cases']) == len(new['cases']) == 29
report = dict(cases=29, input_mismatch=[], prepared_mismatch=[], output_mismatch=[])
for x, y in zip(old['cases'], new['cases'], strict=True):
    assert x['name'] == y['name']
    for item in (x, y):
        assert item['repeat_stable'] and item['input_immutable']
        if item.get('mode') == 'graph':
            assert item['no_recompile']
    for field, target in [('input_sha256', 'input_mismatch'), ('prepared_sha256', 'prepared_mismatch'), ('output_sha256', 'output_mismatch')]:
        if x[field] != y[field]:
            report[target].append(x['name'])
(a.root / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
assert not any(report[k] for k in ('input_mismatch', 'prepared_mismatch', 'output_mismatch')), report
print('All29independent input/prepared/output hashes match')
