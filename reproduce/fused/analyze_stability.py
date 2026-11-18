"""Validate paired inputs and outputs after all timing processes exit."""
import json
from pathlib import Path

import torch
from analyze_outputs import metrics

torch.set_num_threads(4)
root = Path('/work/fused/results')
report = {'paired_timing': json.loads((root/'ref2va-paired/summary.json').read_text()),
          'dense_late_timing': json.loads((root/'ref2va-dense-late/summary.json').read_text()),
          'inputs_match': {}, 'tensor_checks': {}}
for rank in range(2):
    base = json.loads((root/'ref2va-reference'/f'rank-{rank}.json').read_text())
    for name in ['ref2va-paired', 'ref2va-dense-late']:
        current = json.loads((root/name/f'rank-{rank}.json').read_text())
        report['inputs_match'][f'{name}/rank{rank}'] = current['input'] == base['input']
    expected = torch.load(root/'ref2va-reference'/f'output-first-rank-{rank}.pt', weights_only=True)
    for variant in ['reference', 'final']:
        for sample in ['first','last']:
            actual = torch.load(root/'ref2va-paired'/f'output-{variant}-{sample}-rank-{rank}.pt', weights_only=True)
            report['tensor_checks'][f'paired-{variant}-{sample}/rank{rank}'] = [metrics(a,b) for a,b in zip(actual,expected)]
    expected = torch.load(root/'ref2va-dense'/f'output-first-rank-{rank}.pt', weights_only=True)
    actual = torch.load(root/'ref2va-dense-late'/f'output-first-rank-{rank}.pt', weights_only=True)
    report['tensor_checks'][f'dense-late/rank{rank}'] = [metrics(a,b) for a,b in zip(actual,expected)]
assert all(report['inputs_match'].values())
assert all(m['bitwise_equal'] for row in report['tensor_checks'].values() for m in row)
(root.parent/'stability-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print('PAIRED_AND_DENSE_OUTPUTS_BITWISE_EQUAL')
