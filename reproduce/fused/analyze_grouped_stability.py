"""Check the published candidate across standalone and fresh paired runs."""
import json
from pathlib import Path

import torch
from analyze_outputs import metrics

torch.set_num_threads(4)
root = Path('/work/fused/results')
report = {'standalone': json.loads((root/'ref2va-grouped/summary.json').read_text()),
          'paired': json.loads((root/'ref2va-paired-grouped/summary.json').read_text()),
          'inputs_match': {}, 'tensor_checks': {}}
for rank in range(2):
    base = json.loads((root/'ref2va-reference'/f'rank-{rank}.json').read_text())
    for name in ['ref2va-grouped','ref2va-paired-grouped']:
        current = json.loads((root/name/f'rank-{rank}.json').read_text())
        report['inputs_match'][f'{name}/rank{rank}'] = current['input'] == base['input']
    expected = torch.load(root/'ref2va-reference'/f'output-first-rank-{rank}.pt', weights_only=True)
    for variant in ['reference','grouped']:
        for sample in ['first','last']:
            actual = torch.load(root/'ref2va-paired-grouped'/f'output-{variant}-{sample}-rank-{rank}.pt', weights_only=True)
            assert len(actual) == len(expected) == 2
            report['tensor_checks'][f'paired-{variant}-{sample}/rank{rank}'] = [metrics(a,b) for a,b in zip(actual,expected)]
    for sample in ['first','last']:
        actual = torch.load(root/'ref2va-grouped'/f'output-{sample}-rank-{rank}.pt', weights_only=True)
        assert len(actual) == len(expected) == 2
        report['tensor_checks'][f'standalone-{sample}/rank{rank}'] = [metrics(a,b) for a,b in zip(actual,expected)]
assert all(report['inputs_match'].values())
assert all(m['bitwise_equal'] for row in report['tensor_checks'].values() for m in row)
(root.parent/'grouped-stability-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print('GROUPED_STANDALONE_AND_PAIRED_OUTPUTS_BITWISE_EQUAL')
