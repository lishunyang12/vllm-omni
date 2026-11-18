"""Summarize isolated single-change ablations against the oracle-passing #593."""
import json
from pathlib import Path
import torch
root=Path('/work/results')
baseline=torch.load(root/'independent-process/extension/output.pt',weights_only=True,map_location='cpu')
expected=json.loads((root/'independent-process/extension/report.json').read_text())['input_sha256']
result={}
for d in sorted((root/'pr594-ablation').iterdir()):
    if not d.is_dir():continue
    report=json.loads((d/'report.json').read_text());assert report['input_sha256']==expected
    value=torch.load(d/'output.pt',weights_only=True,map_location='cpu')
    delta=value.float()-baseline.float()
    result[d.name]=dict(bitwise_equal_to_593=bool(torch.equal(value.view(torch.uint8),baseline.view(torch.uint8))),max_abs_to_593=float(delta.abs().max()),rmse_to_593=float(delta.square().mean().sqrt()),output_sha256=report['output_sha256'])
(root/'pr594-ablation/comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
