import json
from pathlib import Path
import torch
r=Path('/work/results/independent-matrix')
a=json.loads((r/'extension/report.json').read_text());b=json.loads((r/'alpha-only/report.json').read_text())
assert a['complete'] and b['complete'] and len(a['cases'])==len(b['cases'])==29
rows=[]
for x,y in zip(a['cases'],b['cases']):
    assert x['name']==y['name'] and x['input_sha256']==y['input_sha256']
    old=torch.load(r/'extension'/f"{x['name']}.pt",map_location='cpu',weights_only=True)
    new=torch.load(r/'alpha-only'/f"{y['name']}.pt",map_location='cpu',weights_only=True)
    delta=new.float()-old.float()
    row=dict(name=x['name'],tokens=x['tokens'],mask=x['mask'],threshold=x['threshold'],bitwise_equal=torch.equal(old.view(torch.uint8),new.view(torch.uint8)),max_abs=float(delta.abs().max()),rmse=float(delta.square().mean().sqrt()))
    rows.append(row)
    if not row['bitwise_equal']:print(row,flush=True)
(r/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
