import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,'/work/source/extension/techniques/sparse_backends/tests')
from test_exact_block_mask import hybrid_oracle
root=Path('/work/results/independent-matrix')
meta=json.loads((root/'extension/report.json').read_text())
rows=[]
with torch.inference_mode():
    for entry in [x for x in meta['cases'] if x['name'] in ('case-00','case-01','case-02','case-03','case-05')]:
        generator=torch.Generator(device='cuda').manual_seed(entry['seed'])
        qkv=tuple(torch.randn(entry['batch'],entry['tokens'],entry['heads'],128,device='cuda',dtype=torch.bfloat16,generator=generator) for _ in range(3))
        mask=torch.zeros((entry['tokens']+63)//64,device='cuda',dtype=torch.bool)
        if entry['mask']=='disjoint':
            mask[0]=mask[-1]=True
            if len(mask)>5:mask[4]=True
        expected=hybrid_oracle(*qkv,mask,entry['tau'],entry['threshold'])
        for variant in ('extension','alpha-only'):
            value=torch.load(root/variant/f"{entry['name']}.pt",map_location='cuda',weights_only=True)
            delta=value.float()-expected
            try:
                torch.testing.assert_close(value.float(),expected,atol=.02,rtol=.02);passes=True
            except AssertionError:passes=False
            row=dict(name=entry['name'],variant=variant,oracle_pass=passes,max_abs=float(delta.abs().max()),rmse=float(delta.square().mean().sqrt()))
            rows.append(row);print(row,flush=True)
(root/'oracle.json').write_text(json.dumps(rows,indent=2)+'\n')
