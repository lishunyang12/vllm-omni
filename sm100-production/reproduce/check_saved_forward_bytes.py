"""Verify retained first/last original and candidate DiT outputs on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
a=p.parse_args()
checks=[]
for rank in (0,1):
    for point in ('first','last'):
        old=torch.load(a.root/f'output-baseline-{point}-rank-{rank}.pt',map_location='cpu',weights_only=True)
        new=torch.load(a.root/f'output-candidate-{point}-rank-{rank}.pt',map_location='cpu',weights_only=True)
        assert len(old)==len(new)
        tensors=[]
        for x,y in zip(old,new):
            assert x.dtype==y.dtype and x.shape==y.shape
            xb=x.contiguous().view(torch.uint8);yb=y.contiguous().view(torch.uint8)
            assert torch.equal(xb,yb)
            tensors.append(dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(xb.numpy().tobytes()).hexdigest()))
        checks.append(dict(rank=rank,point=point,byte_equal=True,tensors=tensors))
(a.root/'saved-output-byte-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('SAVED_FORWARD_BYTE_CHECKS_PASSED',len(checks),flush=True)
