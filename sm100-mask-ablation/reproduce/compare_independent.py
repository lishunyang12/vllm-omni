"""Compare separately compiled kernels to each other and the independent oracle."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,'/work/source/candidate/techniques/sparse_backends/tests')
from test_exact_block_mask import hybrid_oracle
root=Path('/work/results/independent-process')
names=('candidate','extension','smem-p-serial','smem-p-vector')
reports={n:json.loads((root/n/'report.json').read_text()) for n in names}
assert len({tuple(d['input_sha256']) for d in reports.values()})==1
values={n:torch.load(root/n/'output.pt',map_location='cuda',weights_only=True) for n in names}
generator=torch.Generator(device='cuda').manual_seed(71)
qkv=tuple(torch.randn(1,8193,1,128,device='cuda',dtype=torch.bfloat16,generator=generator) for _ in range(3))
with torch.inference_mode():
    oracle=hybrid_oracle(*qkv,torch.zeros(129,dtype=torch.bool,device='cuda'),1.0,'diag')
    result={}
    for name,value in values.items():
        delta=value.float()-values['candidate'].float();od=value.float()-oracle
        try:
            torch.testing.assert_close(value.float(),oracle,atol=.02,rtol=.02)
            passes=True
        except AssertionError:
            passes=False
        result[name]=dict(byte_mismatch_to_candidate=int((value.view(torch.uint8)!=values['candidate'].view(torch.uint8)).sum().item()),max_abs_to_candidate=float(delta.abs().max()),rmse_to_candidate=float(delta.square().mean().sqrt()),max_abs_to_oracle=float(od.abs().max()),rmse_to_oracle=float(od.square().mean().sqrt()),oracle_pass=passes)
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
