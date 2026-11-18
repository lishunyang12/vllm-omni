"""Untimed byte comparisons for the same synthetic/H3 inputs used in timings."""
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from sol_attn import sol_attn
from benchmark_sm100 import assert_bitwise_equal, baseline, tensors

root = Path('/work/results')
report = dict(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()), cases=[])
cases = [(1,257,2,'empty',1.0), (2,1024,3,'all',1.0), (1,4096,2,'all',1.0),
         (2,4097,3,'ends',1.0), (1,16384,8,'empty',1.5), (1,16384,8,'empty',1e6),
         (1,32768,8,'ends',1.0), (1,32768,28,'ends',1.0)]

@torch.inference_mode()
def check(name, values, old):
    q,k,v,mask,options = values
    expected=old(q,k,v,**options,exact_block_mask=mask)
    actual=sol_attn(q,k,v,**options,exact_block_mask=mask)
    assert_bitwise_equal(actual,expected)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured=sol_attn(q,k,v,**options,exact_block_mask=mask)
    graph.replay()
    torch.cuda.synchronize()
    assert_bitwise_equal(captured,expected)
    report['cases'].append(dict(case=name,shape=list(q.shape),eager_byte_equal=True,graph_byte_equal=True))
    (root/'strict-byte-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('BYTE_EQUAL',name,flush=True)

with tempfile.TemporaryDirectory(prefix='strict-sol-baseline-') as directory:
    old, hashes=baseline(Path('/work/source/baseline/techniques/sparse_backends'),Path(directory))
    report['baseline_sha256']=hashes
    for case in cases:check(str(case),tensors(case),old)
    for layer in ('00','12','25','49'):
        data=torch.load(f'/work/captures/qkv-layer-{layer}-rank-0.pt',map_location='cpu',weights_only=True)
        for name in ('q','k','v','exact_block_mask'):
            t=data[name]
            if t is not None:assert hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==data['tensor_sha256'][name]
        values=[data[n][:,:,:28].contiguous().cuda() for n in ('q','k','v')]
        values += [data['exact_block_mask'].cuda(),data['options']]
        check('h3-layer-'+layer,values,old)
print('STRICT_BYTE_VALIDATION_PASSED',len(report['cases']),flush=True)
