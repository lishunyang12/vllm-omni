"""Single-source/single-process diagnostic; no package aliases or shared JIT state."""
import hashlib
import json
import os
from pathlib import Path

import torch
from sol_attn import sol_attn, interface

out = Path(os.environ['H3_PROBE_OUTPUT'])
out.mkdir(parents=True, exist_ok=True)
generator = torch.Generator(device='cuda').manual_seed(71)
qkv = tuple(torch.randn(1,8193,1,128,device='cuda',dtype=torch.bfloat16,generator=generator) for _ in range(3))
def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
with torch.inference_mode():
    before=[digest(t) for t in qkv]
    outputs=[]
    for i in range(4):
        value=sol_attn(*qkv,tau=1.0,thresh_type='diag',exact_block_mask=None)
        torch.cuda.synchronize()
        outputs.append(value.cpu())
    assert before==[digest(t) for t in qkv]
    for value in outputs[1:]:
        torch.testing.assert_close(value.view(torch.uint8),outputs[0].view(torch.uint8),atol=0,rtol=0)
    torch.save(outputs[0],out/'output.pt')
    assert torch.isfinite(outputs[0]).all()
report=dict(input_sha256=before,output_sha256=digest(outputs[0]),repeats=4,stable=True,input_immutable=True,
    source=str(interface.__file__),torch=torch.__version__,capability=list(torch.cuda.get_device_capability()),
    compiled_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.suffix in ('.ptx','.cubin')})
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
