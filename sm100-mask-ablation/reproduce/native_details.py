"""Retain native prepare results/LSE without modifying the kernel source."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import torch
from sol_attn import sol_attn, preprocess, interface
out=Path(os.environ['H3_DETAILS_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
tokens=int(os.environ.get('H3_DETAILS_TOKENS','8193'))
generator=torch.Generator(device='cuda').manual_seed(71)
qkv=tuple(torch.randn(1,tokens,1,128,device='cuda',dtype=torch.bfloat16,generator=generator) for _ in range(3))
def sha(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
old_prepare=preprocess.prepare;old_empty=torch.empty;prepared=[];lses=[]
def capture_prepare(*args,**kwargs):
    values=old_prepare(*args,**kwargs);prepared.append(values);return values
def capture_empty(*args,**kwargs):
    value=old_empty(*args,**kwargs)
    if tuple(value.shape)==(1,tokens,1) and value.dtype==torch.float32 and value.device.type=='cuda':lses.append(value)
    return value
preprocess.prepare=capture_prepare;torch.empty=capture_empty
try:
    with torch.inference_mode():
        result=sol_attn(*qkv,tau=1.0,thresh_type='diag')
        torch.cuda.synchronize()
finally:
    preprocess.prepare=old_prepare;torch.empty=old_empty
assert len(prepared)==1 and len(lses)==1,(len(prepared),len(lses))
assert torch.isfinite(result).all() and torch.isfinite(lses[0]).all()
torch.save(dict(output=result.cpu(),lse=lses[0].cpu(),prepared=[x.cpu() for x in prepared[0]]),out/'details.pt')
report=dict(tokens=tokens,source=str(interface.__file__),input_sha256=[sha(t) for t in qkv],output_sha256=sha(result),lse_sha256=sha(lses[0]),prepared=[dict(shape=list(x.shape),dtype=str(x.dtype),sha256=sha(x)) for x in prepared[0]])
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
