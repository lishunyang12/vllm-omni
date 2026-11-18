"""Force every native reduction configuration against a float64 BF16 reference."""
import hashlib
import json
import os
from pathlib import Path
import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor
from sol_attn import preprocess as p
out=Path(os.environ.get('H3_SWEEP_OUTPUT','/work/results/preprocess-sweep'));out.mkdir(parents=True,exist_ok=True)
def sha(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
reports=[]
with torch.inference_mode():
    for tokens in (513,8192,8193,8256):
        g=torch.Generator(device='cuda').manual_seed(71)
        q,k,v=[torch.randn(1,tokens,1,128,device='cuda',dtype=torch.bfloat16,generator=g) for _ in range(3)]
        n=triton.cdiv(tokens,64)
        padded=torch.zeros((1,n*64,1,128),dtype=torch.float64,device='cuda');padded[:,:tokens]=k.double()
        lengths=(tokens-64*torch.arange(n,device='cuda')).clamp(0,64).reshape(1,n,1,1).double()
        exact_sum=padded.reshape(1,n,64,1,128).sum(2)
        exact_mean=exact_sum/lengths
        reference=exact_mean.to(torch.bfloat16);sum_ref=exact_sum.to(torch.bfloat16)
        desc=TensorDescriptor.from_tensor(k,[1,64,1,128])
        native=p.prepare(q,k,v,tau=1.0,scale=128**-.5,thresh_type='diag')
        data=dict(tokens=tokens,input_sha256=[sha(x) for x in (q,k,v)],reference_sha256=sha(reference),native_kc_sha256=sha(native[0]),native_best_config=str(p._reduce_kc_kernel.best_config),configs=[])
        for warps in (4,8):
            for stages in (1,2,3,4):
                kc=torch.empty_like(reference);ks=torch.empty_like(reference)
                kernel=p._reduce_kc_kernel.fn[(1,n,1)](desc,kc,tokens,1,n,128,64,128,num_warps=warps,num_stages=stages)
                p._reduce_vc_kernel.fn[(1,n,1)](desc,ks,tokens,1,n,128,64,128,num_warps=warps,num_stages=stages)
                torch.cuda.synchronize()
                different=kc!=reference;coords=different.nonzero().cpu().tolist()
                row=dict(warps=warps,stages=stages,kc_sha256=sha(kc),mismatched_elements=len(coords),full_block_mismatches=sum(c[1]<tokens//64 for c in coords),tail_block_mismatches=sum(c[1]>=tokens//64 for c in coords),max_abs=float((kc.float()-reference.float()).abs().max()),sum_sha256=sha(ks),sum_mismatches=int((ks!=sum_ref).sum()),examples=[dict(index=c,actual=float(kc[tuple(c)]),expected=float(reference[tuple(c)]),exact_mean=float(exact_mean[tuple(c)]),exact_sum=float(exact_sum[tuple(c)])) for c in coords[:20]])
                data['configs'].append(row)
                folder=out/f't{tokens}-w{warps}-s{stages}';folder.mkdir(exist_ok=True)
                for kind in ('ttir','ttgir','ptx'):
                    if kind in kernel.asm:(folder/f'kernel.{kind}').write_text(kernel.asm[kind])
                print(tokens,warps,stages,'KC mismatch',len(coords),'sum mismatch',row['sum_mismatches'],row['kc_sha256'],flush=True)
        reports.append(data)
        (out/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
