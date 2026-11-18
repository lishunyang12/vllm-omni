"""One native package per process; immutable-input multi-shape regression matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from sol_attn import sol_attn, interface, preprocess
native_prepare=preprocess.prepare
last_prepared=[]
def capture_prepared(*args,**kwargs):
    values=native_prepare(*args,**kwargs)
    last_prepared[:]=values
    return values
preprocess.prepare=capture_prepared

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def sha(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
cases=[(257,2,2,71,'disjoint',1.5,'diag'),(513,1,2,71,'disjoint',1.5,'diag')]
cases += [(t,1,1,71,m,1.0,'diag') for t in (8192,8193,8256) for m in ('none','disjoint','all')]
cases += [(8193,1,1,s,m,1.0,'diag') for s in (13,101) for m in ('none','disjoint')]
cases += [(t,1,1,71,m,1.0,'diag') for t in (16385,32769) for m in ('none','disjoint')]
cases += [(513,1,2,71,'disjoint',1.5,'exact'),(8193,1,1,71,'none',1.0,'exact')]
root=Path(interface.__file__).parent
report=dict(complete=False,torch=torch.__version__,capability=list(torch.cuda.get_device_capability()),source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*.py')},cases=[])
def save():
    tmp=a.output/'report.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output/'report.json')
with torch.inference_mode():
    for index,(tokens,batch,heads,seed,kind,tau,threshold) in enumerate(cases):
        generator=torch.Generator(device='cuda').manual_seed(seed)
        qkv=tuple(torch.randn(batch,tokens,heads,128,device='cuda',dtype=torch.bfloat16,generator=generator) for _ in range(3))
        mask=None
        if kind!='none':
            mask=torch.zeros((tokens+63)//64,device='cuda',dtype=torch.bool)
            if kind=='all':mask.fill_(True)
            else:
                mask[0]=mask[-1]=True
                if len(mask)>5:mask[4]=True
        tensors=qkv+(() if mask is None else (mask,))
        before=[sha(t) for t in tensors]
        outputs=[]
        for repeat in range(2):
            value=sol_attn(*qkv,tau=tau,thresh_type=threshold,exact_block_mask=mask)
            torch.cuda.synchronize();outputs.append(value.cpu())
        assert before==[sha(t) for t in tensors]
        torch.testing.assert_close(outputs[0].view(torch.uint8),outputs[1].view(torch.uint8),atol=0,rtol=0)
        assert torch.isfinite(outputs[0]).all()
        name=f'case-{index:02d}'
        torch.save(outputs[0],a.output/f'{name}.pt')
        report['cases'].append(dict(name=name,tokens=tokens,batch=batch,heads=heads,seed=seed,mask=kind,tau=tau,threshold=threshold,input_sha256=before,output_sha256=sha(outputs[0]),prepared_sha256=[sha(t) for t in last_prepared],repeat_stable=True,input_immutable=True))
        save();print(name,tokens,batch,heads,seed,kind,threshold,report['cases'][-1]['output_sha256'],flush=True)
# Replay both threshold specializations while mutating mask/input values.
with torch.inference_mode():
    for threshold in ('diag','exact'):
        generator=torch.Generator(device='cuda').manual_seed(71)
        qkv=tuple(torch.randn(2,513,3,128,device='cuda',dtype=torch.bfloat16,generator=generator) for _ in range(3))
        mask=torch.zeros(9,device='cuda',dtype=torch.bool)
        options=dict(tau=1.5,thresh_type=threshold,exact_block_mask=mask)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            sol_attn(*qkv,**options);stream.synchronize()
            compiled=set(interface._compiled)
            with torch.cuda.graph(graph,stream=stream):
                captured=sol_attn(*qkv,**options)
        torch.cuda.current_stream().wait_stream(stream)
        for update in ('zero','all','disjoint','zero_again'):
            mask.zero_()
            if update=='all':mask.fill_(True)
            if update=='disjoint':
                mask[0]=mask[4]=mask[-1]=True
                qkv[0].mul_(0.85)
            torch.cuda.synchronize()
            before=[sha(t) for t in (*qkv,mask)]
            graph.replay();torch.cuda.synchronize();first=captured.cpu()
            graph.replay();torch.cuda.synchronize();second=captured.cpu()
            assert before==[sha(t) for t in (*qkv,mask)]
            assert set(interface._compiled)==compiled
            assert torch.equal(first.view(torch.uint8),second.view(torch.uint8))
            assert torch.isfinite(first).all()
            name=f'graph-{threshold}-{update}'
            torch.save(first,a.output/f'{name}.pt')
            report['cases'].append(dict(name=name,tokens=513,batch=2,heads=3,seed=71,mask=update,tau=1.5,threshold=threshold,mode='graph',input_sha256=before,output_sha256=sha(first),prepared_sha256=[sha(t) for t in last_prepared],repeat_stable=True,input_immutable=True,no_recompile=True))
            save();print(name,report['cases'][-1]['output_sha256'],flush=True)
report['complete']=True;save()
