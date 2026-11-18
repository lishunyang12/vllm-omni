"""Check whether query chunking alone preserves the eager reference exactly."""
import ast
import json
from pathlib import Path
import statistics
import time

import torch

source = Path('/work/source/head/vllm_omni/diffusion/attention/ops/sol_attn.py')
node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == '_merge_sol_bsa_residual')
body = ast.unparse(node)
data = torch.load('/work/profile/results/ref2va-SOL_ATTN-capture/residual-input-rank0.pt', weights_only=True)
args = [t.cuda() if isinstance(t, torch.Tensor) else t for t in data['args']]
result = dict(gpu=torch.cuda.get_device_name(), allow_tf32=torch.backends.cuda.matmul.allow_tf32, runs={})
reference = None
with torch.inference_mode():
    for chunk in [256, 512, 1024, 2048, 4096]:
        text = body.replace('range(0, query.shape[1], 256)', f'range(0, query.shape[1], {chunk})')
        text = text.replace('start + 256', f'start + {chunk}')
        ns = {'torch': torch}
        exec(compile(text, str(source), 'exec'), ns)
        fn = ns['_merge_sol_bsa_residual']
        samples = []
        for i in range(12):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            output = fn(*args)
            torch.cuda.synchronize()
            elapsed = 1000 * (time.perf_counter() - start)
            if i >= 3:
                samples.append(elapsed)
        if reference is None:
            reference = output
        diff = output.float() - reference.float()
        row = dict(median_ms=statistics.median(samples), samples_ms=samples,
                   bitwise_equal=bool(torch.equal(output, reference)),
                   max_abs=diff.abs().max().item(), relative_l2=(diff.norm()/reference.float().norm()).item(),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated())
        result['runs'][str(chunk)] = row
        print(chunk, json.dumps(row), flush=True)
        Path('/work/fused/chunk-ablation.json').write_text(json.dumps(result, indent=2)+'\n')
