"""Diagnostic kernel counts, separate from unprofiled latency samples."""
import ast
from collections import Counter
import json
from pathlib import Path

import torch
from sol_residual import merge_sol_residual

root = Path('/work/fused/micro-profile')
root.mkdir(parents=True, exist_ok=True)
source = Path('/work/source/head/vllm_omni/diffusion/attention/ops/sol_attn.py')
node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == '_merge_sol_bsa_residual')
ns = {'torch': torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
data = torch.load('/work/profile/results/ref2va-SOL_ATTN-capture/residual-input-rank0.pt', weights_only=True)
args = [t.cuda() if isinstance(t, torch.Tensor) else t for t in data['args']]
results = {}
with torch.inference_mode():
    for name, fn in [('reference', ns['_merge_sol_bsa_residual']), ('fused', merge_sol_residual)]:
        for _ in range(3):
            fn(*args)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA]) as p:
            fn(*args)
            torch.cuda.synchronize()
        path = root / f'{name}.json'
        p.export_chrome_trace(str(path))
        events = json.loads(path.read_text())['traceEvents']
        kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
        results[name] = dict(kernel_count=len(kernels),
                             kernel_counts=Counter(e['name'] for e in kernels))
(root / 'counts.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
