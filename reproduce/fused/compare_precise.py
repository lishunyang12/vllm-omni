import ast
import json
from pathlib import Path
import statistics
import time

import torch
from vllm_omni.diffusion.attention.ops.sol_residual import merge_sol_residual

out = Path('/work/fused/micro-precise')
out.mkdir(parents=True, exist_ok=True)
source = Path('/work/source/head/vllm_omni/diffusion/attention/ops/sol_attn.py')
node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == '_merge_sol_bsa_residual')
ns = {'torch': torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
data = torch.load('/work/profile/results/ref2va-SOL_ATTN-capture/residual-input-rank0.pt', weights_only=True)
args = [t.cuda() if isinstance(t, torch.Tensor) else t for t in data['args']]
report = dict(gpu=torch.cuda.get_device_name(), allow_tf32=torch.backends.cuda.matmul.allow_tf32,
              shapes=[dict(shape=list(t.shape), dtype=str(t.dtype), stride=list(t.stride()))
                      if isinstance(t, torch.Tensor) else t for t in args], runs={})
print('INPUT', json.dumps(report), flush=True)


def measure(name, fn):
    samples = []
    for i in range(12):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        print(name, i, elapsed, flush=True)
        if i >= 3:
            samples.append(elapsed)
    row = dict(samples_ms=samples, median_ms=statistics.median(samples),
               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               finite=bool(torch.isfinite(output).all().item()))
    return row, output


with torch.inference_mode():
    report['runs']['reference'], expected = measure('reference', lambda: ns['_merge_sol_bsa_residual'](*args))
    for chunk in [2048]:
        name = f'fused_{chunk}'
        row, actual = measure(name, lambda: merge_sol_residual(*args, chunk_size=chunk))
        diff = actual.float() - expected.float()
        row.update(max_abs=diff.abs().max().item(), mean_abs=diff.abs().mean().item(),
                   relative_l2=(diff.norm()/expected.float().norm()).item(),
                   different_fraction=(actual != expected).float().mean().item())
        report['runs'][name] = row
        (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print('RESULT', name, json.dumps(row), flush=True)
