"""Replay actual captured H3 attention inputs; no synthetic model tensors."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

p = argparse.ArgumentParser()
p.add_argument('--kind', choices=['residual', 'bsa'], required=True)
p.add_argument('--input', type=Path, required=True)
p.add_argument('--source', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--disable-clc', action='store_true')
p.add_argument('--full-block-sizes', action='store_true', help='Diagnostic only: replace ragged KV block sizes with 64.')
p.add_argument('--repeats', type=int, default=9)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)


def cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    if isinstance(x, dict):
        return {k: cuda(v) for k, v in x.items()}
    if isinstance(x, list):
        return [cuda(v) for v in x]
    return x


case = cuda(torch.load(a.input, map_location='cpu', weights_only=True))
args, kwargs = case['args'], case['kwargs']
report = dict(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
              torch=torch.__version__, input_file=a.input.name,
              shapes=[list(x.shape) if isinstance(x, torch.Tensor) else x for x in args],
              initial_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
              initial_float32_precision=torch.get_float32_matmul_precision(), runs={})

if a.kind == 'residual':
    source = a.source.read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == '_merge_sol_bsa_residual')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(a.source), 'exec'), namespace)
    fn = lambda: namespace['_merge_sol_bsa_residual'](*args, **kwargs)
    report['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
else:
    from cudnn.block_sparse_attention import block_sparse_attention_forward
    if a.disable_clc:
        kwargs['use_clc'] = False
    if a.full_block_sizes:
        kwargs['block_sizes'] = torch.full_like(kwargs['block_sizes'], 64)
    fn = lambda: block_sparse_attention_forward(*args, **kwargs)['o_tensor']
    report['disable_clc'] = a.disable_clc
    report['full_block_sizes_diagnostic'] = a.full_block_sizes
    indices = args[3]
    counts = kwargs['q2k_block_nums']
    report['metadata_checks'] = dict(
        counts_min=counts.min().item(), counts_max=counts.max().item(),
        index_min=indices.min().item(), index_max=indices.max().item(),
        capacity=indices.shape[-1],
        counts_match_valid_indices=bool(torch.equal((indices >= 0).sum(-1).to(counts.dtype), counts)),
        last_block_size=kwargs['block_sizes'][-1].item(),
        qkv_finite=all(bool(torch.isfinite(x).all().item()) for x in args[:3]))

(a.output/'protocol.json').write_text(json.dumps(report, indent=2)+'\n')


def bench(name, fn):
    samples = []
    for i in range(3 + a.repeats):
        print(f'MICRO_ENTER mode={name} iteration={i}', flush=True)
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        output = fn()
        stop.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - begin) * 1000
        gpu = start.elapsed_time(stop)
        print(f'MICRO_EXIT mode={name} iteration={i} wall_ms={wall:.6f} cuda_ms={gpu:.6f}', flush=True)
        if i >= 3:
            samples.append(dict(wall_ms=wall, cuda_ms=gpu))
        (a.output/'progress.json').write_text(json.dumps(dict(mode=name, completed_iteration=i,
                                                             last_wall_ms=wall, samples=samples), indent=2)+'\n')
    report['runs'][name] = dict(samples=samples,
                              median_wall_ms=statistics.median(x['wall_ms'] for x in samples),
                              median_cuda_ms=statistics.median(x['cuda_ms'] for x in samples),
                              finite=bool(torch.isfinite(output).all().item()))
    (a.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    return output


with torch.inference_mode():
    baseline = bench('eager', fn)
    if a.kind == 'residual':
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = fn()

        def replay():
            graph.replay()
            return graph_out

        graph_result = bench('cuda_graph_same_math', replay)
        report['runs']['cuda_graph_same_math']['bitwise_equal_to_eager'] = bool(torch.equal(baseline, graph_result))
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        tf32 = bench('tf32_diagnostic_only', fn)
        diff = tf32.float() - baseline.float()
        report['runs']['tf32_diagnostic_only'].update(
            max_abs_diff=diff.abs().max().item(), mean_abs_diff=diff.abs().mean().item(),
            relative_l2=(diff.norm()/baseline.float().norm()).item(),
            bitwise_equal_to_eager=bool(torch.equal(baseline, tf32)),
            note='Changes arithmetic precision; no generated-video quality validation or production change.')
        torch.backends.cuda.matmul.allow_tf32 = old
    (a.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print('MICRO_COMPLETE', json.dumps(report['runs']), flush=True)
