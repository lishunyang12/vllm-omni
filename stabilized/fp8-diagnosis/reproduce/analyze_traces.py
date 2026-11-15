"""Summarize worker traces without conflating kernel sums with wall latency."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re


def union_duration(intervals):
    total = 0.0
    end = float('-inf')
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def category(name):
    name = name.lower()
    if 'nccl' in name:
        return 'communication'
    if ('quant' in name or ('cvt' in name and 'fp8' in name)
            or '_to_copy_abs_ceil_div_exp2_log2_max_view' in name
            or '_to_copy_ceil_clone_exp2_log2_squeeze_transpose' in name
            or 'pack_fp32_into_ue8m0' in name):
        return 'quantization'
    if any(s in name for s in ('sdpa', 'fmha', 'attention', 'flash_fprop')):
        return 'attention'
    if any(s in name for s in ('gemm', 'nvjet', 'matmul', 'mxfp8')):
        return 'gemm_or_matmul'
    return 'other'


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def analyze(path, out):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cached = out / 'summary.json'
    if cached.exists():
        saved = json.loads(cached.read_text())
        if saved.get('analysis_version') == 3 and saved['trace_sha256'] == digest:
            return saved
    with gzip.open(path, 'rt') as stream:
        data = json.load(stream)
    events = data['traceEvents']
    cpus = [e for e in events if e.get('cat') == 'cpu_op' and e.get('ph') == 'X']
    by_id = {e.get('args', {}).get('External id'): e for e in cpus}
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    annotations = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X']
    pipeline = [e for e in annotations if e['name'] == 'pipeline_forward']
    assert len(pipeline) == 1, (path, len(pipeline))
    start, stop = pipeline[0]['ts'], pipeline[0]['ts'] + pipeline[0]['dur']
    kernel_groups = defaultdict(lambda: {'count': 0, 'gpu_us': 0.0})
    shape_groups = defaultdict(lambda: {'calls': 0, 'cpu_inclusive_us': 0.0, 'cpu_self_us': 0.0,
                                       'direct_kernel_count': 0, 'direct_kernel_gpu_us': 0.0})
    cpu_threads = defaultdict(list)
    for e in cpus:
        cpu_threads[e['pid'], e['tid']].append(e)
    self_times = {}
    for group in cpu_threads.values():
        stack = []
        for e in sorted(group, key=lambda x: (x['ts'], -x['dur'])):
            while stack and e['ts'] >= stack[-1]['ts'] + stack[-1]['dur']:
                stack.pop()
            self_times[id(e)] = e['dur']
            if stack and e['ts'] + e['dur'] <= stack[-1]['ts'] + stack[-1]['dur'] + .001:
                self_times[id(stack[-1])] -= e['dur']
            stack.append(e)
    def shape_key(e):
        return e['name'], json.dumps(e.get('args', {}).get('Input Dims', []), separators=(',', ':'))
    for e in cpus:
        g = shape_groups[shape_key(e)]
        g['calls'] += 1
        g['cpu_inclusive_us'] += e['dur']
        g['cpu_self_us'] += max(0, self_times[id(e)])
    for e in kernels:
        g = kernel_groups[e['name']]
        g['count'] += 1
        g['gpu_us'] += e['dur']
        parent = by_id.get(e.get('args', {}).get('External id'))
        key = shape_key(parent) if parent else ('unmatched', '[]')
        g = shape_groups[key]
        g['direct_kernel_count'] += 1
        g['direct_kernel_gpu_us'] += e['dur']
    kernel_rows = [dict(name=name, category=category(name), **values) for name, values in kernel_groups.items()]
    kernel_rows.sort(key=lambda r: -r['gpu_us'])
    shape_rows = [dict(name=key[0], input_shapes=key[1], **values) for key, values in shape_groups.items()]
    shape_rows.sort(key=lambda r: -r['direct_kernel_gpu_us'])
    write_csv(out / 'kernels.csv', kernel_rows)
    write_csv(out / 'operators-by-shape.csv', shape_rows)
    category_ms = defaultdict(float)
    for r in kernel_rows:
        category_ms[r['category']] += r['gpu_us'] / 1000
    clipped = lambda e: (max(start, e['ts']), min(stop, e['ts'] + e['dur']))
    active = [e for e in kernels if e['ts'] < stop and e['ts'] + e['dur'] > start]
    runtime = [e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver') and e.get('ph') == 'X']
    bias_rows = []
    for row in shape_rows:
        dims = json.loads(row['input_shapes'])
        if (row['name'] == 'aten::add' and len(dims) >= 2 and len(dims[0]) == 2
                and len(dims[1]) == 1 and dims[0][-1] == dims[1][0]):
            bias_rows.append(row)
    model, variant = path.parents[3].name, path.parents[2].name
    # Match DiT linears only. Wan's small text projections share a BF16 shape
    # with four unquantized conditioning operations, so compare its large
    # image branch (M=27280) only. Qwen covers all 1920 quantized-scope calls.
    streams = defaultdict(list)
    for e in kernels:
        streams[e['args']['device'], e['args']['stream']].append(e)
    next_on_stream = {}
    for group in streams.values():
        ordered = sorted(group, key=lambda e: e['ts'])
        next_on_stream.update({id(a): b for a, b in zip(ordered, ordered[1:])})
    matched = []
    for e in kernels:
        parent = by_id.get(e.get('args', {}).get('External id'), {})
        dims = parent.get('args', {}).get('Input Dims', [])
        selected = False
        if variant == 'pr_bf16' and parent.get('name') == 'aten::addmm':
            m, k = dims[1]
            n = dims[0][0]
            selected = (m in (5, 36, 1024) and k in (3072, 12288) and n in (3072, 9216, 12288)) if model == 'qwen_image' else (m == 27280 and k in (3072, 14336) and n in (3072, 9216, 14336))
        elif variant == 'pr_fp8_block' and 'deep_gemm::sm100_fp8_fp4_gemm' in e['name']:
            selected = model == 'qwen_image' or dims[0][0] == 27280
        elif variant == 'pr_mxfp8' and 'flashinfergemmkernelsdense' in e['name']:
            # These direct CuTe launches lack a cpu_op External id. Associate
            # with the immediately following bias-add on the same CUDA stream.
            following = next_on_stream.get(id(e), {})
            bias = by_id.get(following.get('args', {}).get('External id'), {})
            bd = bias.get('args', {}).get('Input Dims', [])
            assert bias.get('name') == 'aten::add' and len(bd[0]) == 2 and len(bd[1]) == 1, (model, variant, e['name'])
            selected = model == 'qwen_image' or bd[0][0] == 27280
        if selected:
            matched.append(e)
    linear_calls = len({e['args']['External id'] for e in matched}) if variant == 'pr_bf16' else len(matched)
    assert linear_calls == (1920 if model == 'qwen_image' else 720), (model, variant, linear_calls)
    report = {'analysis_version': 3, 'trace_sha256': digest,
              'trace_bytes': path.stat().st_size,
              'pipeline_wall_ms': pipeline[0]['dur'] / 1000,
              'gpu_kernel_count': len(kernels),
              'gpu_kernel_sum_ms': sum(e['dur'] for e in kernels) / 1000,
              'gpu_kernel_union_within_pipeline_ms': union_duration([clipped(e) for e in active]) / 1000,
              'noncommunication_kernel_union_within_pipeline_ms': union_duration([clipped(e) for e in active if category(e['name']) != 'communication']) / 1000,
              'cpu_operator_count': len(cpus),
              'cpu_operator_union_within_pipeline_ms': union_duration([clipped(e) for e in cpus if e['ts'] < stop and e['ts'] + e['dur'] > start]) / 1000,
              'cuda_api_count': len(runtime),
              'cuda_api_inclusive_sum_ms': sum(e['dur'] for e in runtime) / 1000,
              'separate_2d_bias_add_calls': sum(r['calls'] for r in bias_rows),
              'separate_2d_bias_add_gpu_ms': sum(r['direct_kernel_gpu_us'] for r in bias_rows) / 1000,
              'matched_dit_scope': 'all quantized-scope DiT linears' if model == 'qwen_image' else 'large image branch M=27280 only; excludes text KV and conditioning',
              'matched_dit_linear_calls': linear_calls,
              'matched_dit_gpu_kernel_count': len(matched),
              'matched_dit_gemm_gpu_ms': sum(e['dur'] for e in matched) / 1000,
              'kernel_name_categories_ms': dict(category_ms),
              'top_kernels': kernel_rows[:20],
              'notes': ['Two-step diagnostic request after warmup; not a replacement end-to-end latency benchmark.',
                        'Kernel duration sums may overlap across streams and must not be added to CPU times.',
                        'Name-based categories are descriptive; direct kernel/operator associations use External id.',
                        'NCCL duration includes peer waiting and does not alone identify a network bottleneck.']}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.results.glob('*/*/traces/*/trace_rank*.json.gz')):
        model, variant = path.relative_to(args.results).parts[:2]
        rank = int(re.search(r'trace_rank(\d+)', path.name).group(1))
        out = args.output / model / variant / f'rank{rank}'
        out.mkdir(parents=True, exist_ok=True)
        report = analyze(path, out)
        summaries.append(dict(model=model, variant=variant, rank=rank,
                              **{k: v for k, v in report.items() if k != 'top_kernels'}))
        print(model, variant, rank, 'pipeline_ms', round(report['pipeline_wall_ms'], 2),
              'categories_ms', {k: round(v, 2) for k, v in report['kernel_name_categories_ms'].items()}, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')


if __name__ == '__main__':
    main()
