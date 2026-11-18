import collections
import csv
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parent
raw = root/'raw' if (root/'raw').exists() else root
out = root/'analysis'
out.mkdir(exist_ok=True)
cases = {'SOL': 'ref2va-SOL_ATTN-both', 'Dense': 'ref2va-CUDNN_ATTN-both-v2'}
rows, kernel_rows, summaries = [], [], {}
inputs = {}
for backend, name in cases.items():
    p = raw/'sm90/results'/name
    baseline = json.loads((p/'summary.json').read_text())
    summary = dict(baseline=baseline, ranks=[])
    for rank in [0, 1]:
        profile = json.loads((p/f'profile-events-rank{rank}.json').read_text())
        report = json.loads((p/f'rank-{rank}.json').read_text())
        assert report['finite'] and report['world_size'] == 2 and report['repeats'] == 3
        inputs[backend, rank] = report['input']
        totals = collections.defaultdict(float)
        counts = collections.Counter()
        for event in profile['events']:
            totals[event['name']] += event['cuda_ms']
            counts[event['name']] += 1
        for name_, milliseconds in totals.items():
            rows.append(dict(backend=backend, rank=rank, stage=name_, calls=counts[name_], cuda_ms=milliseconds))
        routing = profile['routing']
        densities = [r['selected']/r['capacity'] for r in routing]
        summary['ranks'].append(dict(rank=rank, profiled_wall_s=profile['wall_s'], stage_cuda_ms=dict(totals),
                                    density_mean=sum(densities)/len(densities) if densities else None,
                                    density_min=min(densities) if densities else None,
                                    density_max=max(densities) if densities else None))
    trace_file = p/'attention-layer0-rank0.json'
    trace_text = trace_file.read_text() if trace_file.exists() else gzip.decompress(trace_file.with_suffix('.json.gz').read_bytes()).decode()
    trace = json.loads(trace_text)['traceEvents']
    kernels = [e for e in trace if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    grouped = collections.defaultdict(lambda: dict(calls=0, cuda_us=0))
    for e in kernels:
        grouped[e['name']]['calls'] += 1
        grouped[e['name']]['cuda_us'] += e['dur']
    for name_, stats in sorted(grouped.items(), key=lambda item: -item[1]['cuda_us']):
        kernel_rows.append(dict(backend=backend, name=name_, calls=stats['calls'], cuda_ms=stats['cuda_us']/1000))
    summary['layer0_kernel_count'] = len(kernels)
    summary['layer0_bmm_calls'] = sum(e.get('cat') == 'cpu_op' and e.get('name') == 'aten::bmm' for e in trace)
    summary['layer0_kernel_sum_ms'] = sum(e['dur'] for e in kernels)/1000
    summary['trace_note'] = 'Profiler trace timing has instrumentation overhead; phase CUDA events come from a separate replay.'
    summaries[backend] = summary

for rank in [0, 1]:
    assert inputs['Dense', rank] == inputs['SOL', rank], f'Input mismatch on rank {rank}'

micro = json.loads((raw/'sm90/micro/residual/results.json').read_text())
assert micro['runs']['cuda_graph_same_math']['bitwise_equal_to_eager']
summaries['residual_microbenchmark'] = micro
summaries['same_forward_input_hashes_both_ranks'] = True
summaries['nested_event_note'] = 'Stage intervals include nested child stages. Do not sum parent and child rows.'
(out/'profile-summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
for filename, data in [('stage-times.csv', rows), ('layer0-kernels.csv', kernel_rows)]:
    with (out/filename).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
print('Both-rank Dense/SOL input hashes match; phase and kernel summaries written.')
