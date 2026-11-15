"""Audit published diagnostic records against the original 20-request matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import re


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnosis', type=Path)
    parser.add_argument('stabilized', type=Path)
    args = parser.parse_args()
    root, original = args.diagnosis, args.stabilized
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    runs = read(root / 'runs.json')
    assert len(runs) == 6 and all(r['status'] == 'completed' and r['returncode'] == 0 for r in runs)
    assert read(root / 'evidence/versions.json') == read(original / 'initial/rubin/evidence/versions.json')
    summaries = read(root / 'analysis/summary.json')
    assert len(summaries) == 15
    records = []
    for run in runs:
        model, variant = run['model'], run['variant']
        case = root / 'results' / model / variant
        m = read(case / 'metadata.json')
        old = read(original / 'initial/rubin/results' / model / variant / 'metadata.json')
        assert m['status'] == 'completed' and m['profile']['status'] == 'completed'
        for key in ('source_sha256', 'case', 'variant', 'quantization_config', 'enforce_eager', 'torch', 'cuda'):
            assert m[key] == old[key], (model, variant, key)
        for key in ('model', 'revision'):
            assert m['model'][key] == old['model'][key]
        assert m['benchmark_script_sha256'] == digest(root / 'reproduce/benchmark_case.py')
        measured = [r for r in m['requests'] if r['phase'] == 'measured']
        warmups = [r for r in m['requests'] if r['phase'] == 'warmup']
        assert len(measured) == 3 and len(warmups) == 5
        values = sorted(r['seconds'] for r in measured)
        assert abs(statistics.median(values) - m['summary']['p50_seconds']) < 1e-9
        assert abs((values[1] * .1 + values[2] * .9) - m['summary']['p95_seconds']) < 1e-9
        prior = next(r for r in old['requests'] if r['phase'] == 'measured')
        original_values = [r['seconds'] for r in old['requests'] if r['phase'] == 'measured']
        assert len(original_values) == 20
        assert abs(statistics.median(original_values) - old['summary']['p50_seconds']) < 1e-9
        original_log = (root / 'original-logs' / f'{model}-{variant}.log').read_text()
        marker = f'LATENCY_SAMPLE {model} {variant} warmup 4 '
        assert original_log.count(marker) == 1
        measured_log = original_log.split(marker, 1)[1].split('\n', 1)[1]
        assert not re.search(r'autotun|compil(?:ing|ation|e)', measured_log, re.I)
        if model == 'qwen_image':
            assert 'CUDNN_ATTN' in original_log
        if model == 'qwen_image' and variant == 'pr_mxfp8':
            assert 'FlashInferCutedslMxfp8LinearKernel' in original_log
        if model == 'qwen_image' and variant == 'pr_fp8_block':
            assert 'DeepGemmFp8BlockScaledMMKernel' in original_log
        same_pixels = measured[0]['pixel_sha256'] == measured[-1]['pixel_sha256'] == prior['pixel_sha256']
        assert same_pixels, (model, variant, 'pixel hashes differ')
        ranks = [s for s in summaries if s['model'] == model and s['variant'] == variant]
        expected = [0] if model == 'wan22' else [0, 1, 2, 3]
        assert sorted(s['rank'] for s in ranks) == expected
        traces = sorted(case.rglob('trace_rank*.json.gz'))
        assert len(traces) == len(expected)
        for trace in traces:
            rank = int(trace.name.split('rank')[1].split('.')[0])
            s = next(s for s in ranks if s['rank'] == rank)
            assert s['trace_sha256'] == digest(trace)
            assert s['matched_dit_linear_calls'] == (720 if model == 'wan22' else 1920)
        records.append(dict(model=model, variant=variant, observed_gpus=len(expected),
                            measured_requests=3, warmup_requests=5,
                            p50_seconds=m['summary']['p50_seconds'], p95_seconds=m['summary']['p95_seconds'],
                            original_n20_p50_seconds=old['summary']['p50_seconds'],
                            original_n20_p95_seconds=old['summary']['p95_seconds'],
                            matches_original_pixels=same_pixels,
                            metadata_sha256=digest(case / 'metadata.json')))
    micro = read(root / 'microbench/results.json')
    assert read(root / 'microbench-process.json')['returncode'] == 0
    assert len(micro['shapes']) == 4 and micro['cache_loaded']
    for shape in micro['shapes']:
        for mode in ('bf16_eager', 'mxfp8_eager', 'bf16_graph_replay', 'mxfp8_graph_replay'):
            assert len(shape[mode]['rounds_us_per_call']) == 7 and shape[mode]['calls_per_round'] == 100
    report = dict(status='passed', configurations=6, measured_requests=18, full_warmups=30,
                  worker_traces=15, runtime_versions_match_original=True,
                  all_first_last_pixels_match_original=True, records=records,
                  notes=['Wan used one GPU, despite requested four-GPU kwargs in the original scripts.',
                         'profile.files is empty due to an executed filename-glob bug; actual traces are enumerated and SHA-256 checked here.',
                         'Three-request diagnostic replication is not a replacement for the original 20-request benchmark.'])
    (root / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Verified 6 configurations, 18 full measured requests, 15 traces and all output hashes.')


if __name__ == '__main__':
    main()
