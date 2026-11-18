"""Summarize corrected FP32-pooling measurements and telemetry coverage."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics


def audit(samples, rows):
    covered = 0
    polls = []
    for sample in samples:
        start = sample['start_unix_s']
        duration = sample['wall_s'] if 'wall_s' in sample else sample['wall_ms'] / 1000
        matches = [row for row in rows if start <= float(row['unix_s']) <= start + duration]
        covered += bool(matches)
        polls.extend(matches)
    return dict(samples=len(samples), covered_samples=covered, polls=len(polls),
                clocks=dict(Counter(row['clocks.sm'].strip() for row in polls)),
                power_cap_flags=sum(row['clocks_event_reasons.sw_power_cap'].strip() == 'Active' for row in polls),
                thermal_flags=sum(row['clocks_event_reasons.hw_thermal_slowdown'].strip() == 'Active' for row in polls),
                power_draw_available=bool(polls) and all(row['power.draw'].strip() != '[N/A]' for row in polls))


def ratios(old, new):
    values = [a / b for a, b in zip(old, new, strict=True)]
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--attention', default='process-fp32-750')
    args = parser.parse_args()
    root = args.root
    report = dict(hardware='SM100 engineering GPU, cluster label B100; not production B200',
                  common_pooling='Explicit FP32 accumulation; BF16 output; forced4warps/1stage in both SOL variants',
                  reference='63b4e65 plus common pool-fp32.patch',
                  candidate='150114b plus common pool-fp32.patch',
                  quality_scope='Same repaired-pooling SOL outputs; no generated-video quality metrics',
                  full_forward=None, attention={})
    directory = root / 'results/ref2va-native-fp32-450'
    if (directory / 'summary.json').exists():
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['candidate_bitwise_equal'] and summary['measured_pairs'] == 9
        telemetry = list(csv.DictReader((root / 'evidence/fp32-450-control/telemetry.csv').open()))
        checks = {}
        for rank in range(2):
            data = json.loads((directory / f'rank-{rank}.json').read_text())
            assert data['finite'] and data['candidate_bitwise_equal']
            rows = [row for row in telemetry if row['index'].strip() == str(rank + 1)]
            for name in ('dense', 'baseline', 'candidate'):
                checks[f'rank{rank}/{name}'] = audit([s for s in data['samples'] if s['variant'] == name], rows)
        metrics = summary['runs']
        report['full_forward'] = dict(summary=summary, telemetry=checks,
            sol_reduction_percent=100 * (1 - metrics['candidate']['median_ms'] / metrics['baseline']['median_ms']),
            candidate_slower_than_dense_percent=100 * (metrics['candidate']['median_ms'] / metrics['dense']['median_ms'] - 1),
            paired_sol_speedup=ratios(metrics['baseline']['samples_ms'], metrics['candidate']['samples_ms']))
    directory = root / 'results' / args.attention
    if directory.exists():
        telemetry = list(csv.DictReader((root / 'results' / (args.attention + '-control') / 'telemetry.csv').open()))
        rows = [row for row in telemetry if row['index'].strip() == '1']
        for path in sorted(directory.glob('layer-*.json')):
            data = json.loads(path.read_text())
            if not data.get('complete') or not data.get('valid'):
                report['attention'][path.stem] = dict(valid=False, error=data.get('error'))
                continue
            modes = {}
            for mode, measurements in data['modes'].items():
                assert measurements['validated']
                checks = {name: audit([s for s in measurements['samples'] if s['variant'] == name], rows)
                          for name in measurements['metrics']}
                m = measurements['metrics']
                modes[mode] = dict(metrics=m, telemetry=checks,
                    sol_reduction_percent=100 * (1 - m['candidate']['median_ms'] / m['baseline']['median_ms']))
            report['attention'][path.stem] = dict(valid=True, modes=modes)
    path = root / 'corrected-fp32-summary.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(path)
    if report['full_forward']:
        print('Forward ms:', {name: m['median_ms'] for name, m in report['full_forward']['summary']['runs'].items()})
        print('SOL reduction %:', report['full_forward']['sol_reduction_percent'])
    for name, entry in report['attention'].items():
        print(name, entry.get('valid'), {mode: x['metrics'] for mode, x in entry.get('modes', {}).items()})


if __name__ == '__main__':
    main()
