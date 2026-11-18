"""Join host GPU telemetry to individual timed sample windows; report coverage."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('results', type=Path)
p.add_argument('telemetry', type=Path)
p.add_argument('--gpu-index', default='1')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows = [r for r in csv.DictReader(a.telemetry.open()) if r['index'].strip() == a.gpu_index]
report = {}
for source in sorted(a.results.glob('*.json')):
    data = json.loads(source.read_text())
    cases = data.get('cases', [data])
    for case in cases:
        for mode, measurement in case.get('modes', {}).items():
            samples = measurement.get('samples')
            if samples is None:
                samples = [dict(s, variant=name) for name, metric in measurement['metrics'].items() for s in metric['samples']]
            for variant in measurement['metrics']:
                selected = [s for s in samples if s['variant'] == variant]
                matched = []
                covered = 0
                for sample in selected:
                    start = sample['start_unix_s']
                    end = start + (sample['wall_ms']/1000 if 'wall_ms' in sample else sample['wall_s'])
                    matches = [r for r in rows if start <= float(r['unix_s']) <= end]
                    covered += bool(matches)
                    matched.extend(matches)
                report[f'{source.name}/{mode}/{variant}'] = dict(
                    samples=len(selected), covered_samples=covered, polls=len(matched),
                    clocks=dict(Counter(r['clocks.sm'].strip() for r in matched)),
                    power_cap_flags=sum(r['clocks_event_reasons.sw_power_cap'].strip() == 'Active' for r in matched),
                    thermal_flags=sum(r['clocks_event_reasons.hw_thermal_slowdown'].strip() == 'Active' for r in matched),
                    **{k:v for k,v in measurement['metrics'][variant].items() if k != 'samples'})
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(a.output)
