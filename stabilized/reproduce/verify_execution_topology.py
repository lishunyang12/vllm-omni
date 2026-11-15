"""Recompute effective GPU counts from the published execution telemetry."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stabilized', type=Path)
args = parser.parse_args()
saved = json.loads((args.stabilized / 'execution-topology.json').read_text())
checked = 0
for round_name in ('initial', 'confirmation'):
    for family in ('rubin', 'gb200'):
        root = args.stabilized / round_name / family
        with (root / 'evidence/gpu-telemetry.csv').open() as stream:
            telemetry = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
        for row in telemetry:
            row['time_unix'] = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
        runs = json.loads((root / 'runs.json').read_text())
        for run in runs:
            assert run['returncode'] == 0 and run['status'] == 'completed'
            meta = json.loads((root / 'results' / run['model'] / run['variant'] / 'metadata.json').read_text())
            start = meta['start_unix'] + meta['initialization_seconds'] + 5
            end = run['start_unix'] + run['wall_seconds'] - 2
            stats = []
            for gpu in range(4):
                samples = [r for r in telemetry if int(r['index']) == gpu and start <= r['time_unix'] <= end]
                assert samples
                stats.append(dict(gpu=gpu, telemetry_samples=len(samples),
                    median_memory_mib=statistics.median(float(r['memory.used [MiB]'].split()[0]) for r in samples),
                    max_utilization_percent=max(float(r['utilization.gpu [%]'].split()[0]) for r in samples)))
            observed = [r['gpu'] for r in stats if r['median_memory_mib'] > 1024 and r['max_utilization_percent'] > 0]
            original = next(r for r in saved['results'] if r['round'] == round_name and r['gpu_family'].lower() == family
                            and r['model'] == run['model'] and r['variant'] == run['variant'])
            assert observed == original['observed_active_gpu_indices']
            assert stats == original['telemetry']
            assert observed == ([0] if run['model'] == 'wan22' else [0, 1, 2, 3])
            checked += 1
assert checked == len(saved['results']) == 40
print('Verified telemetry for all 40 cases: Wan single GPU, other models four GPUs.')
