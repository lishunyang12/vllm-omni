"""Audit every measured forward window against the correct physical GPU."""
import argparse
import csv
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
p.add_argument('telemetry',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rows=list(csv.DictReader(a.telemetry.open()))
protocol=json.loads((a.root/'protocol.json').read_text())
indices=protocol['physical_gpu_indices'].split(',')
result=[]
for rank,index in enumerate(indices):
    d=json.loads((a.root/f'rank-{rank}.json').read_text())
    for name in ('dense','baseline','candidate'):
        windows=[s for s in d['samples'] if s['variant']==name]
        selected=[];coverage=[]
        for s in windows:
            start,end=s['start_unix_s'],s['start_unix_s']+s['wall_s']
            match=[r for r in rows if r['index']==index and start<=float(r['unix_s'])<=end]
            assert match
            times=[start,*sorted(float(r['unix_s']) for r in match),end]
            coverage.append(dict(index=s['index'],polls=len(match),max_gap_s=max(y-x for x,y in zip(times,times[1:]))))
            selected.extend(match)
        clocks=[float(r['clocks.sm'].split()[0]) for r in selected]
        result.append(dict(rank=rank,index=index,variant=name,polls=len(selected),coverage=coverage,
                           clocks_mhz=sorted(set(clocks)),clock_median_mhz=statistics.median(clocks),
                           power_cap_active_polls=sum(r['clocks_event_reasons.sw_power_cap']=='Active' for r in selected),
                           thermal_active_polls=sum(r['clocks_event_reasons.hw_thermal_slowdown']=='Active' for r in selected),
                           power_draw_available=all(r['power.draw']!='[N/A]' for r in selected),
                           max_temperature_c=max(int(r['temperature.gpu']) for r in selected)))
a.output.write_text(json.dumps(dict(scope='Full measured windows; physical GPU mapping from protocol; no boundary trimming',results=result),indent=2)+'\n')
for r in result:print(r['rank'],r['variant'],r['clock_median_mhz'],min(r['clocks_mhz']),max(r['clocks_mhz']),r['power_cap_active_polls'],r['thermal_active_polls'])
