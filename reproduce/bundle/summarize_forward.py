import argparse,csv,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
for hardware in ['sm90','sm100']:
 for task in ['t2va','fl2va','ref2va']:
  paths=[a.root/hardware/'results'/f'{task}-{b}' for b in ['CUDNN_ATTN','SOL_ATTN']]
  if not all((x/'summary.json').exists() for x in paths):continue
  s=[json.loads((x/'summary.json').read_text()) for x in paths]
  for rank in range(2):
   r=[json.loads((x/f'rank-{rank}.json').read_text()) for x in paths]
   if r[0]['input']!=r[1]['input']:raise ValueError(f'{hardware}/{task}/rank{rank}: A/B inputs differ; cannot report matched-input comparison')
   if r[0]['parallel_config']!=r[1]['parallel_config']:raise ValueError('A/B parallel config differs')
  d,o=s
  rows.append(dict(hardware=hardware,gpu=d['gpu'],task=task,dense_s=d['median_s'],sol_s=o['median_s'],speedup=d['median_s']/o['median_s'],dense_first3_s=d['median_first3_s'],sol_first3_s=o['median_first3_s'],warmup=d['warmup'],repeats=d['repeats'],dense_min_s=d['min_s'],dense_max_s=d['max_s'],sol_min_s=o['min_s'],sol_max_s=o['max_s'],dense_cv_percent=d['cv_percent'],sol_cv_percent=o['cv_percent'],inputs_match=True))
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
if rows:
 with (a.output/'comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
