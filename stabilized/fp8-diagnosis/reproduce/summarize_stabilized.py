"""Validate and summarize warm latency records without GPU dependencies."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def percentile(values, q):
    ordered=sorted(values)
    index=(len(ordered)-1)*q
    low=math.floor(index)
    high=math.ceil(index)
    return ordered[low]+(ordered[high]-ordered[low])*(index-low)


def summarize(root, family):
    protocol=json.loads((root/'protocol.json').read_text())
    result=[]
    for model in protocol['model_order']:
        for variant in protocol['variant_order']:
            path=root/'results'/model/variant/'metadata.json'
            if not path.exists():
                result.append(dict(gpu=family,model=model,variant=variant,status='unavailable'))
                continue
            metadata=json.loads(path.read_text())
            if metadata['status']!='completed':
                result.append(dict(gpu=family,model=model,variant=variant,status=metadata['status'],error=metadata.get('error')))
                continue
            assert metadata['protocol']==protocol,path
            records=metadata['requests']
            warmups=protocol['warmup_requests']
            count=protocol['measured_requests']
            assert len(records)==warmups+count,path
            assert [r['phase'] for r in records]==['warmup']*warmups+['measured']*count,path
            assert [r['index'] for r in records]==list(range(warmups+count)),path
            values=[r['seconds'] for r in records[warmups:]]
            assert all(math.isfinite(x) and x>0 for x in values),path
            p50=percentile(values,.50)
            p95=percentile(values,.95)
            cv=statistics.stdev(values)/statistics.mean(values)
            half=statistics.median(values[:len(values)//2])
            drift=abs(statistics.median(values[len(values)//2:])-half)/half
            checks={'coefficient_of_variation':cv,'half_median_relative_change':drift,'p95_over_p50':p95/p50}
            stable=all(v<=protocol['stability_limits'][k+'_max'] for k,v in checks.items())
            expected={'p50_seconds':p50,'p95_seconds':p95,**checks}
            for key,value in expected.items():
                assert math.isclose(metadata['summary'][key],value,rel_tol=1e-10,abs_tol=1e-10),(path,key)
            assert metadata['stable']==stable,path
            result.append(dict(gpu=family,model=model,variant=variant,status='completed',stable=stable,n=count,
                p50_seconds=p50,p95_seconds=p95,mean_seconds=statistics.mean(values),min_seconds=min(values),max_seconds=max(values),
                cv_percent=cv*100,half_median_change_percent=drift*100,p95_over_p50=p95/p50,
                initialization_seconds=metadata['initialization_seconds'],
                first_last_measured_pixels_identical=metadata['first_last_measured_pixels_identical'],
                source_commit=metadata['variant']['commit'],
                metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True,help='GPU_LABEL=/path/to/stabilized')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for entry in a.run:
        family,directory=entry.split('=',1)
        rows.extend(summarize(Path(directory),family))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'stabilized-latency.json').write_text(json.dumps({'results':rows},indent=2)+'\n')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (a.output/'stabilized-latency.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print('| GPU | Model | Variant | n | P50 s | P95 s | CV % | Half drift % | Stable |')
    print('|---|---|---|---:|---:|---:|---:|---:|---|')
    for row in rows:
        if row['status']!='completed':
            print(f"| {row['gpu']} | {row['model']} | {row['variant']} | — | — | — | — | — | {row['status']} |")
        else:
            print(f"| {row['gpu']} | {row['model']} | {row['variant']} | {row['n']} | {row['p50_seconds']:.3f} | {row['p95_seconds']:.3f} | {row['cv_percent']:.2f} | {row['half_median_change_percent']:.2f} | {'yes' if row['stable'] else 'no'} |")


if __name__=='__main__':
    main()
