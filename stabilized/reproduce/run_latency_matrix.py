"""Run each precision in a fresh process; keep all samples and failures."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

work = Path(sys.argv[1]).resolve()
root = work / 'stabilized'
lock = (root / 'matrix.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX)
spec = json.loads((root / 'matrix.json').read_text())
protocol = json.loads((root / 'protocol.json').read_text())
(root / 'logs').mkdir(exist_ok=True)
records_path = root / 'runs.json'
records = json.loads(records_path.read_text()) if records_path.exists() else []
for model in protocol['model_order']:
    for variant in protocol['variant_order']:
        case = root / 'results' / model / variant
        meta = case / 'metadata.json'
        if meta.exists() and json.loads(meta.read_text()).get('status') == 'completed':
            print('SKIP_COMPLETED',model,variant,flush=True)
            continue
        if case.exists():
            old = root / 'failed-attempts' / model
            old.mkdir(parents=True,exist_ok=True)
            archived = old / f'{variant}-{time.time_ns()}'
            case.rename(archived)
            prior_log=root/'logs'/f'{model}-{variant}.log'
            if prior_log.exists():
                prior_log.rename(archived/'generation.log')
        source=spec['variants'][variant]['source']
        env=dict(os.environ,PYTHONPATH=str(work/'source'/source),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0,1,2,3')
        record=dict(model=model,variant=variant,start_unix=time.time())
        command=[sys.executable,str(root/'benchmark_case.py'),str(work),model,variant]
        print('START',model,variant,flush=True)
        (root/'status').write_text(f'running {model} {variant}\n')
        with (root/'logs'/f'{model}-{variant}.log').open('w') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                record['returncode']=child.wait(timeout=protocol['case_timeout_seconds'])
                record['status']='completed' if record['returncode']==0 else 'failed'
            except subprocess.TimeoutExpired:
                record.update(status='timeout',timeout_seconds=protocol['case_timeout_seconds'])
            finally:
                try:
                    os.killpg(child.pid,signal.SIGTERM)
                    child.wait(timeout=15)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
        record['wall_seconds']=time.time()-record['start_unix']
        records.append(record)
        records_path.write_text(json.dumps(records,indent=2)+'\n')
        print('END',model,variant,record['status'],flush=True)
        time.sleep(3)
(root/'status').write_text('matrix_complete\n')
