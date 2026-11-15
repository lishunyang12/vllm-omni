"""Run each source/precision pair in a fresh process and retain all logs/failures."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

work = Path(sys.argv[1]).resolve()
lock = (work / 'matrix.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX)
spec = json.loads((work / 'matrix.json').read_text())
encoder_ref = work / 'hf-cache/hub/models--MiniMaxAI--MiniMax-H3/refs/main'
encoder_ref.parent.mkdir(parents=True, exist_ok=True)
encoder_ref.write_text(spec['models']['h3_encoder']['revision'])
selected = os.environ.get('VALIDATION_MODELS', '').split(',') if os.environ.get('VALIDATION_MODELS') else list(spec['cases'])
if not set(selected).issubset(spec['cases']):
    raise ValueError('Unknown VALIDATION_MODELS entry')
(work / 'logs').mkdir(exist_ok=True)
records_path = work / 'evidence/runs.json'
records = json.loads(records_path.read_text()) if records_path.exists() else []
for key in selected:
    if key == 'h3_video':
        while not (work / 'evidence/h3-video-download.exit').exists():
            time.sleep(5)
        if (work / 'evidence/h3-video-download.exit').read_text().strip() != '0':
            raise RuntimeError('H3 video weights download failed; see evidence/h3-video-download.log')
        models = json.loads((work / 'evidence/models.json').read_text())
        models.update(json.loads((work / 'evidence/h3-video-model.json').read_text()))
        (work / 'evidence/models.json').write_text(json.dumps(models, indent=2) + '\n')
    for label, variant in spec['variants'].items():
        case_dir = work / 'outputs' / key / label
        if case_dir.exists():
            metadata_path = case_dir / 'metadata.json'
            if metadata_path.exists() and json.loads(metadata_path.read_text()).get('status') == 'completed':
                print('SKIP_COMPLETED', key, label, flush=True)
                continue
            attempts = work / 'failed-attempts' / key
            attempts.mkdir(parents=True, exist_ok=True)
            archived = attempts / f'{label}-{time.time_ns()}'
            case_dir.rename(archived)
            old_log = work / 'logs' / f'{key}-{label}.log'
            if old_log.exists():
                old_log.rename(archived / 'generation.log')
        record = dict(model=key, variant=label, start_unix=time.time())
        env = dict(os.environ, PYTHONPATH=str(work / 'source' / variant['source']),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0,1,2,3')
        command = [sys.executable, str(work / 'generate_case.py'), str(work), key, label]
        record['command'] = command
        print('START', key, label, flush=True)
        with (work / 'logs' / f'{key}-{label}.log').open('w') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                record['returncode'] = child.wait(timeout=900)
            except subprocess.TimeoutExpired:
                record.update(status='timeout', timeout_seconds=900)
            finally:
                # Clean up child workers even when the parent exits on a model-load error.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=15)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        record.setdefault('status', 'completed' if record.get('returncode') == 0 else 'failed')
        record['wall_seconds'] = time.time() - record['start_unix']
        records.append(record)
        (work / 'evidence/runs.json').write_text(json.dumps(records, indent=2) + '\n')
        print('END', key, label, record['status'], flush=True)
        time.sleep(3)
