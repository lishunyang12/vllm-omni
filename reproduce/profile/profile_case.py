import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import urllib.request

# multiprocessing spawn imports this main module in each worker.
if os.environ.get('H3_FORWARD_RESULT'):
    from profile_probe import install
    install()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, default=Path('/work'))
    p.add_argument('--task', choices=['t2va', 'fl2va', 'ref2va'], required=True)
    p.add_argument('--backend', choices=['CUDNN_ATTN', 'SOL_ATTN'], required=True)
    p.add_argument('--cpu-offload', action='store_true')
    a = p.parse_args()
    request_path = a.work / 'bundle' / f'{a.task}-request.json'
    request = json.loads(request_path.read_text())
    model_info = json.loads((a.work / 'evidence/models.json').read_text())
    model = model_info['paths']['Ref2VA' if a.task == 'ref2va' else 'FL2VA']
    out = Path(os.environ['H3_FORWARD_RESULT'])
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob('rank-*.json')):
        raise RuntimeError('Refusing to mix old results into a fresh benchmark')
    multi_modal = {}
    assets = []
    for i, item in enumerate(request.get('conditions', [])):
        ext = {'image': '.jpg', 'video': '.mp4', 'audio': '.mp3'}[item['type']]
        target = a.work / 'assets' / f'{a.task}-{i}{ext}'
        if not target.exists():
            urllib.request.urlretrieve(item['uri'], target)
        assets.append({'path': target.name, 'source': item['uri'], 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        multi_modal.setdefault(item['type'], []).append(str(target))
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    settings = dict(model=model, dtype='bfloat16', trust_remote_code=True, num_gpus=2,
                    tensor_parallel_size=1, ulysses_degree=2, ring_degree=1,
                    text_encoder_tp_size=2, vae_patch_parallel_size=2,
                    vae_use_tiling=True, enforce_eager=True,
                    diffusion_attention_backend=a.backend,
                    task_type='ref2va' if a.task == 'ref2va' else 'fl2va',
                    enable_cpu_offload=a.cpu_offload,
                    init_timeout=1800, stage_init_timeout=1800)
    protocol = dict(request=request, request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
                    assets=assets, model=model_info, engine=settings,
                    runtime_commit='6f5a99768f668bff08e40b3219e308743157d36c',
                    test_fix_commit='731ea29270e2d9f30e68f94d43a31021b4727eef',
                    measurement='First conditional denoising step, entire 50-layer DiT forward; identical input replay; max rank wall latency')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    engine = Omni(**settings)
    terminal_exception = None
    try:
        params = OmniDiffusionSamplingParams(height=None, width=None, num_frames=1,
                    num_inference_steps=50, seed=request['seed'], fps=24, guidance_scale=1.0,
                    extra_args={'task': a.task, 'target': request['target'], 'flow_shift': 12.0, 'audio_flow_shift': 3.0})
        (out / 'armed').write_text('Capture the first conditional DiT call of this real request.\n')
        engine.generate({'prompt': request['prompt'], 'multi_modal_data': multi_modal, 'modalities': ['video']}, params)
    except Exception as exc:
        terminal_exception = f'{type(exc).__name__}: {exc}'
        print('TERMINAL_EXCEPTION', terminal_exception, flush=True)
    finally:
        engine.close()
    if os.environ.get('H3_PROFILE_MODE') == 'capture':
        assert (out/'bsa-input-rank0.pt').exists() and (out/'residual-input-rank0.pt').exists()
        assert all((out/f'profile-capture-rank{r}.json').exists() for r in range(2))
        print('CAPTURE_COMPLETE: diagnostic inputs saved; no latency claim', flush=True)
        return
    reports = [json.loads((out / f'rank-{r}.json').read_text()) for r in range(2)]
    assert all(r['finite'] and r['intentional_abort'] for r in reports)
    n = reports[0]['repeats']
    assert n == reports[1]['repeats'] and n >= 3
    samples = [max(r['samples'][i]['wall_s'] for r in reports) for i in range(n)]
    summary = {'task': a.task, 'backend': a.backend, 'gpu': reports[0]['gpu'],
               'warmup': reports[0]['warmup'], 'repeats': n, 'max_rank_wall_samples_s': samples,
               'median_s': statistics.median(samples), 'median_first3_s': statistics.median(samples[:3]),
               'min_s': min(samples), 'max_s': max(samples),
               'cv_percent': 100 * statistics.stdev(samples) / statistics.mean(samples),
               'terminal_exception': terminal_exception, 'quality_checked': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('FORWARD_SUMMARY', json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
