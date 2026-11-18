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
    from sequence_probe import install
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
    expected = json.loads((Path(__file__).parent / 'input-manifest.json').read_text())
    request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
    if request_sha256 != expected['request_sha256']:
        raise RuntimeError('Request differs from the published benchmark input')
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
    if assets != expected['assets']:
        raise RuntimeError('Reference asset content differs from the published benchmark input')
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    source_root = Path(sys.modules['vllm_omni'].__file__).resolve().parent.parent
    source_hashes = {}
    for name in ['vllm_omni/diffusion/attention/ops/sol_attn.py',
                 'vllm_omni/diffusion/attention/backends/sol_attn.py']:
        path = source_root / name
        if path.exists():
            source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    from sol_attn import interface as sol_interface
    import torch
    sol_interface_path = Path(sol_interface.__file__)
    upstream_runtime = dict(interface=str(sol_interface_path),
                            interface_sha256=hashlib.sha256(sol_interface_path.read_bytes()).hexdigest(),
                            selected_backend=sol_interface._backend_for_arch(torch.cuda.get_device_capability(0)),
                            capability=list(torch.cuda.get_device_capability(0)))
    upstream_runtime['candidate_sha256'] = {n: hashlib.sha256((sol_interface_path.parent / n).read_bytes()).hexdigest() for n in ('interface.py', 'sm100/softmax.py', 'sm100/mainloop.py')}
    gpu_count = int(os.environ.get("H3_GPU_COUNT", "2"))
    settings = dict(model=model, dtype='bfloat16', trust_remote_code=True, num_gpus=gpu_count,
                    tensor_parallel_size=1, ulysses_degree=gpu_count, ring_degree=1,
                    text_encoder_tp_size=gpu_count, vae_patch_parallel_size=gpu_count,
                    vae_use_tiling=True, enforce_eager=True, cache_backend='none',
                    diffusion_attention_backend=a.backend,
                    task_type='ref2va' if a.task == 'ref2va' else 'fl2va',
                    enable_cpu_offload=a.cpu_offload,
                    init_timeout=1800, stage_init_timeout=1800)
    protocol = dict(request=request, request_sha256=request_sha256,
                    assets=assets, model=model_info, engine=settings,
                    runtime_commit=os.environ['H3_SOURCE_COMMIT'],
                    upstream_commit=os.environ["H3_UPSTREAM_COMMIT"],
                    source_hashes=source_hashes, upstream_runtime=upstream_runtime,
                    variant=os.environ['H3_VARIANT'],
                    gpu_freq_request_mhz=os.environ.get('H3_GPU_FREQ_REQUEST_MHZ'),
                    physical_gpu_indices=os.environ.get('H3_PHYSICAL_GPU_INDICES'),
                    test_fix_commit='731ea29270e2d9f30e68f94d43a31021b4727eef',
                    measurement='Dense reference trajectory; captures at steps 0/10/25; step-10 paired forward with 3 warmup/9 measured rounds')
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
    reports = [json.loads((out / f'rank-{r}.json').read_text()) for r in range(gpu_count)]
    assert all(r['capture_complete'] and r['finite'] and r['step10_dense_bytes_match'] for r in reports)
    assert reports[0]['captured_steps'] == [0, 10, 25]
    forward = [json.loads((out / 'forward-step-10' / f'rank-{r}.json').read_text()) for r in range(gpu_count)]
    summary = dict(capture_complete=True, captured_steps=[0,10,25], trajectory='dense',
                   terminal_exception=terminal_exception, forward_step=10, runs={})
    for variant in ['dense','baseline','candidate']:
        rows = [[s for s in r['samples'] if s['variant']==variant] for r in forward]
        assert all(len(x)==9 for x in rows)
        times = [max(rs[i]['wall_s'] for rs in rows) for i in range(9)]
        summary['runs'][variant] = dict(samples_s=times, median_s=statistics.median(times),
            cv_percent=100*statistics.stdev(times)/statistics.mean(times))
    assert all(r['candidate_bitwise_equal'] for r in forward)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SEQUENCE_SUMMARY',json.dumps(summary),flush=True)

if __name__ == '__main__':
    main()
