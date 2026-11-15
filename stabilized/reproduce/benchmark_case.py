"""Measure fixed warmup and repeated requests with one resident engine per case."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path)
    parser.add_argument('model_key', choices=['h3_image', 'qwen_image', 'wan22', 'h3_video'])
    parser.add_argument('variant', choices=['base_bf16', 'pr_bf16', 'pr_mxfp8', 'pr_fp8_block'])
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    import vllm_omni
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    spec = json.loads((args.work / 'stabilized/matrix.json').read_text())
    protocol = json.loads((args.work / 'stabilized/protocol.json').read_text())
    case = spec['cases'][args.model_key]
    variant = spec['variants'][args.variant]
    model = json.loads((args.work / 'evidence/models.json').read_text())[args.model_key]
    out = args.work / 'stabilized/results' / args.model_key / args.variant
    out.mkdir(parents=True, exist_ok=False)
    expected_source = args.work / 'source' / variant['source']
    assert Path(vllm_omni.__file__).resolve().is_relative_to(expected_source.resolve()), vllm_omni.__file__
    parallel = DiffusionParallelConfig(ulysses_degree=4, ring_degree=1, tensor_parallel_size=1,
                                      text_encoder_tp_size=4 if args.model_key in ('h3_image', 'h3_video') else 1)
    kwargs = dict(model=model['path'], num_gpus=4, parallel_config=parallel, dtype='bfloat16',
                  enforce_eager=True, init_timeout=900, stage_init_timeout=900)
    if variant['quantization']:
        # Keep encoders and VAE identical. Quantize the diffusion transformer only.
        kwargs['quantization_config'] = {'transformer': {'method': variant['quantization']}, 'text_encoder': None}
    if args.model_key == 'h3_video':
        # The pinned H3 checkpoint supplies the native video/audio VAE modules.
        kwargs['trust_remote_code'] = True
    if args.model_key == 'wan22':
        kwargs['flow_shift'] = 5.0
    params = dict(height=case['height'], width=case['width'], seed=case['seed'],
                  generator=torch.Generator(device='cpu').manual_seed(case['seed']),
                  num_inference_steps=case['steps'], guidance_scale=case['guidance_scale'],
                  true_cfg_scale=case['true_cfg_scale'], num_frames=case['frames'],
                  output_type='np' if args.model_key == 'h3_video' else 'pil', fps=case['fps'])
    if args.model_key == 'h3_video':
        params['extra_args'] = case['extra_args']
    source_paths = ['vllm_omni/quantization/factory.py', 'vllm_omni/quantization/mxfp8_config.py',
                    'vllm_omni/quantization/mxfp8_cuda.py', 'vllm_omni/entrypoints/async_omni.py',
                    'vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3_image.py']
    source_hashes = {name: hashlib.sha256((expected_source / name).read_bytes()).hexdigest()
                     for name in source_paths if (expected_source / name).is_file()}
    metadata = dict(source_sha256=source_hashes, case=case, variant=variant, model=model, source=vllm_omni.__file__,
                    quantization_config=kwargs.get('quantization_config'),
                    parallelism=dict(ulysses_degree=4, tensor_parallel_size=1,
                                     text_encoder_tp_size=parallel.text_encoder_tp_size),
                    trust_remote_code=kwargs.get('trust_remote_code', False),
                    enforce_eager=True, torch=torch.__version__, cuda=torch.version.cuda,
                    start_unix=time.time(), status='initializing', protocol=protocol,
                    gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    gpu_capabilities=[torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())],
                    requests=[])
    metadata['benchmark_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def save():
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save()
    start = time.perf_counter()
    omni = Omni(**kwargs)
    metadata['initialization_seconds'] = time.perf_counter() - start
    metadata['status'] = 'generating'
    save()
    try:
        warmups = protocol['warmup_requests']
        measurements = protocol['measured_requests']
        for index in range(warmups + measurements):
            phase = 'warmup' if index < warmups else 'measured'
            # A fresh generator reproduces the fixed seed rather than advancing RNG state.
            params['generator'] = torch.Generator(device='cpu').manual_seed(case['seed'])
            sampling = OmniDiffusionSamplingParams(**params)
            started = time.perf_counter()
            outputs = omni.generate({'prompt': case['prompt'], 'negative_prompt': case['negative_prompt']}, sampling)
            elapsed = time.perf_counter() - started
            if not isinstance(outputs, list) or len(outputs) != 1:
                raise ValueError('Expected exactly one completed output')
            frames = outputs[0].images
            while True:
                if isinstance(frames, dict):
                    frames = frames['video']
                elif isinstance(frames, (list, tuple)) and len(frames) == 1 and not isinstance(frames[0], Image.Image):
                    frames = frames[0]
                else:
                    break
            if isinstance(frames, torch.Tensor):
                frames = frames.detach().cpu().numpy()
            if isinstance(frames, np.ndarray):
                if frames.ndim == 5 and frames.shape[0] == 1:
                    frames = frames[0]
                if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
                    raise ValueError(f'Expected uint8 FHWC RGB, received {frames.shape} {frames.dtype}')
                frames = [Image.fromarray(frame) for frame in frames]
            if isinstance(frames, Image.Image):
                frames = [frames]
            if not isinstance(frames, (list, tuple)) or len(frames) != case['frames']:
                raise ValueError('Unexpected output frame count/type')
            if not all(isinstance(f, Image.Image) and f.size == (case['width'],case['height']) for f in frames):
                raise ValueError('Unexpected output frame size/type')
            record = {'phase':phase, 'index':index, 'seconds':elapsed, 'frame_count':len(frames)}
            if index in (warmups, warmups + measurements - 1):
                record['pixel_sha256'] = [hashlib.sha256(np.asarray(f.convert('RGB')).tobytes()).hexdigest() for f in frames]
            if index == warmups:
                # Viewing evidence outside the measured interval; preserve native dimensions.
                for frame_index in sorted({0,len(frames)//2,len(frames)-1}):
                    frames[frame_index].save(out/f'frame-{frame_index:05d}.png')
            metadata['requests'].append(record)
            metadata['status'] = phase
            save()
            print('LATENCY_SAMPLE', args.model_key,args.variant,phase,index,elapsed,flush=True)
            del frames, outputs, sampling
        values = np.asarray([r['seconds'] for r in metadata['requests'] if r['phase']=='measured'])
        p50, p95 = np.percentile(values,[50,95],method='linear')
        first, second = np.median(values[:len(values)//2]),np.median(values[len(values)//2:])
        stats = dict(n=len(values), p50_seconds=float(p50),p95_seconds=float(p95),
                     mean_seconds=float(values.mean()),stddev_seconds=float(values.std(ddof=1)),
                     min_seconds=float(values.min()),max_seconds=float(values.max()),
                     coefficient_of_variation=float(values.std(ddof=1)/values.mean()),
                     half_median_relative_change=float(abs(second-first)/first),p95_over_p50=float(p95/p50))
        limits = protocol['stability_limits']
        checks = {k:stats[k] <= limits[k+'_max'] for k in ('coefficient_of_variation','half_median_relative_change','p95_over_p50')}
        metadata.update(status='completed',summary=stats,stability_checks=checks,stable=all(checks.values()))
        metadata['first_last_measured_pixels_identical'] = metadata['requests'][warmups]['pixel_sha256']==metadata['requests'][-1]['pixel_sha256']
        save()
        print('LATENCY_CASE_COMPLETED',args.model_key,args.variant,json.dumps(stats),metadata['stable'],flush=True)
    except BaseException as error:
        metadata.update(status='failed',error=repr(error))
        save()
        raise
    finally:
        omni.close()


if __name__ == '__main__':
    main()
