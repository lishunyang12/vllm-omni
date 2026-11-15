"""Generate one matched quality case; save lossless pixels before MP4 encoding."""
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

    spec = json.loads((args.work / 'matrix.json').read_text())
    case = spec['cases'][args.model_key]
    variant = spec['variants'][args.variant]
    model = json.loads((args.work / 'evidence/models.json').read_text())[args.model_key]
    out = args.work / 'outputs' / args.model_key / args.variant
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
                    start_unix=time.time(), status='initializing')
    def save():
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save()
    start = time.perf_counter()
    omni = Omni(**kwargs)
    metadata['initialization_seconds'] = time.perf_counter() - start
    metadata['status'] = 'generating'
    save()
    try:
        start = time.perf_counter()
        outputs = omni.generate({'prompt': case['prompt'], 'negative_prompt': case['negative_prompt']},
                                OmniDiffusionSamplingParams(**params))
        metadata['generation_seconds'] = time.perf_counter() - start
        output = outputs[0] if isinstance(outputs, list) else outputs
        frames = output.images
        # H3 returns explicit uint8 FHWC frames; Wan/image pipelines return PIL.
        # Do not infer float tensor normalization from its values.
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
                raise ValueError(f'Expected uint8 FHWC RGB frames, received {frames.shape} {frames.dtype}')
            frames = [Image.fromarray(frame) for frame in frames]
        if isinstance(frames, Image.Image):
            frames = [frames]
        if not isinstance(frames, (list, tuple)) or not all(isinstance(frame, Image.Image) for frame in frames):
            raise TypeError(f'Expected lossless PIL frames, received {type(frames)}; refusing guessed normalization')
        if len(frames) != case['frames']:
            raise ValueError(f"Expected {case['frames']} frames, got {len(frames)}")
        frame_dir = out / 'frames'
        frame_dir.mkdir()
        hashes = []
        for index, frame in enumerate(frames):
            frame = frame.convert('RGB')
            assert frame.size == (case['width'], case['height']), frame.size
            frame.save(frame_dir / f'{index:05d}.png')
            hashes.append(hashlib.sha256(np.asarray(frame).tobytes()).hexdigest())
        metadata.update(status='completed', frame_count=len(frames), pixel_sha256=hashes)
        save()
        print('QUALITY_CASE_COMPLETED', out, flush=True)
    finally:
        omni.close()


if __name__ == '__main__':
    main()
