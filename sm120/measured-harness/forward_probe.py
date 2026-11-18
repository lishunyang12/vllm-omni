"""Measure an actual H3 first-step forward, then stop before sampler advancement."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time


def install():
    import torch
    import torch.distributed as dist
    from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    from vllm_omni.diffusion.attention.backends.sol_attn import SolAttnImpl
    from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl
    from sol_attn import get_sol_attn_backend
    original = MiniMaxH3DiTModel.forward
    if getattr(original, '_h3_benchmark_probe', False):
        return

    def describe(value):
        if isinstance(value, torch.Tensor):
            raw = value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(raw).hexdigest()}
        if isinstance(value, dict):
            return {str(k): describe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [describe(v) for v in value]
        if dataclasses.is_dataclass(value):
            return {f.name: describe(getattr(value, f.name)) for f in dataclasses.fields(value)}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f'Unrecorded forward input type: {type(value)}')

    def measured(self, *args, **kwargs):
        root = Path(os.environ['H3_FORWARD_RESULT'])
        if not (root / 'armed').exists():
            return original(self, *args, **kwargs)
        if not is_forward_context_available() or get_forward_context().denoise_step_idx != 0:
            raise RuntimeError('Benchmark expected first conditional denoising step 0')
        rank, world = dist.get_rank(), dist.get_world_size()
        expected_world = int(os.environ.get("H3_GPU_COUNT", "2"))
        if world != expected_world or len(self.blocks) != 50:
            raise RuntimeError(f'Expected full 50-layer DiT and SP{expected_world}, got {len(self.blocks)}, world {world}')
        if args:
            raise RuntimeError('Unexpected positional forward arguments')
        meta = describe(kwargs)
        config = get_forward_context().omni_diffusion_config
        parallel = config.parallel_config
        if parallel.ulysses_degree != expected_world or parallel.tensor_parallel_size != 1:
            raise RuntimeError(f'Benchmark requires TP1 + Ulysses{expected_world}, got {parallel}')
        backends = {}
        for name, module in self.named_modules():
            if hasattr(module, 'attn_backend'):
                backends[name] = {'backend': module.attn_backend.get_name(), 'impl': type(module.attention).__name__}
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        warmup = int(os.environ.get('H3_FORWARD_WARMUP', '3'))
        repeats = int(os.environ.get('H3_FORWARD_REPEATS', '9'))
        sparse_impls = [m.attention for m in self.modules() if hasattr(m, 'attention') and isinstance(m.attention, SolAttnImpl)]
        if len(sparse_impls) != 50:
            raise RuntimeError(f'Expected 50 SOL DiT attention layers, found {len(sparse_impls)}')
        for impl in sparse_impls:
            impl.dense_attention = CuDNNAttentionImpl(1, 128, impl.softmax_scale)
        samples = []
        torch.cuda.reset_peak_memory_stats()
        last_outputs = {}
        for i in range(warmup + repeats):
            order = ['dense', 'sol'] if i % 2 == 0 else ['sol', 'dense']
            for variant in order:
                for impl in sparse_impls:
                    impl.dense_layers = set(range(50)) if variant == 'dense' else set()
                dist.barrier()
                torch.cuda.synchronize()
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter()
                start.record()
                result = original(self, **kwargs)
                stop.record()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - wall
                event_s = start.elapsed_time(stop) / 1000
                if i >= warmup:
                    samples.append({'index': i - warmup, 'variant': variant, 'wall_s': elapsed, 'cuda_event_s': event_s})
                if i == warmup:
                    torch.save([x.detach().cpu() for x in result], root / f'output-{variant}-first-rank-{rank}.pt')
                last_outputs[variant] = result
                print(f'H3_PAIRED_REPLAY rank={rank} variant={variant} warmup={i < warmup} iteration={i} wall_s={elapsed:.6f}', flush=True)
        for variant, output in last_outputs.items():
            torch.save([x.detach().cpu() for x in output], root / f'output-{variant}-last-rank-{rank}.pt')
        finite = all(bool(torch.isfinite(t).all().item()) for tensors in last_outputs.values() for t in tensors)
        if not finite:
            raise RuntimeError('Non-finite DiT output')
        if describe(kwargs) != meta:
            raise RuntimeError('Repeated forward mutated its inputs')
        report = {'rank': rank, 'world_size': world, 'gpu': props.name,
                  'capability': list(torch.cuda.get_device_capability()), 'memory_bytes': props.total_memory,
                  'warmup': warmup, 'repeats': repeats, 'denoise_step': 0,
                  'sol_backend': get_sol_attn_backend(device), 'dense_backend': 'CUDNN_ATTN',
                  'input': meta, 'backends': backends, 'samples': samples,
                  'output': describe(result), 'finite': finite,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'parallel_config': dataclasses.asdict(parallel), 'intentional_abort': True}
        temp = root / f'rank-{rank}.json.tmp'
        temp.write_text(json.dumps(report, indent=2, default=str) + '\n')
        temp.replace(root / f'rank-{rank}.json')
        dist.barrier()
        raise RuntimeError('H3_FORWARD_BENCHMARK_COMPLETE: intentional stop before sampler advancement and VAE decode')

    measured._h3_benchmark_probe = True
    MiniMaxH3DiTModel.forward = measured
