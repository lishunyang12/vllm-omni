"""Measure an actual H3 step-10 forward and return Dense output to the sampler."""
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
        if not is_forward_context_available() or get_forward_context().denoise_step_idx != 10:
            raise RuntimeError('Benchmark expected conditional denoising step 10')
        root = root / 'forward-step-10'
        root.mkdir(parents=True, exist_ok=True)
        rank, world = dist.get_rank(), dist.get_world_size()
        expected_world = int(os.environ.get("H3_GPU_COUNT", "2"))
        if world != expected_world or len(self.blocks) != 50:
            raise RuntimeError(f'Expected full 50-layer DiT and SP{expected_world}, got {len(self.blocks)}, world {world}')
        if args:
            raise RuntimeError('Unexpected positional forward arguments')
        meta = describe(kwargs)
        config = get_forward_context().omni_diffusion_config
        if config.cache_backend not in (None, 'none'):
            raise RuntimeError('Step-10 benchmark requires disabled diffusion cache')
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
        import sys
        import tempfile
        import sol_attn as sol_package
        sys.path.insert(0, "/work/source/candidate/techniques/sparse_backends/benchmarks")
        from benchmark_sm100 import assert_bitwise_equal, baseline
        baseline_files = tempfile.TemporaryDirectory(prefix="sol-forward-baseline-")
        old_sol, baseline_hashes = baseline(
            Path("/work/source/extension-fp32/techniques/sparse_backends"), Path(baseline_files.name))
        candidate_sol = sol_package.sol_attn
        # Fix pooling config as well as precision to keep work and overhead matched.
        import importlib
        import triton
        for function in (old_sol, candidate_sol):
            package = function.__module__.rsplit('.', 1)[0]
            prep = importlib.import_module(package + '.preprocess')
            for kernel in (prep._reduce_kc_kernel, prep._reduce_vc_kernel):
                kernel.configs = [triton.Config({}, num_warps=4, num_stages=1)]
                kernel.cache.clear()
        samples = []
        torch.cuda.reset_peak_memory_stats()
        last_outputs = {}
        first_outputs = {}
        for i in range(warmup + repeats):
            names = ['dense', 'baseline', 'candidate']
            offset = i % len(names)
            order = names[offset:] + names[:offset]
            if i % 2:
                order.reverse()
            for variant in order:
                sol_package.sol_attn = old_sol if variant == 'baseline' else candidate_sol
                for impl in sparse_impls:
                    impl.dense_layers = set(range(50)) if variant == 'dense' else set()
                dist.barrier()
                torch.cuda.synchronize()
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_unix_s = time.time()
                wall = time.perf_counter()
                start.record()
                result = original(self, **kwargs)
                stop.record()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - wall
                event_s = start.elapsed_time(stop) / 1000
                if i >= warmup:
                    samples.append({'index': i - warmup, 'variant': variant, 'wall_s': elapsed,
                                    'cuda_event_s': event_s, 'start_unix_s': start_unix_s})
                if i == warmup:
                    first_outputs[variant] = [x.detach().cpu() for x in result]
                    torch.save(first_outputs[variant], root / f'output-{variant}-first-rank-{rank}.pt')
                last_outputs[variant] = result
                print(f'H3_PAIRED_REPLAY rank={rank} variant={variant} warmup={i < warmup} iteration={i} wall_s={elapsed:.6f}', flush=True)
        for variant, output in last_outputs.items():
            torch.save([x.detach().cpu() for x in output], root / f'output-{variant}-last-rank-{rank}.pt')
        sol_package.sol_attn = candidate_sol
        for outputs in (first_outputs, last_outputs):
            assert len(outputs['baseline']) == len(outputs['candidate'])
            for old, new in zip(outputs['baseline'], outputs['candidate']):
                assert_bitwise_equal(new, old)
        finite = all(bool(torch.isfinite(t).all().item()) for tensors in last_outputs.values() for t in tensors)
        if not finite:
            raise RuntimeError('Non-finite DiT output')
        if describe(kwargs) != meta:
            raise RuntimeError('Repeated forward mutated its inputs')
        report = {'rank': rank, 'world_size': world, 'gpu': props.name,
                  'capability': list(torch.cuda.get_device_capability()), 'memory_bytes': props.total_memory,
                  'warmup': warmup, 'repeats': repeats, 'denoise_step': 10,
                  'sol_backend': get_sol_attn_backend(device), 'dense_backend': 'CUDNN_ATTN',
                  'baseline_kernel_sha256': baseline_hashes, 'candidate_bitwise_equal': True,
                  'input': meta, 'backends': backends, 'samples': samples,
                  'output': describe(last_outputs['dense']), 'output_variant': 'dense', 'finite': finite,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'parallel_config': dataclasses.asdict(parallel), 'returns_dense_to_sampler': True, 'cache_backend': config.cache_backend}
        temp = root / f'rank-{rank}.json.tmp'
        temp.write_text(json.dumps(report, indent=2, default=str) + '\n')
        temp.replace(root / f'rank-{rank}.json')
        dist.barrier()
        for impl in sparse_impls:
            impl.dense_layers = set(range(50))
        return last_outputs['dense']

    measured._h3_benchmark_probe = True
    MiniMaxH3DiTModel.forward = measured
