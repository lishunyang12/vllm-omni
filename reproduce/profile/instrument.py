"""External instrumentation only: no changes to production attention math."""
import contextlib
import functools
import json
import os
import time
from pathlib import Path

import torch

state = dict(mode=None, events=[], routing=[], attention_index=0, attention_calls=0,
             trace_done=False, impl_names={})


def register_model(model):
    state['impl_names'] = {id(module.attention): name for name, module in model.named_modules()
                           if hasattr(module, 'attn_backend') and hasattr(module, 'attention')}


def root():
    return Path(os.environ['H3_FORWARD_RESULT'])


@contextlib.contextmanager
def span(name):
    mode = state['mode']
    if mode is None:
        yield
        return
    if mode == 'events':
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        start = time.perf_counter()
        yield
        b.record()
        state['events'].append((name, state['attention_index'], a, b, time.perf_counter() - start))
    else:
        with torch.profiler.record_function('H3/' + name):
            yield


def wrap(obj, name, label, after=None):
    original = getattr(obj, name)

    @functools.wraps(original)
    def measured(*args, **kwargs):
        with span(label):
            result = original(*args, **kwargs)
        if after is not None and state['mode'] == 'events':
            after(result, args, kwargs)
        return result

    setattr(obj, name, measured)


def record_routing(result, args, kwargs):
    counts = result.sum(-1)
    state['routing'].append(dict(
        attention_index=state['attention_index'], shape=list(result.shape),
        selected=counts.sum(), capacity=result.numel(), min_blocks=counts.min(),
        max_blocks=counts.max(), full_rows=(counts == result.shape[-1]).sum(),
        total_rows=counts.numel()))


def install():
    from vllm_omni.diffusion.attention.ops import sol_attn as sol
    from vllm_omni.diffusion.attention.ops import cudnn_block_sparse as bsa
    from vllm_omni.diffusion.attention.backends.sol_attn import SolAttnImpl
    from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl
    from vllm_omni.diffusion.distributed import comm

    original_require = sol.require_sol_bsa

    def require(*args, **kwargs):
        prepare = original_require(*args, **kwargs)

        def timed_prepare(*a, **kw):
            with span('sol.prepare'):
                return prepare(*a, **kw)
        return timed_prepare

    sol.require_sol_bsa = require
    wrap(sol, '_sol_bsa_block_map', 'sol.routing', record_routing)
    wrap(sol, '_merge_sol_bsa_residual', 'sol.residual_merge')
    timed_merge = sol._merge_sol_bsa_residual

    def capture_merge(*args, **kwargs):
        if (state['mode'] == 'capture' and torch.distributed.get_rank() == 0
                and state['attention_index'] == int(os.environ.get('H3_CAPTURE_BSA', '49'))):
            path = root() / 'residual-input-rank0.pt'
            if not path.exists():
                cpu_args = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in args]
                cpu_kwargs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
                torch.save(dict(args=cpu_args, kwargs=cpu_kwargs), path)
                print(f'H3_RESIDUAL_CAPTURED layer={state["attention_index"]} path={path}', flush=True)
        return timed_merge(*args, **kwargs)

    sol._merge_sol_bsa_residual = capture_merge
    wrap(sol, 'block_sparse_attn_bshd', 'sol.exact_sparse_total')
    wrap(bsa, 'block_map_to_indices', 'sol.map_to_indices')
    original_kernel_require = bsa.require_cudnn_sparse

    def kernel_require(*args, **kwargs):
        kernel = original_kernel_require(*args, **kwargs)

        def measured_kernel(*a, **kw):
            if (state['mode'] == 'capture' and torch.distributed.get_rank() == 0
                    and state['attention_index'] == int(os.environ.get('H3_CAPTURE_BSA', '49'))):
                rank = torch.distributed.get_rank()
                path = root() / f'bsa-input-rank{rank}.pt'
                if not path.exists():
                    cpu_args = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in a]
                    cpu_kwargs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
                    torch.save(dict(args=cpu_args, kwargs=cpu_kwargs), path)
                    print(f'H3_BSA_CAPTURED rank={rank} layer={state["attention_index"]} path={path}', flush=True)
            with span('sol.cudnn_bsa_kernel'):
                if state['mode'] == 'capture':
                    torch.cuda.synchronize()
                    print(f'H3_BSA_ENTER rank={torch.distributed.get_rank()} layer={state["attention_index"]}', flush=True)
                result = kernel(*a, **kw)
                if state['mode'] == 'capture':
                    torch.cuda.synchronize()
                    print(f'H3_BSA_EXIT rank={torch.distributed.get_rank()} layer={state["attention_index"]}', flush=True)
                return result
        return measured_kernel

    bsa.require_cudnn_sparse = kernel_require
    wrap(comm, 'all_to_all_4D', 'comm.all_to_all_4D')
    wrap(comm, 'all_to_all_5D', 'comm.all_to_all_5D')

    for cls, label in [(SolAttnImpl, 'attention.SOL'), (CuDNNAttentionImpl, 'attention.Dense')]:
        original = cls.forward_cuda

        def make(original, label):
            @functools.wraps(original)
            def measured(self, *args, **kwargs):
                active = state['mode'] is not None
                module_name = state['impl_names'].get(id(self), '')
                is_dit = module_name.startswith('blocks.')
                current_label = label if is_dit else label.replace('attention.', 'attention.refiner.')
                if active:
                    state['attention_index'] = int(module_name.split('.')[1]) if is_dit else -1
                target = int(os.environ.get('H3_TRACE_LAYER', '0'))
                do_trace = (state['mode'] == 'trace' and is_dit and state['attention_index'] == target
                            and torch.distributed.get_rank() == 0 and not state['trace_done'])
                if do_trace:
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=False, with_stack=False,
                                                profile_memory=False) as prof:
                        with span(current_label):
                            result = original(self, *args, **kwargs)
                        torch.cuda.synchronize()
                    path = root() / f'attention-layer{target}-rank0.json'
                    prof.export_chrome_trace(str(path))
                    (root() / f'attention-layer{target}-operators.txt').write_text(
                        prof.key_averages().table(sort_by='self_device_time_total', row_limit=60))
                    state['trace_done'] = True
                    print(f'H3_TRACE_SAVED {path}', flush=True)
                else:
                    with span(current_label):
                        result = original(self, *args, **kwargs)
                if active:
                    state['attention_calls'] += 1
                return result
            return measured
        cls.forward_cuda = make(original, label)


def run_profile(original, model, kwargs):
    rank = torch.distributed.get_rank()
    torch.distributed.barrier()
    torch.cuda.synchronize()
    mode = os.environ.get('H3_PROFILE_MODE', 'events')
    state.update(mode=mode, attention_index=0, attention_calls=0, events=[], routing=[], trace_done=False)
    start = time.perf_counter()
    result = original(model, **kwargs)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    state['mode'] = None
    events = [dict(name=n, attention_index=i, cuda_ms=a.elapsed_time(b), cpu_enqueue_ms=cpu*1000)
              for n, i, a, b, cpu in state['events']]
    routing = [{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in r.items()}
               for r in state['routing']]
    report = dict(rank=rank, mode=mode, wall_s=wall, attention_calls=state['attention_calls'],
                  events=events, routing=routing,
                  timing_note='Nested CUDA event intervals; do not sum parent and child spans. Instrumented run, not baseline latency.')
    (root()/f'profile-{mode}-rank{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'H3_PROFILE_COMPLETE rank={rank} mode={mode} wall_s={wall:.6f} events={len(events)}', flush=True)
    torch.distributed.barrier()
    return result
