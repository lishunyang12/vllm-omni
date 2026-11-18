"""Standalone host-overhead diagnostic; never changes serving GC policy."""
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from vllm_omni.diffusion.attention.ops.sol_residual import merge_sol_residual
from final_norm_residual import merge_sol_residual as experimental_copy

data = torch.load('/work/profile/results/ref2va-SOL_ATTN-capture/residual-input-rank0.pt', weights_only=True)
args = [t.cuda() if isinstance(t, torch.Tensor) else t for t in data['args']]
events = []
started = {}


def callback(phase, info):
    generation = info['generation']
    if phase == 'start':
        started[generation] = time.perf_counter()
    else:
        events.append(dict(generation=generation, duration_ms=1000*(time.perf_counter()-started[generation])))


report = {}
gc.callbacks.append(callback)
try:
    with torch.inference_mode():
        for name, fn, enabled in [('canonical_enabled', merge_sol_residual, True),
                                  ('copy_enabled', experimental_copy, True),
                                  ('canonical_disabled', merge_sol_residual, False),
                                  ('copy_disabled', experimental_copy, False)]:
            gc.collect()
            if enabled:
                gc.enable()
            else:
                gc.disable()
            samples = []
            start_events = len(events)
            for i in range(12):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = fn(*args)
                torch.cuda.synchronize()
                if i >= 3:
                    samples.append(1000*(time.perf_counter()-start))
            report[name] = dict(median_ms=statistics.median(samples), samples_ms=samples,
                                gc_events=events[start_events:])
            print(name,json.dumps(report[name]),flush=True)
finally:
    gc.enable()
    gc.callbacks.remove(callback)
Path('/work/fused/gc-probe.json').write_text(json.dumps(report,indent=2)+'\n')
