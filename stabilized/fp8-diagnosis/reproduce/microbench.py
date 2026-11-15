"""Isolate eager host dispatch on Qwen's observed GEMM shapes; synthetic weights."""
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import statistics
import sys
import time

import torch
from flashinfer.autotuner import AutoTuner, autotune
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutedslMxfp8LinearKernel
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize


def time_calls(fn, repeats=100, rounds=7):
    values = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1e6/repeats)
    return {'median_us_per_call':statistics.median(values), 'rounds_us_per_call':values,
            'calls_per_round':repeats}


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cache = Path('/work/xdg-cache/vllm/flashinfer_autotune/diffusion/fdbba43075f1737a/rank0.json')
    report = {'kind':'Synthetic GEMM shape diagnostic, not model latency or a validated optimization',
              'gpu':torch.cuda.get_device_name(0), 'capability':torch.cuda.get_device_capability(0),
              'cache_loaded':AutoTuner.get().load_configs(str(cache)) if cache.exists() else False,
              'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest() if cache.exists() else None,
              'shapes':[]}
    for m,n,k in [(5,12288,3072),(36,12288,3072),(1024,12288,3072),(1024,3072,12288)]:
        shape_dir = out / f'M{m}-N{n}-K{k}'; shape_dir.mkdir(exist_ok=True)
        x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        w = torch.randn(n,k,device='cuda',dtype=torch.bfloat16) * .02
        bias = torch.randn(n,device='cuda',dtype=torch.bfloat16)
        wq,ws = mxfp8_e4m3_quantize(w.contiguous())
        layer = torch.nn.Module(); layer.params_dtype=torch.bfloat16
        layer.register_parameter('weight',torch.nn.Parameter(wq,requires_grad=False))
        layer.register_parameter('weight_scale',torch.nn.Parameter(ws,requires_grad=False))
        kernel = FlashInferCutedslMxfp8LinearKernel(Mxfp8LinearLayerConfig((n,k)))
        kernel.process_weights_after_loading(layer)
        bf16 = lambda: torch.nn.functional.linear(x,w,bias)
        mxfp8 = lambda: kernel.apply_weights(layer,x,bias)
        with autotune(tune_mode=True): mxfp8()
        for _ in range(30): bf16(); mxfp8()
        torch.cuda.synchronize()
        row = {'M':m,'N':n,'K':k,'kernel':type(kernel).__name__,
               'bf16_eager':time_calls(bf16),'mxfp8_eager':time_calls(mxfp8)}
        # Graph replay is a diagnostic control that removes most Python dispatch.
        # It is NOT an end-to-end model graph or a proposed production change.
        for name,fn in [('bf16',bf16),('mxfp8',mxfp8)]:
            try:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): graph_output=fn()
                for _ in range(20): graph.replay()
                row[name+'_graph_replay']=time_calls(graph.replay)
                del graph_output,graph
            except Exception as e:
                row[name+'_graph_replay']={'error':repr(e)}
        profile=cProfile.Profile()
        torch.cuda.synchronize(); profile.enable()
        for _ in range(200): mxfp8()
        profile.disable(); torch.cuda.synchronize()
        stream=io.StringIO(); stats=pstats.Stats(profile,stream=stream).sort_stats('tottime')
        stats.print_stats(35)
        (shape_dir/'python-profile.txt').write_text(stream.getvalue())
        functions=[]
        for (filename,line,function),(primitive,calls,self_seconds,cumulative_seconds,callers) in stats.stats.items():
            functions.append({'file':filename.split('/site-packages/')[-1] if '/site-packages/' in filename else Path(filename).name,
                              'line':line,'function':function,'calls':calls,'primitive_calls':primitive,
                              'self_seconds':self_seconds,'cumulative_seconds':cumulative_seconds})
        functions.sort(key=lambda r:-r['self_seconds'])
        (shape_dir/'python-functions.json').write_text(json.dumps(functions,indent=2)+'\n')
        report['shapes'].append(row)
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print('MICROBENCH',json.dumps(row),flush=True)
        del layer,kernel,wq,ws,w,x,bias,bf16,mxfp8
        torch.cuda.empty_cache()


if __name__=='__main__':main()
