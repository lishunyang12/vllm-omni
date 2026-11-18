"""Untimed exact-block route counts; trace output must match the real kernel."""
import argparse
import json
from pathlib import Path
import sys

import torch

from ablation import assert_bytes, digest


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, '/work/source/candidate-fp32/techniques/sparse_backends')
    from sol_attn import sol_attn
    from sol_attn import preprocess
    import triton
    import hashlib
    for kernel in (preprocess._reduce_kc_kernel, preprocess._reduce_vc_kernel):
        kernel.configs = [triton.Config({}, num_warps=4, num_stages=1)]
        kernel.cache.clear()
    prepared = []
    native_prepare = preprocess.prepare
    def capture_prepare(*args, **kwargs):
        values = native_prepare(*args, **kwargs)
        prepared.append(values)
        return values
    preprocess.prepare = capture_prepare
    from route_helpers import compile_trace, summarize_trace
    data = torch.load(a.input, map_location='cpu', weights_only=True)
    for key, value in data['tensor_sha256'].items():
        assert digest(data[key]) == value
    q, k, v = [data[n][:, :, :28].contiguous().cuda() for n in ('q','k','v')]
    mask = data['exact_block_mask'].cuda()
    report = dict(input=a.input.name, shape=list(q.shape), denoise_step=data['denoise_step'],
                  layer=data['layer'], options=data['options'], timing_samples=0, routes={}, pooling='FP32 accumulation, BF16 output,4warps/1stage',
                  preprocess_sha256=hashlib.sha256(Path(preprocess.__file__).read_bytes()).hexdigest())
    for name, protected in [('none', None), ('protected', mask)]:
        prepared.clear()
        expected = sol_attn(q, k, v, **data['options'], exact_block_mask=protected)
        actual, trace = compile_trace(q, k, v, data['options'], protected, 'native', a.output.parent/'generated'/name)
        assert_bytes(actual, expected)
        assert len(prepared) == 2
        prep_hashes = [[digest(value) for value in values] for values in prepared]
        assert prep_hashes[0] == prep_hashes[1]

        stats, bits = summarize_trace(trace, protected, (q.shape[1]+63)//64)
        stats['bitwise_equal_to_uninstrumented'] = True
        stats['prepared_sha256'] = prep_hashes[0]
        report['routes'][name] = stats
        del actual, expected, trace, bits
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
