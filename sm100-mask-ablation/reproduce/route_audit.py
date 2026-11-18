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
    sys.path.insert(0, '/work/source/candidate/techniques/sparse_backends')
    from sol_attn import sol_attn
    from route_helpers import compile_trace, summarize_trace
    data = torch.load(a.input, map_location='cpu', weights_only=True)
    for key, value in data['tensor_sha256'].items():
        assert digest(data[key]) == value
    q, k, v = [data[n][:, :, :28].contiguous().cuda() for n in ('q','k','v')]
    mask = data['exact_block_mask'].cuda()
    report = dict(input=a.input.name, shape=list(q.shape), denoise_step=data['denoise_step'],
                  layer=data['layer'], options=data['options'], timing_samples=0, routes={})
    for name, protected in [('none', None), ('protected', mask)]:
        expected = sol_attn(q, k, v, **data['options'], exact_block_mask=protected)
        actual, trace = compile_trace(q, k, v, data['options'], protected, 'native', a.output.parent/'generated'/name)
        assert_bytes(actual, expected)
        stats, bits = summarize_trace(trace, protected, (q.shape[1]+63)//64)
        stats['bitwise_equal_to_uninstrumented'] = True
        report['routes'][name] = stats
        del actual, expected, trace, bits
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
