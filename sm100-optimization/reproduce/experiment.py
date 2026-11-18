"""Correctness gates and same-input latency of narrowly scoped SM100 candidates."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch
from sol_attn import interface
import replay
from candidates import make_candidate


@torch.inference_mode()
def validate(functions, output):
    spec = importlib.util.spec_from_file_location('oracle', Path(interface.__file__).parents[1] / 'tests/test_exact_block_mask.py')
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    rows = []
    for tokens, tau in [(63, 1e6), (257, 1e6), (513, 1.5)]:
        for threshold in ['diag', 'exact']:
            torch.manual_seed(71)
            q, k, v = [torch.randn(1, tokens, 2, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
            mask = torch.zeros((tokens+63)//64, device='cuda', dtype=torch.bool)
            mask[0] = mask[-1] = True
            options = dict(scale=128**-.5, tau=tau, thresh_type=threshold, exact_block_mask=mask)
            stock = interface.sol_attn(q, k, v, **options)
            expected = oracle.hybrid_oracle(q, k, v, mask, tau, threshold)
            for name, fn in functions.items():
                result = fn(q, k, v, **options)
                torch.testing.assert_close(result.float(), expected, atol=.02, rtol=.02)
                torch.testing.assert_close(result, stock, atol=0, rtol=0)
                rows.append(dict(name=name, tokens=tokens, tau=tau, threshold=threshold,
                                 bitwise_equal=bool(torch.equal(result, stock)), oracle_passed=True))
                replay.save(output, dict(cases=rows, passed=len(rows)))
    # Long iteration streams, all-exact and no protection, identity scales,
    # changing row maxima, zeros and tiny/subnormal values.
    for kind in ['random', 'all_exact', 'zero_q', 'zero_v', 'tiny_v', 'subnormal_v']:
        torch.manual_seed(153)
        q, k, v = [torch.randn(1, 4097, 2, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        mask = torch.zeros(65, device='cuda', dtype=torch.bool)
        if kind == 'all_exact': mask.fill_(True)
        if kind == 'zero_q': q.zero_()
        if kind == 'zero_v': v.zero_()
        if kind == 'tiny_v': v.mul_(2.0**-120)
        if kind == 'subnormal_v': v.mul_(2.0**-130)
        options = dict(scale=128**-.5, tau=1., thresh_type='diag', exact_block_mask=mask)
        stock = interface.sol_attn(q, k, v, **options)
        for name, fn in functions.items():
            result = fn(q, k, v, **options)
            torch.testing.assert_close(result, stock, atol=0, rtol=0)
            rows.append(dict(name=name, tokens=4097, kind=kind, bitwise_equal=bool(torch.equal(result, stock))))
            replay.save(output, dict(cases=rows, passed=len(rows)))
    print('VALIDATION_PASSED', len(rows), flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--candidates', default='skip_identity,wide32,skip_identity_wide32')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=15)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() not in [(10, 0), (10, 3)]:
        raise RuntimeError('This experiment requires the SM100 kernel family')
    functions, provenance = {}, {}
    for name in args.candidates.split(','):
        functions[name], provenance[name] = make_candidate(name, args.output.parent/'generated')
    if args.validate:
        validate(functions, args.output)
        return
    if args.input is None: parser.error('--input required')
    data = torch.load(args.input, map_location='cpu', weights_only=True)
    for name in ['q','k','v','exact_block_mask']:
        t=data[name]
        if t is not None:
            assert hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == data['tensor_sha256'][name]
    q,k,v=[data[name][:,:,:28].contiguous().cuda() for name in ['q','k','v']]
    mask=data['exact_block_mask'].cuda() if data['exact_block_mask'] is not None else None
    callables=replay.functions(q,k,v,data['options'],mask)
    callables.pop('warp')
    for name, fn in functions.items():
        callables[name]=lambda fn=fn: fn(q,k,v,**data['options'],exact_block_mask=mask)
    baseline=callables['native']()
    quality={}
    for name in functions:
        out=callables[name]()
        torch.testing.assert_close(out,baseline,atol=0,rtol=0)
        quality[name]=replay.compare(out,baseline)
    report=dict(layer=data['layer'],shape=list(q.shape),capture_sha256=data['tensor_sha256'],
                gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                warmup=args.warmup,repeats=args.repeats,quality=quality,provenance=provenance,samples=[])
    replay.save(args.output,report)
    names=list(callables)
    for i in range(args.warmup+args.repeats):
        offset=i%len(names);order=names[offset:]+names[:offset]
        if (i//len(names))%2: order=order[::-1]
        for name in order:
            torch.cuda.synchronize()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start_unix_s,wall=time.time(),time.perf_counter()
            begin.record();result=callables[name]();end.record();torch.cuda.synchronize()
            wall_ms=1000*(time.perf_counter()-wall)
            if i>=args.warmup:
                report['samples'].append(dict(index=i-args.warmup,variant=name,event_ms=begin.elapsed_time(end),
                                              wall_ms=wall_ms,start_unix_s=start_unix_s))
        print('ITERATION',i,flush=True)
    for name in functions:
        torch.testing.assert_close(callables[name](),baseline,atol=0,rtol=0)
    report['runs']={}
    for name in names:
        times=[s['event_ms'] for s in report['samples'] if s['variant']==name]
        report['runs'][name]=dict(median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),
                                  cv_percent=100*statistics.stdev(times)/statistics.mean(times))
    replay.save(args.output,report)
    print('RESULT',json.dumps(report['runs']),flush=True)


if __name__=='__main__':main()
