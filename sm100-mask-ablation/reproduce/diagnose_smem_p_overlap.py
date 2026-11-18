"""Lightweight D diagnostic: baseline, validated serial, failed overlap, optional refinement.

No full quadratic oracle is constructed. This diagnoses a failed byte gate;
existing oracle/73-case validation remains mandatory before acceptance. The
script collects all results and exits nonzero for byte/input/repeat failures.
"""
import argparse
import hashlib
import tempfile
from pathlib import Path

import torch

from validate_design import isolated, save


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def stats(actual, expected):
    actual, expected = actual.contiguous().cpu(), expected.contiguous().cpu()
    mismatch = actual.view(torch.uint8) != expected.view(torch.uint8)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    result = dict(byte_mismatch=int(mismatch.sum()), byte_count=mismatch.numel(),
                  bitwise_equal=not bool(mismatch.any()), finite=finite, max_abs=None, rmse=None)
    if finite:
        difference = actual.double() - expected.double()
        result.update(max_abs=float(difference.abs().max()), rmse=float(difference.square().mean().sqrt()),
                      relative_l2=float(difference.norm() / expected.double().norm().clamp_min(1e-300)))
    if bool(mismatch.any()):
        index = mismatch.nonzero()[0].tolist()
        elem = [*index[:-1], index[-1] // actual.element_size()]
        result['first_difference'] = dict(byte_index=index, element_index=elem,
                                          actual=float(actual[tuple(elem)]), expected=float(expected[tuple(elem)]))
    return result


def by_block(actual, expected):
    return [dict(q_block=qb, token_start=qb * 64,
                 **stats(actual[:, qb * 64:(qb + 1) * 64], expected[:, qb * 64:(qb + 1) * 64]))
            for qb in range((actual.shape[1] + 63) // 64)]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--serial', type=Path, required=True)
    parser.add_argument('--overlap', type=Path, required=True)
    parser.add_argument('--publish-first', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--tokens', default='8193', help='comma-separated token counts')
    parser.add_argument('--masks', default='none', help='none,zero,all,disjoint')
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=71)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--threshold', choices=['diag', 'exact'], default='diag')
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--modes', default='eager', help='eager,graph')
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    assert args.repeats >= 2
    modes, masks = args.modes.split(','), args.masks.split(',')
    assert set(modes) <= {'eager', 'graph'}
    assert set(masks) <= {'none', 'zero', 'all', 'disjoint'}
    roots = dict(baseline=args.baseline, serial=args.serial, overlap=args.overlap)
    if args.publish_first:
        roots['publish_first'] = args.publish_first
    report = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  oracle_scope='No oracle in this focused diagnostic; retain the separate full validation gate.',
                  source_sha256={}, cases=[], failures=[], complete=False, passed=False)
    with tempfile.TemporaryDirectory(prefix='sol-d-overlap-diagnostic-') as tmp:
        modules = {}
        for name, root in roots.items():
            modules[name], report['source_sha256'][name] = isolated(root, Path(tmp), '_ddiag_' + name)
        save(args.report, report)
        for tokens in map(int, args.tokens.split(',')):
            for mask_kind in masks:
                generator = torch.Generator(device='cuda').manual_seed(args.seed)
                qkv = tuple(torch.randn(1, tokens, args.heads, 128, dtype=torch.bfloat16,
                                        device='cuda', generator=generator) for _ in range(3))
                mask = None
                if mask_kind != 'none':
                    mask = torch.zeros((tokens + 63) // 64, dtype=torch.bool, device='cuda')
                    if mask_kind == 'all':
                        mask.fill_(True)
                    elif mask_kind == 'disjoint':
                        mask[0] = mask[-1] = True
                        if len(mask) > 5:
                            mask[4] = True
                inputs = dict(zip(('q', 'k', 'v'), qkv))
                if mask is not None:
                    inputs['mask'] = mask
                hashes = {name: digest(value) for name, value in inputs.items()}
                options = dict(tau=args.tau, thresh_type=args.threshold, exact_block_mask=mask)
                case = dict(tokens=tokens, heads=args.heads, seed=args.seed, options={
                    'mask': mask_kind, 'tau': args.tau, 'threshold': args.threshold},
                    input_sha256=hashes, runs=[], pairwise_by_mode={})
                report['cases'].append(case)
                first, last, retained = {}, {}, []
                names = list(modules)
                for mode in modes:
                    calls = {name: (lambda name=name: modules[name].sol_attn(*qkv, **options)) for name in names}
                    if mode == 'graph':
                        for name in names:
                            calls[name]()
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.stream(stream):
                                calls[name]()
                                stream.synchronize()
                                with torch.cuda.graph(graph, stream=stream):
                                    output = calls[name]()
                            torch.cuda.current_stream().wait_stream(stream)
                            def replay(graph=graph, output=output):
                                graph.replay()
                                return output
                            calls[name] = replay
                            retained.append((graph, output, stream))
                    for repeat in range(args.repeats):
                        offset = repeat % len(names)
                        order = names[offset:] + names[:offset]
                        if repeat % 2:
                            order.reverse()
                        for name in order:
                            output = calls[name]().detach().cpu().clone()
                            if name not in first:
                                first[name] = output
                            last[name] = output
                            after = {key: digest(value) for key, value in inputs.items()}
                            changed = [key for key in hashes if hashes[key] != after[key]]
                            row = dict(mode=mode, repeat=repeat, variant=name, order=order,
                                       sha256=digest(output), changed_inputs=changed,
                                       input_sha256_after=after, versus_first=stats(output, first[name]),
                                       versus_baseline=stats(output, first['baseline']))
                            case['runs'].append(row)
                            if changed or not row['versus_first']['bitwise_equal'] or not row['versus_baseline']['bitwise_equal']:
                                report['failures'].append(dict(tokens=tokens, mask=mask_kind, mode=mode,
                                                               repeat=repeat, variant=name,
                                                               changed_inputs=changed,
                                                               repeat_bytes=row['versus_first']['byte_mismatch'],
                                                               baseline_bytes=row['versus_baseline']['byte_mismatch']))
                            save(args.report, report)
                            print(tokens, mask_kind, mode, repeat, name,
                                  'base_bytes', row['versus_baseline']['byte_mismatch'],
                                  'max_abs', row['versus_baseline']['max_abs'],
                                  'rmse', row['versus_baseline']['rmse'],
                                  'repeat_bytes', row['versus_first']['byte_mismatch'],
                                  'input_changes', changed, flush=True)
                    case['pairwise_by_mode'][mode] = {
                        name: dict(total=stats(last[name], last['baseline']),
                                   by_q_block=by_block(last[name], last['baseline'])) for name in names}
                    save(args.report, report)
                case['unique_outputs'] = {name: sorted({row['sha256'] for row in case['runs']
                                                        if row['variant'] == name}) for name in names}
                save(args.report, report)
        report['complete'] = True
        report['passed'] = not report['failures']
        save(args.report, report)
        if report['failures']:
            raise AssertionError(f"Diagnostic complete: {len(report['failures'])} strict-gate failures; see {args.report}")
        print('ALL BYTE/IMMUTABILITY/REPEAT GATES PASSED; oracle validation remains separate', flush=True)


if __name__ == '__main__':
    main()
