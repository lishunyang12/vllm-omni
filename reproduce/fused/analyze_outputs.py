"""Compare saved complete-forward tensors outside all benchmark timings."""
import argparse
import json
from pathlib import Path

import torch


def metrics(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('Output shape or dtype mismatch')
    a, b = actual.double().reshape(-1), expected.double().reshape(-1)
    diff = a - b
    bn = b.norm()
    an = a.norm()
    return dict(shape=list(actual.shape), dtype=str(actual.dtype),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                bitwise_equal=bool(torch.equal(actual.contiguous().view(torch.uint8),
                                              expected.contiguous().view(torch.uint8))),
                max_abs=diff.abs().max().item(), mean_abs=diff.abs().mean().item(),
                rmse=diff.square().mean().sqrt().item(),
                relative_l2=(diff.norm() / bn).item(),
                cosine_similarity=(torch.dot(a, b) / (an * bn)).item(),
                different_fraction=(actual != expected).float().mean().item(),
                reference_abs_max=b.abs().max().item(), reference_rms=b.square().mean().sqrt().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    variants = ['reference', 'dense']
    if (args.root / 'ref2va-fused' / 'summary.json').exists():
        variants.append('fused')
    if (args.root / 'ref2va-fused-repeat' / 'summary.json').exists():
        variants.append('fused-repeat')
    for variant in ['precise', 'precise-repeat', 'final', 'final-repeat', 'grouped', 'grouped-repeat']:
        if (args.root / f'ref2va-{variant}' / 'summary.json').exists():
            variants.append(variant)
    result = dict(scope='First conditional full 50-layer DiT forward; no decoded video/image/audio metrics',
                  timings={}, inputs_match={}, repeatability={}, comparisons={})
    for variant in variants:
        directory = args.root / f'ref2va-{variant}'
        result['timings'][variant] = json.loads((directory / 'summary.json').read_text())
    for rank in range(2):
        reports = {v: json.loads((args.root / f'ref2va-{v}' / f'rank-{rank}.json').read_text()) for v in variants}
        result['inputs_match'][str(rank)] = all(r['input'] == reports['reference']['input'] for r in reports.values())
        if not result['inputs_match'][str(rank)]:
            raise ValueError(f'Forward input mismatch on rank {rank}')
        first = {}
        for variant in variants:
            directory = args.root / f'ref2va-{variant}'
            first[variant] = torch.load(directory / f'output-first-rank-{rank}.pt', map_location='cpu', weights_only=True)
            last = torch.load(directory / f'output-last-rank-{rank}.pt', map_location='cpu', weights_only=True)
            assert len(first[variant]) == len(last)
            result['repeatability'][f'{variant}/rank{rank}'] = [metrics(a, b) for a, b in zip(last, first[variant])]
        pairs = [('reference', 'dense')]
        if 'fused' in first:
            pairs.extend([('fused', 'reference'), ('fused', 'dense')])
        if 'fused-repeat' in first:
            pairs.append(('fused-repeat', 'fused'))
        for variant in ['precise', 'precise-repeat', 'final', 'final-repeat', 'grouped', 'grouped-repeat']:
            if variant in first:
                pairs.extend([(variant, 'reference'), (variant, 'dense')])
        for actual, expected in pairs:
            assert len(first[actual]) == len(first[expected])
            result['comparisons'][f'{actual}_vs_{expected}/rank{rank}'] = [
                metrics(a, b) for a, b in zip(first[actual], first[expected])]
    output = args.root.parent / 'output-comparison.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
