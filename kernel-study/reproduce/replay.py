"""Same-input Dense/native SOL/experimental warp-MMA replay. No production dispatch changes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from sol_attn import interface


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def compare(actual, expected):
    a, b = actual.double(), expected.double()
    d = a - b
    return dict(finite=bool(torch.isfinite(a).all()), bitwise_equal=bool(torch.equal(actual, expected)),
                max_abs=float(d.abs().max()), rmse=float(d.square().mean().sqrt()),
                relative_l2=float(d.norm() / b.norm().clamp_min(1e-30)))


def functions(q, k, v, options, mask):
    options = dict(options, exact_block_mask=mask)

    def dense():
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                scale=options['scale']).transpose(1, 2)

    def native():
        return interface.sol_attn(q, k, v, **options)

    def warp():
        # Compile the SM120 recipe for the actual current CUDA device. The
        # synthetic arch tuple selects only the experimental recipe/cache key.
        interface._validate_inputs(q, k, v, options['thresh_type'], options['sink_tokens'],
                                   options['sink_start'], mask)
        return interface._sol_attn_cute(q, k, v, arch=(12, 0), **options)

    return dict(dense=dense, native=native, warp=warp)


def compile_trace(q, k, v, options, mask, variant, generated):
    import cutlass.cute as cute
    from sol_attn.common import to_cute_tensor
    from sol_attn.preprocess import prepare

    if variant == 'native':
        import sol_attn.sm100.mainloop as module
        text = Path(module.__file__).read_text()
        needle = '''                        word_mask = Int32(
                            cute.arch.vote_ballot_sync(exact_pred)
                        )'''
        assert text.count(needle) == 1
        text = text.replace(needle, needle + '''
                        if lane == Int32(0):
                            mLSE_bth[batch_idx, q_block_idx, head_idx,
                                     route_start // Int32(128), word] = word_mask''')
        begin = text.index('        if (lane & Int32(2)) == Int32(0) and owner_row < q_len:')
        end = text.index('\n    cute.arch.barrier()', begin)
        text = text[:begin] + text[end:]
        generated.mkdir(parents=True, exist_ok=True)
        path = generated / 'sm100_route_trace.py'
        path.write_text(text)
        name = 'sol_attn.sm100._study_route_trace'
        spec = importlib.util.spec_from_file_location(name, path)
        diagnostic = importlib.util.module_from_spec(spec)
        sys.modules[name] = diagnostic
        spec.loader.exec_module(diagnostic)
        operator, width = diagnostic.forward, 128
    else:
        from sol_attn.sm120 import make_kernel
        operator, width = make_kernel(debug_route_trace=True), 64
    b, t, h, _ = q.shape
    blocks = (t + 63) // 64
    trace = torch.zeros((b, blocks, h, (blocks + width - 1) // width, width // 32),
                        device=q.device, dtype=torch.int32)
    kc, vc, threshold = prepare(q, k, v, scale=options['scale'], tau=options['tau'],
                               thresh_type=options['thresh_type'])
    output = torch.empty_like(v)
    args = interface._to_cute_tensors([q, k, v, output, kc, vc, threshold, trace])
    start, end = interface._sink_block_range(t, options['sink_start'], options['sink_tokens'])
    mask_arg = None if mask is None else to_cute_tensor(mask.view(torch.uint8), assumed_align=1)
    stream = interface._stream(q.device)
    compiled = cute.compile(operator, *args, options['scale'], start, end, mask_arg,
                            stream=stream, options='--enable-tvm-ffi')
    compiled(*args, options['scale'], start, end, mask_arg, stream=stream)
    torch.cuda.synchronize()
    return output, trace


def summarize_trace(trace, mask, blocks):
    words = trace.cpu().numpy().view(np.uint32)
    bits = np.unpackbits(words.view(np.uint8), axis=-1, bitorder='little')
    bits = bits.reshape(*words.shape[:3], -1)[..., :blocks]
    counts = bits.sum(-1)
    protected = np.zeros(blocks, dtype=bool) if mask is None else mask.cpu().numpy()
    # Shape [batch, Q block, head, KV block].
    forced = protected[:, None] | protected[None, :]
    missing = int((~bits.astype(bool) & forced[None, :, None, :]).sum())
    if missing:
        raise AssertionError(f'Protection contract broken: {missing} selected pairs missing')
    padded = np.pad(bits, ((0, 0), (0, 0), (0, 0), (0, (-blocks) % 256)))
    group_counts = padded.reshape(*bits.shape[:3], -1, 256).sum(-1)
    odd_groups = int((group_counts % 2).sum())
    summary = dict(exact_block_fraction=float(bits.mean()), protected_blocks=int(protected.sum()),
                   protected_pair_lower_bound=float(forced.mean()), protected_pairs_missing=missing,
                   per_head_fraction=bits.mean(axis=(0, 1, 3)).tolist(),
                   per_query_exact_count_percentiles=np.percentile(counts, [0, 10, 50, 90, 100]).tolist(),
                   sm100_padded_peer_blocks=odd_groups,
                   sm100_padding_fraction_of_selected=float(odd_groups / max(1, counts.sum())))
    if (~protected).any():
        summary['unprotected_query_exact_fraction'] = float(bits[:, ~protected].mean())
    return summary, bits


@torch.inference_mode()
def validate(output):
    test_path = Path(interface.__file__).parents[1] / 'tests/test_exact_block_mask.py'
    spec = importlib.util.spec_from_file_location('study_oracle', test_path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    rows = []
    for tokens, tau in [(63, 1e6), (257, 1e6), (513, 1.5)]:
        for thresh_type in ('diag', 'exact'):
            torch.manual_seed(71)
            q, k, v = [torch.randn(1, tokens, 2, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
            mask = torch.zeros((tokens+63)//64, device='cuda', dtype=torch.bool)
            mask[0] = mask[-1] = True
            options = dict(scale=128**-0.5, tau=tau, thresh_type=thresh_type,
                           kv_splits=1, sink_tokens=0, sink_start=0)
            expected = oracle.hybrid_oracle(q, k, v, mask, tau, thresh_type)
            for name, fn in functions(q, k, v, options, mask).items():
                if name == 'dense':
                    continue
                result = fn()
                torch.testing.assert_close(result.float(), expected, atol=.02, rtol=.02)
                rows.append(dict(tokens=tokens, tau=tau, threshold=thresh_type, variant=name,
                                 **compare(result, expected)))
                save(output, dict(passed=len(rows), cases=rows))
    print('COMPATIBILITY_VALIDATION', len(rows), 'passed', flush=True)
    trace_checks = {}
    traces = {}
    for name in ('native', 'warp'):
        actual, trace = compile_trace(q, k, v, options, mask, name, output.parent / 'generated')
        expected = functions(q, k, v, options, mask)[name]()
        torch.testing.assert_close(actual, expected, atol=.002, rtol=.002)
        stats, bits = summarize_trace(trace, mask, (q.shape[1]+63)//64)
        trace_checks[name] = dict(output=compare(actual, expected), routing=stats)
        traces[name] = bits
    save(output, dict(passed=len(rows), cases=rows, trace_checks=trace_checks,
                      trace_different_decision_fraction=float(np.not_equal(traces['native'], traces['warp']).mean())))
    print('TRACE_VALIDATION passed', flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=15)
    p.add_argument('--heads', type=int, default=0, help='0 keeps every captured head')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--profile-only', choices=['dense', 'native', 'warp'])
    a = p.parse_args()
    if a.validate:
        validate(a.output)
        return
    if a.input is None:
        p.error('--input required unless --validate')
    data = torch.load(a.input, map_location='cpu', weights_only=True)
    verified = {}
    for key in ('q', 'k', 'v', 'exact_block_mask'):
        t = data[key]
        if t is not None:
            sha = hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            assert sha == data['tensor_sha256'][key], f'Capture hash mismatch: {key}'
            verified[key] = sha
    q, k, v = [data[key][:, :, :a.heads or None].contiguous().cuda() for key in ('q', 'k', 'v')]
    mask = data['exact_block_mask']
    mask = None if mask is None else mask.cuda()
    funcs = functions(q, k, v, data['options'], mask)
    names = [a.profile_only] if a.profile_only else list(funcs)
    report = dict(input_file=a.input.name, capture_sha256=verified, shape=list(q.shape),
                  captured_shape=list(data['q'].shape), layer=data['layer'],
                  trajectory=data['trajectory'], denoise_step=data['denoise_step'], options=data['options'],
                  gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  interface_sha256=hashlib.sha256(Path(interface.__file__).read_bytes()).hexdigest(),
                  warmup=a.warmup, repeats=a.repeats, samples=[],
                  measurement='Single attention operator, same frozen Q/K/V, preprocessing included; no model latency claim')
    if a.profile_only:
        fn = funcs[a.profile_only]
        for _ in range(a.warmup):
            fn()
        if a.profile_only != 'dense':
            # Hardware-counter runs isolate the main kernel. Unprofiled latency
            # above/below continues to include the real preprocessing calls.
            from sol_attn import preprocess
            cached = preprocess.prepare(q, k, v, scale=data['options']['scale'],
                                        tau=data['options']['tau'], thresh_type=data['options']['thresh_type'])
            preprocess.prepare = lambda *args, **kwargs: cached
        report['profile_boundary'] = 'Main attention kernel; SOL summaries precomputed outside NVTX range'
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push('H3_SOL_PROFILE')
        fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        save(a.output, report)
        return
    for i in range(a.warmup + a.repeats):
        # Rotate and reverse orders, so every variant occupies each position.
        offset = i % len(names)
        order = names[offset:] + names[:offset]
        if (i // len(names)) % 2:
            order = order[::-1]
        for name in order:
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timestamp, wall = time.time(), time.perf_counter()
            begin.record()
            result = funcs[name]()
            end.record()
            torch.cuda.synchronize()
            elapsed = 1000 * (time.perf_counter() - wall)
            if i >= a.warmup:
                report['samples'].append(dict(index=i-a.warmup, variant=name, wall_ms=elapsed,
                                              event_ms=begin.elapsed_time(end), start_unix_s=timestamp))
        print('REPLAY_ITERATION', i, flush=True)
    reference = funcs['dense']()
    stock = funcs['native']()
    alternate = funcs['warp']()
    report['quality'] = dict(native_vs_dense=compare(stock, reference), warp_vs_dense=compare(alternate, reference),
                             warp_vs_native=compare(alternate, stock))
    if not all(item['finite'] for item in report['quality'].values()):
        raise RuntimeError('Nonfinite attention output')
    report['runs'] = {}
    for name in names:
        values = [s['event_ms'] for s in report['samples'] if s['variant'] == name]
        report['runs'][name] = dict(median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values),
                                   cv_percent=100*statistics.stdev(values)/statistics.mean(values))
    save(a.output, report)
    if a.trace:
        traces = {}
        report['routing'] = {}
        for variant, expected in [('native', stock), ('warp', alternate)]:
            out, trace = compile_trace(q, k, v, data['options'], mask, variant, a.output.parent / 'generated')
            delta = compare(out, expected)
            torch.testing.assert_close(out, expected, atol=.002, rtol=.002)
            stats, bits = summarize_trace(trace, mask, (q.shape[1]+63)//64)
            stats['diagnostic_output_vs_uninstrumented'] = delta
            report['routing'][variant] = stats
            traces[variant] = bits
            np.savez_compressed(a.output.with_name(a.output.stem + f'-{variant}-routes.npz'), words=trace.cpu().numpy())
            save(a.output, report)
        report['routing']['different_decision_fraction'] = float(np.not_equal(traces['native'], traces['warp']).mean())
        save(a.output, report)
    print(json.dumps({k: report[k] for k in ('runs', 'quality', 'routing') if k in report}), flush=True)


if __name__ == '__main__':
    main()
