"""Diagnose baseline/A/C discrepancies without weakening numerical gates.

Run only inside an already allocated CUDA environment. No GPU allocation is
requested. Reports all differences first, then exits nonzero if any baseline
byte, oracle, repeatability, or input-immutability gate failed.
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from validate_design import isolated, load_test, save


def digest(tensor):
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def inputs(tokens=513, heads=2, batch=1, seed=71):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(torch.randn(batch, tokens, heads, 128, device="cuda",
                             dtype=torch.bfloat16, generator=generator) for _ in range(3))


def make_mask(tokens, kind):
    if kind == "none":
        return None
    mask = torch.zeros((tokens + 63) // 64, device="cuda", dtype=torch.bool)
    if kind == "all":
        mask.fill_(True)
    elif kind == "disjoint":
        mask[0] = mask[-1] = True
        if len(mask) > 5:
            mask[4] = True
    else:
        assert kind == "zero"
    return mask


def tensor_stats(actual, expected):
    actual = actual.detach().cpu().contiguous()
    expected = expected.detach().cpu().contiguous()
    a, b = actual.double(), expected.double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result = dict(elements=a.numel(), finite=finite, max_abs=None, rmse=None, relative_l2=None,
                  nonfinite_actual=int((~torch.isfinite(a)).sum()),
                  nonfinite_expected=int((~torch.isfinite(b)).sum()))
    if finite:
        diff = a - b
        result.update(max_abs=float(diff.abs().max()),
                      rmse=float(diff.square().mean().sqrt()),
                      relative_l2=float(diff.norm() / b.norm().clamp_min(1e-300)),
                      unequal_float_elements=int((actual != expected).sum()))
    if actual.dtype == expected.dtype:
        mismatch = actual.view(torch.uint8) != expected.view(torch.uint8)
        result.update(byte_count=mismatch.numel(), byte_mismatch=int(mismatch.sum()),
                      bitwise_equal=not bool(mismatch.any()))
        if bool(mismatch.any()):
            idx = mismatch.nonzero()[0].tolist()
            element_idx = [*idx[:-1], idx[-1] // actual.element_size()]
            result["first_byte_mismatch"] = dict(
                byte_index=idx, element_index=element_idx,
                actual=float(actual[tuple(element_idx)]),
                expected=float(expected[tuple(element_idx)]))
    try:
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.02, rtol=0.02)
    except AssertionError as error:
        result["oracle_tolerance_pass"] = False
        result["tolerance_failure"] = str(error)
    else:
        result["oracle_tolerance_pass"] = True
    return result


def compare_by_block(actual, expected, mask):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    result = dict(total=tensor_stats(actual, expected), by_q_block=[])
    protected = [False] * ((actual.shape[1] + 63) // 64) if mask is None else mask.cpu().tolist()
    for qb, is_protected in enumerate(protected):
        lo, hi = qb * 64, min((qb + 1) * 64, actual.shape[1])
        row = dict(q_block=qb, protected=is_protected, token_start=lo, token_end=hi,
                   **tensor_stats(actual[:, lo:hi], expected[:, lo:hi]))
        row["by_head"] = [dict(head=head, **tensor_stats(actual[:, lo:hi, head:head + 1],
                                                        expected[:, lo:hi, head:head + 1]))
                           for head in range(actual.shape[2])]
        result["by_q_block"].append(row)
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--mask", choices=("disjoint", "none", "zero", "all"), default="disjoint")
    parser.add_argument("--tokens", type=int, default=513)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--threshold", choices=("diag", "exact"), default="diag")
    parser.add_argument("--prime-boundaries", action="store_true",
                        help="first run the preceding nine cases from the 73-case runner")
    args = parser.parse_args()
    assert args.repeats >= 2
    assert torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    modes = args.modes.split(",")
    assert set(modes) <= {"eager", "graph"}
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, arguments={key: str(value) if isinstance(value, Path) else value
                                                     for key, value in vars(args).items()},
                  complete=False, passed=False, prime_cases=[], runs=[], failures=[])
    roots = dict(baseline=args.baseline, a=args.a, c=args.c)
    with tempfile.TemporaryDirectory(prefix="sol-protected-q-diagnostic-") as tmp:
        modules, source_hashes = {}, {}
        for name, root in roots.items():
            modules[name], source_hashes[name] = isolated(root, Path(tmp), "_diag_" + name)
        report["source_sha256"] = source_hashes
        oracle_module = load_test(args.baseline / "techniques/sparse_backends/tests/test_exact_block_mask.py",
                                  "_diag_oracle", "_diag_baseline")
        save(args.report, report)
        names = list(modules)
        if args.prime_boundaries:
            for tokens in (63, 64, 65, 127, 128, 129, 255, 256, 257):
                qkv = inputs(tokens, batch=2 if tokens == 257 else 1,
                             heads=3 if tokens == 129 else 2, seed=71)
                mask = make_mask(tokens, "disjoint")
                results = {name: module.sol_attn(*qkv, tau=1.5, thresh_type="diag", exact_block_mask=mask).cpu()
                           for name, module in modules.items()}
                row = dict(tokens=tokens, comparisons={name: tensor_stats(value, results["baseline"])
                                                        for name, value in results.items()})
                report["prime_cases"].append(row)
                for name, stats in row["comparisons"].items():
                    if not stats["bitwise_equal"]:
                        report["failures"].append(f"prime/{tokens}/{name}/baseline_bytes")
                save(args.report, report)

        qkv = inputs(args.tokens, heads=args.heads, seed=args.seed)
        mask = make_mask(args.tokens, args.mask)
        immutable = dict(zip(("q", "k", "v"), qkv))
        if mask is not None:
            immutable["mask"] = mask
        initial_hashes = {name: digest(value) for name, value in immutable.items()}
        report["input_sha256"] = initial_hashes
        report["mask_values"] = None if mask is None else mask.cpu().tolist()
        options = dict(tau=args.tau, thresh_type=args.threshold, exact_block_mask=mask)
        oracle_mask = mask if mask is not None else make_mask(args.tokens, "zero")
        oracle = oracle_module.hybrid_oracle(*qkv, oracle_mask, args.tau, args.threshold).cpu()
        report["oracle_sha256"] = digest(oracle)
        report["oracle_input_unchanged"] = {name: digest(value) == initial_hashes[name]
                                           for name, value in immutable.items()}
        if not all(report["oracle_input_unchanged"].values()):
            report["failures"].append("oracle/input_mutation")

        first_outputs, latest_outputs = {}, {}
        graph_retained = []
        for mode in modes:
            calls = {name: (lambda name=name: modules[name].sol_attn(*qkv, **options)) for name in names}
            if mode == "graph":
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
                    graph_retained.append((graph, output, stream))
            for repeat in range(args.repeats):
                offset = repeat % len(names)
                order = names[offset:] + names[:offset]
                if repeat % 2:
                    order.reverse()
                for name in order:
                    result = calls[name]().detach().cpu().clone()
                    after = {key: digest(value) for key, value in immutable.items()}
                    changed = [key for key, value in after.items() if value != initial_hashes[key]]
                    if name not in first_outputs:
                        first_outputs[name] = result
                    reference = first_outputs[name]
                    stable = tensor_stats(result, reference)
                    versus_oracle = compare_by_block(result, oracle, mask)
                    row = dict(mode=mode, repeat=repeat, variant=name, order=order,
                               output_sha256=digest(result), input_sha256_after=after,
                               changed_inputs=changed, versus_first_run=stable,
                               versus_oracle=versus_oracle)
                    latest_outputs[name] = result
                    if "baseline" in first_outputs:
                        row["versus_baseline_first"] = compare_by_block(result, first_outputs["baseline"], mask)
                        if not row["versus_baseline_first"]["total"]["bitwise_equal"]:
                            report["failures"].append(f"{mode}/{repeat}/{name}/baseline_bytes")
                    if changed:
                        report["failures"].append(f"{mode}/{repeat}/{name}/input_mutation:{changed}")
                    if not stable["bitwise_equal"]:
                        report["failures"].append(f"{mode}/{repeat}/{name}/repeatability")
                    if not versus_oracle["total"]["oracle_tolerance_pass"]:
                        report["failures"].append(f"{mode}/{repeat}/{name}/oracle")
                    report["runs"].append(row)
                    save(args.report, report)
                    comparison = row.get("versus_baseline_first", {}).get("total", {})
                    print(mode, repeat, name, "bytes", comparison.get("byte_mismatch"),
                          "max_abs", comparison.get("max_abs"), "repeat_bytes", stable["byte_mismatch"],
                          "oracle_pass", versus_oracle["total"]["oracle_tolerance_pass"],
                          "input_changes", changed, flush=True)
            report.setdefault("pairwise_by_mode", {})[mode] = {
                f"{left}_vs_{right}": compare_by_block(latest_outputs[left], latest_outputs[right], mask)
                for left, right in (("a", "baseline"), ("c", "baseline"), ("c", "a"))}
            save(args.report, report)
        report["unique_output_sha256"] = {
            name: sorted({row["output_sha256"] for row in report["runs"] if row["variant"] == name}) for name in names}
        report["complete"] = True
        report["passed"] = not report["failures"]
        save(args.report, report)
        for mode, comparisons in report["pairwise_by_mode"].items():
            for name, comparison in comparisons.items():
                print("BLOCK_SUMMARY", mode, name, flush=True)
                for row in comparison["by_q_block"]:
                    print(row["q_block"], "protected", row["protected"], "bytes", row["byte_mismatch"],
                          "max_abs", row["max_abs"], "rmse", row["rmse"], flush=True)
        if report["failures"]:
            raise AssertionError(f"Diagnostic complete with {len(report['failures'])} failed strict gates; see {args.report}")
        print("ALL STRICT GATES PASSED", flush=True)


if __name__ == "__main__":
    main()
