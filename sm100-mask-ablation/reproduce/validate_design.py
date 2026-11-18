"""Run isolated SM100 correctness checks; no GPU allocation and no timing.

Package renaming follows scripts/ablation.py. The existing independent hybrid
oracle and regression cases are loaded from the pinned baseline source tree.
Each eager SOL call also requires candidate/baseline byte equivalence. During
CUDA Graph capture the wrapper records only the candidate, as required by the
existing graph test; its post-replay eager comparison checks both versions.
"""
import argparse
import hashlib
import importlib
import json
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

import torch


def isolated(root, destination, name):
    target = destination / name
    shutil.copytree(root / "techniques/sparse_backends/sol_attn", target)
    hashes = {}
    for path in target.rglob("*.py"):
        source = path.read_text()
        hashes[str(path.relative_to(target))] = hashlib.sha256(source.encode()).hexdigest()
        source = re.sub(r"\bfrom sol_attn(?=[. ])", f"from {name}", source)
        source = re.sub(r"\bimport sol_attn(?=[. ])", f"import {name}", source)
        path.write_text(source)
    sys.path.insert(0, str(destination))
    return importlib.import_module(name), hashes


def assert_bytes(actual, expected):
    torch.testing.assert_close(actual.contiguous().view(torch.uint8),
                               expected.contiguous().view(torch.uint8), atol=0, rtol=0)


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def load_test(path, name, package, oracle_name=None):
    source = path.read_text().replace("from sol_attn", "from " + package)
    if oracle_name:
        source = source.replace("from test_exact_block_mask", "from " + oracle_name)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tests-root", type=Path)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--sections", default="small,long,mask,groups,graph,frozen")
    args = parser.parse_args()
    sections = set(args.sections.split(","))
    assert sections <= {"small", "long", "mask", "groups", "graph", "frozen"}
    assert torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    report = dict(baseline=str(args.baseline), candidate=str(args.candidate),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), sections=sorted(sections), cases=[], complete=False)
    with tempfile.TemporaryDirectory(prefix="sol-design-validation-") as tmp:
        baseline, bh = isolated(args.baseline, Path(tmp), "_design_baseline")
        candidate, ch = isolated(args.candidate, Path(tmp), "_design_candidate")
        report["source_sha256"] = dict(baseline=bh, candidate=ch)
        tests_root = args.tests_root or args.baseline / "techniques/sparse_backends/tests"
        oracle = load_test(tests_root / "test_exact_block_mask.py",
                           "_design_oracle", "_design_baseline")
        tests = load_test(tests_root / "test_sm100_rescale_mask.py",
                          "_design_regression", "_design_candidate", "_design_oracle")
        save(args.report, report)

        def checked(*qkv, **options):
            if torch.cuda.is_current_stream_capturing():
                return candidate.sol_attn(*qkv, **options)
            expected = baseline.sol_attn(*qkv, **options)
            actual = candidate.sol_attn(*qkv, **options)
            assert bool(torch.isfinite(actual).all())
            assert_bytes(actual, expected)
            return actual

        tests.sol_attn = checked

        def passed(name, **extra):
            report["cases"].append(dict(name=name, passed=True, **extra))
            save(args.report, report)
            print("PASSED", name, flush=True)

        if "small" in sections:
            for threshold in ("diag", "exact"):
                for tokens in (63, 64, 65, 127, 128, 129, 255, 256, 257, 513):
                    tests.test_tile_boundaries_match_independent_oracle(tokens, threshold)
                    passed(f"tile-oracle/{threshold}/{tokens}", bitwise=True)
                for tokens, tau in ((63, 1e6), (257, 1e6), (513, 1e6), (513, 1.5)):
                    oracle.test_disjoint_exact_rows_and_columns_match_hybrid_oracle(
                        checked, tokens, tau, threshold)
                    passed(f"hybrid-oracle/{threshold}/{tokens}/tau={tau}", bitwise=True)

        if "long" in sections:
            for tokens in (4095, 4096, 4097):
                for kind in ("random", "mixed_maxima", "zero_q", "zero_v", "tiny_v", "subnormal_v"):
                    tests.test_long_exact_stream_matches_dense(tokens, kind)
                    passed(f"exact-stream/{tokens}/{kind}", bitwise=True)

        if "mask" in sections:
            tests.test_absent_empty_and_updated_mask_preserve_sink_contract()
            passed("absent-zero-disjoint-sink", bitwise=True)
            oracle.test_empty_mask_preserves_legacy_output_and_mutated_mask_uses_current_values(checked)
            passed("unaligned-zero-all-mask", bitwise=True)

        if "groups" in sections:
            # Multi-group protected Q paths are absent from the existing small
            # oracle cases. Check index lifetimes across 129/257/513 KV blocks.
            for tokens in (8193, 16385, 32769):
                qkv = tests.inputs(tokens, heads=1)
                for kind in ("none", "zero", "all", "disjoint", "alternating"):
                    mask = None
                    if kind != "none":
                        mask = torch.zeros((tokens + 63) // 64, device="cuda", dtype=torch.bool)
                        if kind == "all":
                            mask.fill_(True)
                        elif kind == "disjoint":
                            mask[0] = mask[4] = mask[-1] = True
                        elif kind == "alternating":
                            mask[::2] = True
                    checked(*qkv, tau=1.0, exact_block_mask=mask)
                    passed(f"group-boundaries/{tokens}/{kind}", bitwise=True)

        if "graph" in sections:
            for threshold in ("diag", "exact"):
                tests.test_graph_reads_updated_inputs_and_masks_without_recompile(threshold)
                passed(f"graph-existing/{threshold}", bitwise=True, no_recompile=True)
                q, k, v = tests.inputs(513, batch=2, heads=3)
                mask = torch.zeros(9, device="cuda", dtype=torch.bool)
                options = dict(tau=1.5, thresh_type=threshold, exact_block_mask=mask)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    candidate.sol_attn(q, k, v, **options)
                    stream.synchronize()
                    compiled = set(tests.interface._compiled)
                    with torch.cuda.graph(graph, stream=stream):
                        captured = candidate.sol_attn(q, k, v, **options)
                torch.cuda.current_stream().wait_stream(stream)
                for update in ("zero", "all", "disjoint", "zero_again"):
                    mask.zero_()
                    if update == "all":
                        mask.fill_(True)
                    elif update == "disjoint":
                        mask[2] = mask[5] = mask[-1] = True
                    q.copy_(q.flip(1))
                    k.mul_(0.75)
                    v.neg_()
                    graph.replay()
                    actual = checked(q, k, v, **options)
                    assert_bytes(captured, actual)
                    expected = oracle.hybrid_oracle(q, k, v, mask, 1.5, threshold)
                    torch.testing.assert_close(captured.float(), expected, atol=0.02, rtol=0.02)
                    assert set(tests.interface._compiled) == compiled
                    passed(f"graph-flips/{threshold}/{update}", bitwise=True, no_recompile=True)

        if "frozen" in sections:
            for path in args.input:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                for key, expected_hash in payload["tensor_sha256"].items():
                    actual_hash = hashlib.sha256(payload[key].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    assert actual_hash == expected_hash, key
                qkv = tuple(payload[key][:, :, :args.heads].contiguous().cuda() for key in ("q", "k", "v"))
                checked(*qkv, **payload["options"], exact_block_mask=payload["exact_block_mask"].cuda())
                passed(f"frozen/{path.name}", bitwise=True, tensor_sha256=payload["tensor_sha256"])
        torch.cuda.synchronize()
        report["complete"] = True
        save(args.report, report)
        print("ALL PASSED", len(report["cases"]), flush=True)


if __name__ == "__main__":
    main()
