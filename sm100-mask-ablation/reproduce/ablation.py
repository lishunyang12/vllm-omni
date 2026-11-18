"""Frozen H3 input ablation; diagnostic variants never change production code."""
import argparse
import hashlib
import importlib
import json
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def isolated(root, destination, name, ranges=None):
    source = root / "techniques/sparse_backends/sol_attn"
    target = destination / name
    shutil.copytree(source, target)
    hashes = {}
    for path in target.rglob("*.py"):
        text = path.read_text()
        hashes[str(path.relative_to(target))] = hashlib.sha256(text.encode()).hexdigest()
        text = re.sub(r"\bfrom sol_attn(?=[. ])", f"from {name}", text)
        text = re.sub(r"\bimport sol_attn(?=[. ])", f"import {name}", text)
        if ranges is not None and path.relative_to(target).as_posix() == "common/selector.py":
            start = text.index("    protected = False\n", text.index("def sol_attn_route_is_exact"))
            end = text.index("    return ((column_mean", start)
            clauses = []
            for lo, hi in ranges:
                for axis in ("q_block", "kv_block"):
                    clauses.append(f"(({axis} >= Int32({lo})) and ({axis} < Int32({hi})))")
            expr = " or ".join(clauses) or "False"
            text = text[:start] + f"    protected = valid and ({expr})\n" + text[end:]
        path.write_text(text)
    sys.path.insert(0, str(destination))
    loaded = importlib.import_module(name)
    return loaded.sol_attn, hashes


def assert_bytes(a, b):
    torch.testing.assert_close(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8), atol=0, rtol=0)


def runs(mask):
    indices = mask.nonzero().flatten().tolist()
    result = []
    for idx in indices:
        if result and result[-1][1] == idx:
            result[-1][1] += 1
        else:
            result.append([idx, idx + 1])
    return result


def save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("/work"))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--heads", type=int, default=28)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=16)
    p.add_argument("--modes", default="eager,graph")
    a = p.parse_args()
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("This original-upstream ablation requires SM100")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(a.input, map_location="cpu", weights_only=True)
    for key, value in payload["tensor_sha256"].items():
        assert digest(payload[key]) == value, key
    q, k, v = [payload[n][:, :, :a.heads].contiguous().cuda() for n in ("q", "k", "v")]
    mask = payload["exact_block_mask"].cuda()
    zeros = torch.zeros_like(mask)
    options = payload["options"]
    protected_ranges = runs(mask.cpu())
    report = dict(pins=json.loads((a.work / "pins.json").read_text()), input=a.input.name,
                  input_sha256=payload["tensor_sha256"], shape=list(q.shape), options=options,
                  protected_ranges=protected_ranges, protected_blocks=int(mask.sum()),
                  denoise_step=payload["denoise_step"], layer=payload["layer"],
                  trajectory=payload["trajectory"], warmup=a.warmup, repeats=a.repeats,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  caveat="none/zero variants remove H3 protection; not valid quality-preserving replacements",
                  modes={})
    with tempfile.TemporaryDirectory(prefix="sol-mask-ablation-") as tmp:
        destination = Path(tmp)
        upstream, up_hash = isolated(a.work / "source/upstream", destination, "_sol_upstream")
        extension, ex_hash = isolated(a.work / "source/extension", destination, "_sol_extension")
        static, static_hash = isolated(a.work / "source/extension", destination, "_sol_static", protected_ranges)
        candidate, ca_hash = isolated(a.work / "source/candidate", destination, "_sol_candidate")
        report["source_sha256"] = dict(upstream=up_hash, extension=ex_hash, static_base=static_hash, candidate=ca_hash)

        def dense():
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=options["scale"]
                ).transpose(1, 2)

        functions = {
            "dense": dense,
            "upstream_none": lambda: upstream(q, k, v, **options),
            "extension_none": lambda: extension(q, k, v, **options),
            "extension_zero": lambda: extension(q, k, v, **options, exact_block_mask=zeros),
            "extension_mask": lambda: extension(q, k, v, **options, exact_block_mask=mask),
            "extension_static": lambda: static(q, k, v, **options),
            "optimized_mask": lambda: candidate(q, k, v, **options, exact_block_mask=mask),
        }
        references = {"unprotected": functions["upstream_none"](), "protected": functions["extension_mask"]()}
        groups = dict(upstream_none="unprotected", extension_none="unprotected", extension_zero="unprotected",
                      extension_mask="protected", extension_static="protected", optimized_mask="protected")
        for name, fn in functions.items():
            result = fn()
            assert bool(torch.isfinite(result).all()), name
            if name in groups:
                assert_bytes(result, references[groups[name]])
            print("VALIDATED", name, flush=True)
        report["bitwise_equivalent_groups"] = groups
        save(a.output, report)
        for mode in a.modes.split(","):
            assert mode in ("eager", "graph")
            calls, retained = {}, []
            for name, fn in functions.items():
                for _ in range(a.warmup):
                    fn()
                torch.cuda.synchronize()
                if mode == "graph":
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = fn()
                    graph.replay()
                    torch.cuda.synchronize()
                    if name in groups:
                        assert_bytes(output, references[groups[name]])
                    retained.append((graph, output))
                    calls[name] = graph.replay
                else:
                    calls[name] = fn
            samples = []
            names = list(calls)
            for iteration in range(a.repeats):
                offset = iteration % len(names)
                order = names[offset:] + names[:offset]
                if iteration % 2:
                    order.reverse()
                for name in order:
                    torch.cuda.synchronize()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    unix = time.time()
                    wall = time.perf_counter()
                    begin.record()
                    calls[name]()
                    end.record()
                    torch.cuda.synchronize()
                    samples.append(dict(round=iteration, variant=name, start_unix_s=unix,
                                        wall_s=time.perf_counter()-wall, event_ms=begin.elapsed_time(end)))
            metrics = {}
            for name in names:
                values = [s["event_ms"] for s in samples if s["variant"] == name]
                metrics[name] = dict(median_ms=statistics.median(values), cv_percent=100*statistics.stdev(values)/statistics.mean(values))
            report["modes"][mode] = dict(samples=samples, metrics=metrics)
            save(a.output, report)
            print(mode, json.dumps(metrics), flush=True)
        for name, group in groups.items():
            assert_bytes(functions[name](), references[group])
        report["complete"] = True
        save(a.output, report)


if __name__ == "__main__":
    main()
