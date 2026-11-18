"""Recompute paired forward statistics from rank reports (standard library only)."""

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(root):
    protocol = json.loads((root / "protocol.json").read_text())
    count = protocol["engine"]["num_gpus"]
    ranks = [json.loads((root / f"rank-{i}.json").read_text()) for i in range(count)]
    for i, rank in enumerate(ranks):
        assert rank["rank"] == i and rank["world_size"] == count
        assert rank["warmup"] == 3 and rank["repeats"] == 9
        assert rank["finite"] and rank["intentional_abort"]
        assert rank["denoise_step"] == 0
        assert rank["parallel_config"]["tensor_parallel_size"] == 1
        assert rank["parallel_config"]["ulysses_degree"] == count
        assert sum(v["impl"] == "SolAttnImpl" for v in rank["backends"].values()) == 50
        assert rank["input"] == ranks[0]["input"]
    runs = {}
    for variant in ("dense", "sol"):
        samples = []
        for rank in ranks:
            selected = [x for x in rank["samples"] if x["variant"] == variant]
            assert [x["index"] for x in selected] == list(range(9))
            assert all(math.isfinite(x["wall_s"]) and x["wall_s"] > 0 for x in selected)
            samples.append(selected)
        times = [1000 * max(r[i]["wall_s"] for r in samples) for i in range(9)]
        runs[variant] = {
            "samples_ms": times,
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "cv_percent": 100 * statistics.stdev(times) / statistics.mean(times),
        }
    dense, sol = runs["dense"], runs["sol"]
    paired = [d / s for d, s in zip(dense["samples_ms"], sol["samples_ms"])]
    return {
        "path": str(root), "gpu": ranks[0]["gpu"], "gpu_count": count,
        "capability": ranks[0]["capability"], "sol_backend": ranks[0]["sol_backend"],
        "warmup_pairs": 3, "measured_pairs": 9, "runs": runs,
        "ratio_of_medians": dense["median_ms"] / sol["median_ms"],
        "latency_reduction_percent": 100 * (1 - sol["median_ms"] / dense["median_ms"]),
        "paired_speedups": paired, "all_pairs_faster": all(x > 1 for x in paired),
        "peak_allocated_GiB": max(r["peak_allocated_bytes"] for r in ranks) / 1024**3,
        "request_sha256": protocol["request_sha256"],
        "omni_source": protocol["runtime_commit"], "sol_source": protocol["upstream_commit"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [summarize(root) for root in args.roots]
    text = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
