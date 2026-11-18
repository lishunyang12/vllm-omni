"""Aggregate GPU kernel activities only, avoiding nested operator double-counting."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(path):
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    assert kernels
    by_name = defaultdict(list)
    for event in kernels:
        by_name[event["name"]].append(event["dur"] / 1000)
    rows = [
        {"kernel": name, "calls": len(times), "total_ms": sum(times),
         "mean_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}
        for name, times in by_name.items()
    ]
    return {
        "trace": path.name,
        "scope": "Diagnostic profile; kernel activity sums can overlap across streams and are not benchmark wall latency.",
        "kernel_activity_sum_ms": sum(e["dur"] for e in kernels) / 1000,
        "kernel_time_span_ms": (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) / 1000,
        "kernels": sorted(rows, key=lambda r: r["total_ms"], reverse=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps([summarize(p) for p in args.traces], indent=2) + "\n")


if __name__ == "__main__":
    main()
