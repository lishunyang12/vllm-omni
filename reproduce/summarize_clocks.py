"""Join read-only GPU telemetry to measured per-rank forward intervals."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.telemetry.open()))
    protocol = json.loads((args.root / "protocol.json").read_text())
    results = []
    for rank in range(protocol["engine"]["num_gpus"]):
        report = json.loads((args.root / f"rank-{rank}.json").read_text())
        for variant in ("dense", "sol"):
            windows = [(s["start_unix_s"] + 0.5, s["start_unix_s"] + s["wall_s"] - 0.5)
                       for s in report["samples"] if s["variant"] == variant]
            selected = [r for r in rows if int(r["index"]) == rank and
                        any(start <= float(r["unix_s"]) <= end for start, end in windows)]
            assert selected, (rank, variant)
            stats = {}
            for field in ("clocks.sm", "power.draw", "temperature.gpu", "utilization.gpu"):
                values = [float(r[field].split()[0]) for r in selected]
                stats[field] = {"min": min(values), "median": statistics.median(values), "max": max(values)}
            for field in ("clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_thermal_slowdown"):
                stats[field] = {"active_fraction": sum(r[field] == "Active" for r in selected) / len(selected)}
            results.append({"rank": rank, "variant": variant, "telemetry_samples": len(selected), "stats": stats})
    args.output.write_text(json.dumps({
        "scope": "Read-only samples within measured forward windows; excludes 0.5s at each boundary. GPU indices equal ranks in this run.",
        "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
