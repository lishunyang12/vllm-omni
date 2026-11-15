"""Extract first-request timings from completed quality-run metadata (no GPU needed)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="Directory containing matrix.json and outputs/")
    args = parser.parse_args()
    spec = json.loads((args.work / "matrix.json").read_text())
    rows = []
    for model, case in spec["cases"].items():
        for variant, config in spec["variants"].items():
            relative = Path("outputs") / model / variant / "metadata.json"
            raw = (args.work / relative).read_bytes()
            metadata = json.loads(raw)
            if metadata["status"] != "completed":
                raise ValueError(f"Incomplete case: {relative}")
            if metadata["case"] != case or metadata["variant"] != config:
                raise ValueError(f"Case differs from manifest: {relative}")
            row = {
                "model": model,
                "variant": variant,
                "source_commit": config["commit"],
                "width": case["width"],
                "height": case["height"],
                "frames": case["frames"],
                "steps": case["steps"],
                "samples": 1,
                "explicit_warmup_requests": 0,
                "initialization_seconds": metadata["initialization_seconds"],
                "generation_seconds": metadata["generation_seconds"],
                "metadata_path": relative.as_posix(),
                "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            }
            for key in ("initialization_seconds", "generation_seconds"):
                if not math.isfinite(row[key]) or row[key] <= 0:
                    raise ValueError(f"Invalid {key}: {relative}")
            rows.append(row)

    report = {
        "measurement": "First synchronous offline Omni.generate request after initialization",
        "units": "seconds",
        "clock": "time.perf_counter wall clock",
        "hardware": "4 NVIDIA Rubin SM107 GPUs",
        "concurrency": 1,
        "method": {
            "initialization": "Wall time of Omni(**kwargs), reported separately",
            "generation": "Wall time around synchronous Omni.generate, including sampling-parameter construction",
            "excludes": ["model download", "Omni initialization", "subsequent frame conversion and PNG saving", "MP4 encoding", "quality scoring"],
            "explicit_warmup_requests": 0,
            "samples_per_case": 1,
            "cache_policy": "Fresh process per case; persistent caches were not reset between cases",
            "limitations": [
                "May include first-use kernel compilation and autotuning; cache state is not controlled",
                "No warmed latency distribution, P50/P95 or causal speedup estimate",
                "Offline API timing, not HTTP client latency or an SGLang comparison",
                "H3 video uses the full t2va request; only video was saved and quality-scored",
            ],
        },
        "results": rows,
    }
    (args.work / "latency.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.work / "latency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print("| Model | Base BF16 (s) | PR BF16 (s) | PR MXFP8 (s) | PR blockwise FP8 (s) |")
    print("|---|---:|---:|---:|---:|")
    by_case = {(row["model"], row["variant"]): row for row in rows}
    for model in spec["cases"]:
        values = [f"{by_case[model, variant]['generation_seconds']:.3f}" for variant in spec["variants"]]
        print("| " + " | ".join([model, *values]) + " |")


if __name__ == "__main__":
    main()
