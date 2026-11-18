"""Read-only GPU telemetry for correlating recorded forward windows with clocks."""

import argparse
import csv
import io
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", required=True, help="Only the GPUs allocated to this test, e.g. 0,1")
    parser.add_argument("--stop-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    fields = ["index", "clocks.sm", "power.draw", "power.limit", "clocks_event_reasons.sw_power_cap",
              "clocks_event_reasons.hw_thermal_slowdown", "temperature.gpu", "utilization.gpu"]
    command = ["nvidia-smi", "-i", args.indices, "--query-gpu=" + ",".join(fields), "--format=csv,noheader"]
    deadline = time.monotonic() + args.seconds
    with args.output.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unix_s", *fields])
        while time.monotonic() < deadline:
            before = time.time()
            result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=10)
            timestamp = (before + time.time()) / 2
            for row in csv.reader(io.StringIO(result.stdout)):
                writer.writerow([timestamp, *[cell.strip() for cell in row]])
            handle.flush()
            if args.stop_file.exists():
                break
            time.sleep(0.25)


if __name__ == "__main__":
    main()
