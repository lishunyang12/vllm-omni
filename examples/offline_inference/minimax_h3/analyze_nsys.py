#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate MiniMax-H3 CUDA kernel time inside task NVTX ranges."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

TASK_LABELS = {
    "minimax_h3_task:t2va": "T2V",
    "minimax_h3_task:fl2va_first_frame": "I2V",
}
CATEGORY_ORDER = (
    "NCCL AllGather",
    "NCCL SendRecv",
    "NCCL Other",
    "Dense Attention",
    "Other",
)
ATTENTION_PATTERN = re.compile(
    r"fmha|sdpa|flash[_ ]?attn|flashattention|attention.*(?:fwd|forward)|"
    r"(?:fwd|forward).*attention|sm120.*fmha",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TaskRange:
    name: str
    label: str
    start: int
    end: int


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if "nccl" in lowered:
        compact = lowered.replace("_", "")
        if "allgather" in compact:
            return "NCCL AllGather"
        if "sendrecv" in compact or "send" in lowered or "recv" in lowered:
            return "NCCL SendRecv"
        return "NCCL Other"
    if ATTENTION_PATTERN.search(name):
        return "Dense Attention"
    return "Other"


def _require_tables(connection: sqlite3.Connection) -> None:
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
    missing = required - tables
    if missing:
        raise ValueError("Nsight SQLite is missing required tables: " + ", ".join(sorted(missing)))


def load_task_ranges(connection: sqlite3.Connection) -> list[TaskRange]:
    rows = connection.execute(
        """
        SELECT n.start, n.end, COALESCE(n.text, strings.value, '') AS range_name
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS strings ON strings.id = n.textId
        WHERE n.end IS NOT NULL
        ORDER BY n.start
        """
    )
    ranges = []
    for start, end, name in rows:
        if name in TASK_LABELS:
            ranges.append(TaskRange(name, TASK_LABELS[name], int(start), int(end)))
    if not ranges:
        expected = ", ".join(TASK_LABELS)
        raise ValueError(f"No MiniMax-H3 task NVTX ranges found. Expected one of: {expected}")
    return ranges


def _kernels_in_range(connection: sqlite3.Connection, task_range: TaskRange) -> Iterable[tuple[int, int, str]]:
    return connection.execute(
        """
        SELECT k.deviceId,
               k.end - k.start AS duration_ns,
               COALESCE(demangled.value, short.value, '') AS kernel_name
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        LEFT JOIN StringIds AS demangled ON demangled.id = k.demangledName
        LEFT JOIN StringIds AS short ON short.id = k.shortName
        WHERE k.start >= ? AND k.end <= ? AND k.end > k.start
        """,
        (task_range.start, task_range.end),
    )


def aggregate_task(
    connection: sqlite3.Connection,
    ranges: list[TaskRange],
    device_ids: tuple[int, ...],
) -> dict[str, object]:
    category_ns: dict[str, int] = defaultdict(int)
    device_ns: dict[int, int] = defaultdict(int, {device: 0 for device in device_ids})
    kernel_count = 0
    for task_range in ranges:
        for device_id, duration_ns, kernel_name in _kernels_in_range(connection, task_range):
            duration = int(duration_ns)
            category_ns[classify_kernel(kernel_name)] += duration
            device_ns[int(device_id)] += duration
            kernel_count += 1

    total_ns = sum(category_ns.values())
    if total_ns <= 0:
        raise ValueError(f"No CUDA kernels found inside {ranges[0].label} NVTX range")
    for category in CATEGORY_ORDER:
        category_ns.setdefault(category, 0)

    nccl_ns = sum(category_ns[name] for name in ("NCCL AllGather", "NCCL SendRecv", "NCCL Other"))
    device_values = list(device_ns.values())
    mean_device_ns = sum(device_values) / len(device_values)
    max_deviation_pct = max(abs(value - mean_device_ns) / mean_device_ns * 100 for value in device_values)
    max_min_skew_pct = (max(device_values) - min(device_values)) / mean_device_ns * 100

    return {
        "label": ranges[0].label,
        "ranges": len(ranges),
        "kernel_count": kernel_count,
        "total_kernel_time_ms": total_ns / 1e6,
        "categories": {
            name: {
                "time_ms": category_ns[name] / 1e6,
                "percent": category_ns[name] / total_ns * 100,
            }
            for name in CATEGORY_ORDER
        },
        "nccl_total": {
            "time_ms": nccl_ns / 1e6,
            "percent": nccl_ns / total_ns * 100,
        },
        "devices": {
            str(device): {
                "time_ms": duration / 1e6,
                "vs_mean_percent": (duration - mean_device_ns) / mean_device_ns * 100,
            }
            for device, duration in sorted(device_ns.items())
        },
        "load_balance": {
            "max_deviation_percent": max_deviation_pct,
            "max_min_skew_percent": max_min_skew_pct,
        },
    }


def analyze(sqlite_path: Path) -> dict[str, dict[str, object]]:
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as connection:
        _require_tables(connection)
        ranges = load_task_ranges(connection)
        grouped: dict[str, list[TaskRange]] = defaultdict(list)
        for task_range in ranges:
            grouped[task_range.label].append(task_range)
        device_ids = tuple(
            int(row[0])
            for row in connection.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId")
        )
        return {label: aggregate_task(connection, task_ranges, device_ids) for label, task_ranges in grouped.items()}


def render_markdown(results: dict[str, dict[str, object]]) -> str:
    sections = []
    for label in ("T2V", "I2V"):
        if label not in results:
            continue
        result = results[label]
        categories = result["categories"]
        nccl = result["nccl_total"]
        balance = result["load_balance"]
        lines = [
            f"### {label}",
            "",
            "GPU kernel 聚合时间占比：",
            "",
            f"- NCCL AllGather：{categories['NCCL AllGather']['percent']:.2f}%",
            f"- NCCL SendRecv：{categories['NCCL SendRecv']['percent']:.2f}%",
            f"- NCCL 其他：{categories['NCCL Other']['percent']:.2f}%",
            f"- NCCL 总计：{nccl['percent']:.2f}%",
            f"- Dense FlashAttention/FMHA：{categories['Dense Attention']['percent']:.2f}%",
            f"- 其余 GEMM、norm、elementwise、VAE：{categories['Other']['percent']:.2f}%",
            "",
            "GPU 负载均衡（累计 kernel time）：",
            "",
        ]
        for device, values in result["devices"].items():
            lines.append(f"- GPU {device}：{values['time_ms']:.2f} ms （相对均值 {values['vs_mean_percent']:+.2f}%）")
        lines.extend(
            [
                f"- 最大偏离均值：{balance['max_deviation_percent']:.2f}%",
                f"- max-min/mean：{balance['max_min_skew_percent']:.2f}%",
                f"- 聚合 kernel 时间：{result['total_kernel_time_ms']:.2f} ms",
            ]
        )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, help="SQLite exported from an .nsys-rep")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sqlite_path = args.sqlite.expanduser().resolve()
    results = analyze(sqlite_path)
    print(render_markdown(results))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
