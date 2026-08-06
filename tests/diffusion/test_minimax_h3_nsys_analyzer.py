# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _load_module():
    script = Path(__file__).parents[2] / "examples/offline_inference/minimax_h3/analyze_nsys.py"
    spec = importlib.util.spec_from_file_location("analyze_minimax_h3_nsys", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_aggregate_categories_and_gpu_balance(tmp_path):
    analyzer = _load_module()
    assert analyzer.classify_kernel("cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16") == "Dense Attention"
    database = tmp_path / "report.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, text TEXT, textId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER,
                demangledName INTEGER, shortName INTEGER
            );
            """
        )
        strings = [
            (1, "ncclKernel_AllGather"),
            (2, "ncclDevKernel_SendRecv"),
            (3, "fmhaSm120Kernel"),
            (4, "gemm_kernel"),
        ]
        connection.executemany("INSERT INTO StringIds VALUES (?, ?)", strings)
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)",
            [
                (0, 1000, "minimax_h3_task:t2va"),
                (2000, 3000, "minimax_h3_task:fl2va_first_frame"),
            ],
        )
        kernels = []
        for base in (0, 2000):
            for device in range(4):
                offset = base + device * 2
                kernels.extend(
                    [
                        (offset + 10, offset + 110, device, 1, 1),
                        (offset + 120, offset + 170, device, 2, 2),
                        (offset + 180, offset + 380, device, 3, 3),
                        (offset + 390, offset + 790, device, 4, 4),
                    ]
                )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
            kernels,
        )

    results = analyzer.analyze(database)
    assert set(results) == {"T2V", "I2V"}
    t2v = results["T2V"]
    assert t2v["categories"]["NCCL AllGather"]["percent"] == 100 / 750 * 100
    assert t2v["categories"]["NCCL SendRecv"]["percent"] == 50 / 750 * 100
    assert t2v["categories"]["Dense Attention"]["percent"] == 200 / 750 * 100
    assert t2v["nccl_total"]["percent"] == 150 / 750 * 100
    assert t2v["load_balance"]["max_deviation_percent"] == 0
    markdown = analyzer.render_markdown(results)
    assert "### T2V" in markdown and "### I2V" in markdown
    assert "NCCL 总计：20.00%" in markdown
