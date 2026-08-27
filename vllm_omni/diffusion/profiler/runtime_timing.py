# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Low-overhead timing for diffusion communication and offload profiling."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLE_ENV = "VLLM_OMNI_DIFFUSION_TIMING"


class DiffusionRuntimeTiming:
    """Collect CPU durations and deferred CUDA-event durations per denoise run."""

    def __init__(self) -> None:
        self.enabled = os.getenv(_ENABLE_ENV, "0") == "1" and torch.cuda.is_available()
        self._lock = Lock()
        self._active = False
        self._request_index = 0
        self._cuda_pairs: list[tuple[str, Any, Any, int]] = []
        self._cpu_ms: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._bytes: dict[str, int] = defaultdict(int)

    @property
    def active(self) -> bool:
        return self.enabled and self._active

    def begin_request(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._request_index += 1
            self._cuda_pairs.clear()
            self._cpu_ms.clear()
            self._counts.clear()
            self._bytes.clear()
            self._active = True

    def start_cuda(self, stream: Any | None = None) -> Any | None:
        if not self.active:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    def finish_cuda(
        self,
        name: str,
        start: Any | None,
        *,
        stream: Any | None = None,
        num_bytes: int = 0,
    ) -> None:
        if start is None or not self.active:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record(stream)
        with self._lock:
            self._cuda_pairs.append((name, start, end, num_bytes))

    def wait_event(self, name: str, stream: Any, event: Any) -> None:
        if not self.active:
            stream.wait_event(event)
            return
        start = self.start_cuda(stream)
        stream.wait_event(event)
        self.finish_cuda(name, start, stream=stream)

    @contextmanager
    def cpu_range(self, name: str, *, num_bytes: int = 0) -> Iterator[None]:
        if not self.active:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._cpu_ms[name] += elapsed_ms
                self._counts[name] += 1
                self._bytes[name] += num_bytes

    def finish_request(self) -> dict[str, Any] | None:
        if not self.active:
            return None

        torch.accelerator.synchronize()
        with self._lock:
            cuda_pairs = list(self._cuda_pairs)
            totals = dict(self._cpu_ms)
            counts = dict(self._counts)
            byte_counts = dict(self._bytes)
            request_index = self._request_index
            self._active = False

        for name, start, end, num_bytes in cuda_pairs:
            totals[name] = totals.get(name, 0.0) + float(start.elapsed_time(end))
            counts[name] = counts.get(name, 0) + 1
            byte_counts[name] = byte_counts.get(name, 0) + num_bytes

        metrics = {
            name: {
                "total_ms": round(total_ms, 6),
                "count": counts.get(name, 0),
                "bytes": byte_counts.get(name, 0),
            }
            for name, total_ms in sorted(totals.items())
        }
        transfer_ms = sum(metrics.get(name, {}).get("total_ms", 0.0) for name in ("dlo.h2d", "dlo.allgather"))
        wait_ms = sum(metrics.get(name, {}).get("total_ms", 0.0) for name in ("dlo.prefetch_wait", "dlo.resident_wait"))
        hidden_ms = max(0.0, transfer_ms - wait_ms)
        derived = {
            "dlo_transfer_ms": round(transfer_ms, 6),
            "dlo_exposed_wait_ms": round(wait_ms, 6),
            "dlo_hidden_ms": round(hidden_ms, 6),
            "dlo_overlap_pct": round(100.0 * hidden_ms / transfer_ms, 3) if transfer_ms else 0.0,
        }
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        payload = {
            "rank": rank,
            "request": request_index,
            "metrics": metrics,
            "derived": derived,
        }
        logger.info("[DiffusionRuntimeTiming] %s", json.dumps(payload, sort_keys=True))
        return payload


_RUNTIME_TIMING = DiffusionRuntimeTiming()


def get_diffusion_runtime_timing() -> DiffusionRuntimeTiming:
    return _RUNTIME_TIMING
