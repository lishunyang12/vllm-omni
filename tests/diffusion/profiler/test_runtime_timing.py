# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterator

import pytest

from vllm_omni.diffusion.profiler import runtime_timing as runtime_timing_module

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeCudaEvent:
    def __init__(self, timestamps: Iterator[float]) -> None:
        self._timestamps = timestamps
        self.timestamp = 0.0

    def record(self, _stream=None) -> None:
        self.timestamp = next(self._timestamps)

    def elapsed_time(self, end: "_FakeCudaEvent") -> float:
        return end.timestamp - self.timestamp


class _FakeStream:
    def wait_event(self, _event) -> None:
        return


def test_runtime_timing_collects_deferred_cuda_and_cpu_metrics(monkeypatch) -> None:
    timestamps = iter([0.0, 4.0, 4.0, 10.0, 10.0, 12.0])
    synchronized = []
    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_TIMING", "1")
    monkeypatch.setattr(runtime_timing_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        runtime_timing_module.torch.cuda,
        "Event",
        lambda **_kwargs: _FakeCudaEvent(timestamps),
    )
    monkeypatch.setattr(runtime_timing_module.torch.accelerator, "synchronize", lambda: synchronized.append(True))

    timing = runtime_timing_module.DiffusionRuntimeTiming()
    timing.begin_request()
    h2d_start = timing.start_cuda()
    timing.finish_cuda("dlo.h2d", h2d_start, num_bytes=128)
    gather_start = timing.start_cuda()
    timing.finish_cuda("dlo.allgather", gather_start, num_bytes=256)
    timing.wait_event("dlo.prefetch_wait", _FakeStream(), object())
    with timing.cpu_range("dlo.cpu_pack", num_bytes=512):
        pass

    payload = timing.finish_request()

    assert synchronized == [True]
    assert payload is not None
    assert payload["metrics"]["dlo.h2d"] == {"total_ms": 4.0, "count": 1, "bytes": 128}
    assert payload["metrics"]["dlo.allgather"] == {"total_ms": 6.0, "count": 1, "bytes": 256}
    assert payload["metrics"]["dlo.prefetch_wait"]["total_ms"] == 2.0
    assert payload["metrics"]["dlo.cpu_pack"]["count"] == 1
    assert payload["metrics"]["dlo.cpu_pack"]["bytes"] == 512
    assert payload["derived"] == {
        "dlo_transfer_ms": 10.0,
        "dlo_exposed_wait_ms": 2.0,
        "dlo_hidden_ms": 8.0,
        "dlo_overlap_pct": 80.0,
    }


def test_runtime_timing_is_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_DIFFUSION_TIMING", raising=False)
    monkeypatch.setattr(runtime_timing_module.torch.cuda, "is_available", lambda: True)
    timing = runtime_timing_module.DiffusionRuntimeTiming()

    timing.begin_request()

    assert not timing.active
    assert timing.start_cuda() is None
    assert timing.finish_request() is None
