# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.distributed import comm

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeFastGroup:
    def __init__(self) -> None:
        self.modes: list[int] = []
        self.outputs: list[torch.Tensor | None] = []
        self.allocations = 0

    def all_to_all_4d(self, input_tensor: torch.Tensor, *, mode: int, out: torch.Tensor | None = None) -> torch.Tensor:
        self.modes.append(mode)
        self.outputs.append(out)
        return input_tensor if out is None else out

    def empty_output(self, input_tensor: torch.Tensor, *, mode: int) -> torch.Tensor:
        self.allocations += 1
        return torch.empty_like(input_tensor)


@pytest.mark.parametrize(("scatter_idx", "gather_idx", "expected_mode"), [(2, 1, 0), (1, 2, 1)])
def test_fast_ulysses_4d_mode_mapping(monkeypatch, scatter_idx, gather_idx, expected_mode):
    fake_group = _FakeFastGroup()
    monkeypatch.setattr(comm, "_ULYSSES_TRANSPORT", "packed")
    monkeypatch.setattr(comm.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(comm, "_get_fast_ulysses_group", lambda group, backend: fake_group)

    input_tensor = torch.empty((1, 4, 4, 8))
    output = comm.all_to_all_4D(input_tensor, scatter_idx, gather_idx, group=object())

    assert output is input_tensor
    assert fake_group.modes == [expected_mode]


def test_fast_ulysses_zero_copy_reuses_each_output_slot(monkeypatch):
    fake_group = _FakeFastGroup()
    monkeypatch.setattr(comm, "_ULYSSES_TRANSPORT", "pitched")
    monkeypatch.setattr(comm, "_FAST_ULYSSES_ZERO_COPY", True)
    monkeypatch.setattr(comm.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(comm, "_get_fast_ulysses_group", lambda group, backend: fake_group)
    comm._FAST_ULYSSES_OUTPUTS.clear()

    input_tensor = torch.empty((1, 4, 4, 8))
    group = object()
    q_first = comm.all_to_all_4D(input_tensor, group=group, output_slot=0)
    q_second = comm.all_to_all_4D(input_tensor, group=group, output_slot=0)
    k = comm.all_to_all_4D(input_tensor, group=group, output_slot=1)
    resized_q = comm.all_to_all_4D(torch.empty((1, 8, 4, 8)), group=group, output_slot=0)

    assert q_first is q_second
    assert q_first is not k
    assert resized_q is not q_first
    assert fake_group.allocations == 3
    assert fake_group.outputs == [q_first, q_first, k, resized_q]


def test_fast_ulysses_auto_caches_selected_backend(monkeypatch):
    timings = {"pitched": 2.0, "packed": 1.0}
    monkeypatch.setattr(comm, "_benchmark_fast_backend", lambda input, mode, group, backend: timings[backend])
    comm._FAST_ULYSSES_AUTO_BACKENDS.clear()

    input_tensor = torch.empty((1, 4, 4, 8))
    group = object()
    assert comm._select_fast_backend(input_tensor, 0, group) == "packed"
    timings["packed"] = 4.0
    assert comm._select_fast_backend(input_tensor, 0, group) == "packed"


def test_fast_ulysses_auto_prefers_pitched_within_five_percent(monkeypatch):
    monkeypatch.setattr(
        comm,
        "_benchmark_fast_backend",
        lambda input, mode, group, backend: {"pitched": 1.0, "packed": 0.97}[backend],
    )
    comm._FAST_ULYSSES_AUTO_BACKENDS.clear()

    assert comm._select_fast_backend(torch.empty((1, 4, 4, 8)), 0, object()) == "pitched"


def test_fast_ulysses_rejects_5d_collective(monkeypatch):
    monkeypatch.setattr(comm, "_ULYSSES_TRANSPORT", "packed")
    monkeypatch.setattr(comm.dist, "get_world_size", lambda group: 2)

    with pytest.raises(RuntimeError, match="supports 4D Ulysses only"):
        comm.all_to_all_5D(torch.empty((1, 4, 3, 4, 8)), group=object())
