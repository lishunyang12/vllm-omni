# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.distributed import comm

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeFastGroup:
    def __init__(self) -> None:
        self.modes: list[int] = []

    def all_to_all_4d(self, input_tensor: torch.Tensor, *, mode: int) -> torch.Tensor:
        self.modes.append(mode)
        return input_tensor


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


def test_fast_ulysses_rejects_5d_collective(monkeypatch):
    monkeypatch.setattr(comm, "_ULYSSES_TRANSPORT", "packed")
    monkeypatch.setattr(comm.dist, "get_world_size", lambda group: 2)

    with pytest.raises(RuntimeError, match="supports 4D Ulysses only"):
        comm.all_to_all_5D(torch.empty((1, 4, 3, 4, 8)), group=object())
