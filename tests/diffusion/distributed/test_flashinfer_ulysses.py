# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel import ulysses
from vllm_omni.diffusion.distributed import flashinfer_ulysses as fi

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _group():
    return SimpleNamespace(
        ulysses_group=SimpleNamespace(group_name="test"), ulysses_world_size=8, ulysses_rank=0, ring_world_size=1
    )


def test_rdma_requires_explicit_permute_and_strict_mode(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_ULYSSES_A2A_BACKEND", "flashinfer-pcie")
    monkeypatch.setattr(fi, "ensure_flashinfer_pcie_available", lambda: None)
    with pytest.raises(ValueError, match="ulysses_a2a_permute"):
        ulysses.UlyssesParallelAttention(_group(), 2, 1, False)
    strategy = ulysses.UlyssesParallelAttention(_group(), 2, 1, False, True)
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda **kwargs: "advanced_uaa")
    x = torch.empty(1, 2, 56, 4)
    with pytest.raises(ValueError, match="strict"):
        strategy.pre_attention(x, x, x, None)


def test_rdma_dispatches_gate_and_reverse_without_nccl(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_ULYSSES_A2A_BACKEND", "flashinfer-pcie")
    monkeypatch.setattr(fi, "ensure_flashinfer_pcie_available", lambda: None)
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda **kwargs: "strict")
    calls = []

    def scatter(x, group_name, world, slot):
        calls.append((slot, x))
        return x.reshape(1, x.shape[1] * world, x.shape[2] // world, x.shape[3])

    def gather(x, group_name, world):
        calls.append(("o", x))
        return x.reshape(1, x.shape[1] // world, x.shape[2] * world, x.shape[3])

    def forbidden(*args, **kwargs):
        raise AssertionError("RDMA dispatch reached NCCL")

    monkeypatch.setattr(fi, "flashinfer_ulysses_qkv_fwd", scatter)
    monkeypatch.setattr(fi, "flashinfer_ulysses_o_rev", gather)
    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", forbidden)
    monkeypatch.setattr(ulysses.dist, "get_world_size", lambda group: 8)
    strategy = ulysses.UlyssesParallelAttention(_group(), 2, 1, False, True)
    q, k, v, g = [torch.full((1, 2, 56, 4), i, dtype=torch.bfloat16) for i in range(4)]
    query, _, _, metadata, context = strategy.pre_attention(q, k, v, AttentionMetadata(extra={"gate_compress": g}))
    output = strategy.post_attention(query, context)
    assert [slot for slot, _ in calls] == ["q", "k", "v", "g", "o"]
    assert torch.equal(output, q)
    assert torch.equal(metadata.extra["gate_compress"], g.reshape_as(query))


def test_failed_close_keeps_communicator_for_retry(monkeypatch):
    calls = []

    class Communicator:
        def close(self):
            calls.append("close")
            if len(calls) == 1:
                raise RuntimeError("transport failure")

    state = fi._State(object(), 8, 1024, Communicator())
    monkeypatch.setattr(fi, "_STATES", {("test", torch.device("cuda:0")): state})
    with pytest.raises(RuntimeError, match="transport failure"):
        fi.clear_flashinfer_ulysses_communicators()
    assert len(fi._STATES) == 1
    fi.clear_flashinfer_ulysses_communicators()
    assert not fi._STATES
