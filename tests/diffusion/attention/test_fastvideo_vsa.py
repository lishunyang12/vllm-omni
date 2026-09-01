# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (
    FastVideoVSABackend,
    FastVideoVSAImpl,
    _build_h3_block_map,
    _get_h3_tile_metadata,
)
from vllm_omni.diffusion.attention.backends.registry import (
    DiffusionAttentionBackendEnum,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_fastvideo_vsa_backend_is_registered():
    assert DiffusionAttentionBackendEnum.FASTVIDEO_VSA.get_path().endswith("fastvideo_vsa.FastVideoVSABackend")


def test_fastvideo_vsa_reports_missing_optional_kernel(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ImportError, match="fastvideo-kernel"):
        FastVideoVSABackend.validate_available()


def test_fastvideo_vsa_tiles_3d_sequence_and_untiles(monkeypatch):
    calls: dict[str, Any] = {}
    fake_module = types.ModuleType("fastvideo_kernel")

    def fake_video_sparse_attn_bshd(
        q,
        k,
        v,
        variable_block_sizes,
        q_variable_block_sizes,
        compress_attn_weight,
        topk,
        block_size,
    ):
        calls["q_shape"] = tuple(q.shape)
        calls["vbs"] = variable_block_sizes.detach().cpu().tolist()
        calls["q_vbs"] = q_variable_block_sizes.detach().cpu().tolist()
        calls["compress_sum"] = float(compress_attn_weight.detach().cpu().sum())
        calls["compress_shape"] = tuple(compress_attn_weight.shape)
        calls["topk"] = topk
        calls["block_size"] = block_size
        return q + k + v

    setattr(fake_module, "video_sparse_attn_bshd", fake_video_sparse_attn_bshd)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", fake_module)

    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={
            "topk": 1,
            "block_size": (4, 8, 8),
            "min_seq_len": 1,
            "disable_when_sp_active": False,
        },
    )
    query = torch.randn(1, 300, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    attn_metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (3, 10, 10)})
    monkeypatch.setattr(impl, "_fallback_reason", lambda *args, **kwargs: None)

    output = impl.forward_cuda(query, key, value, attn_metadata)

    assert output.shape == query.shape
    assert calls["q_shape"] == (1, 1024, 2, 8)
    assert calls["vbs"] == [192, 48, 48, 12]
    assert calls["q_vbs"] == [192, 48, 48, 12]
    assert calls["compress_shape"] == (1, 1024, 2, 8)
    assert calls["compress_sum"] == 0.0
    assert calls["topk"] == 1
    assert calls["block_size"] == (4, 8, 8)


def test_fastvideo_vsa_uses_learned_gate_when_provided(monkeypatch):
    calls: dict[str, Any] = {}
    fake_module = types.ModuleType("fastvideo_kernel")

    def fake_video_sparse_attn_bshd(
        q,
        k,
        v,
        variable_block_sizes,
        q_variable_block_sizes,
        compress_attn_weight,
        topk,
        block_size,
    ):
        calls["compress_sum"] = float(compress_attn_weight.detach().cpu().sum())
        calls["compress_shape"] = tuple(compress_attn_weight.shape)
        return q + k + v

    setattr(fake_module, "video_sparse_attn_bshd", fake_video_sparse_attn_bshd)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", fake_module)

    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={
            "topk": 1,
            "block_size": (4, 8, 8),
            "min_seq_len": 1,
            "disable_when_sp_active": False,
        },
    )
    query = torch.randn(1, 300, 2, 8)
    gate = torch.ones_like(query)
    metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (3, 10, 10), "gate_compress": gate})
    monkeypatch.setattr(impl, "_fallback_reason", lambda *args, **kwargs: None)

    output = impl.forward_cuda(query, query, query, metadata)

    assert output.shape == query.shape
    assert calls["compress_shape"] == (1, 1024, 2, 8)
    # Only the 300 real tokens carry ones; padded gate slots stay zero.
    assert calls["compress_sum"] == gate.numel()


def test_fastvideo_vsa_falls_back_without_dit_shape(monkeypatch):
    calls: dict[str, Any] = {}

    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={
            "topk": 1,
            "block_size": (4, 8, 8),
            "min_seq_len": 1,
            "disable_when_sp_active": False,
        },
    )

    def fake_fallback(query, key, value, attn_metadata, reason):
        calls["reason"] = reason
        return torch.zeros_like(query)

    monkeypatch.setattr(impl, "_fallback", fake_fallback)

    query = torch.randn(1, 512, 2, 8)
    output = impl.forward_cuda(query, query, query, AttentionMetadata())

    assert output.shape == query.shape
    assert calls["reason"] == "vsa_dit_seq_shape metadata is required"


def test_fastvideo_vsa_falls_back_for_mask(monkeypatch):
    calls: dict[str, Any] = {}

    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={"min_seq_len": 1},
    )

    def fake_fallback(query, key, value, attn_metadata, reason):
        calls["reason"] = reason
        return torch.zeros_like(query)

    monkeypatch.setattr(impl, "_fallback", fake_fallback)

    query = torch.randn(1, 512, 2, 8)
    mask = torch.ones(1, 512, dtype=torch.bool)
    output = impl.forward_cuda(query, query, query, AttentionMetadata(attn_mask=mask))

    assert output.shape == query.shape
    assert calls["reason"]


def test_fastvideo_vsa_allows_topk_equal_to_num_blocks():
    query = torch.randn(1, 512, 2, 8)
    metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (4, 8, 16)})

    all_blocks = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={"topk": 2, "block_size": (4, 8, 8), "min_seq_len": 1},
    )
    too_many = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={"topk": 3, "block_size": (4, 8, 8), "min_seq_len": 1},
    )

    # CPU/float32 is rejected later, but k=N itself must not trigger fallback.
    assert all_blocks._fallback_reason(query, query, query, metadata) == "dtype torch.float32 is not supported"
    assert too_many._fallback_reason(query, query, query, metadata) == "topk 3 > num_blocks 2"


def test_fastvideo_vsa_runs_packed_prefix_with_64_token_override(monkeypatch):
    calls: dict[str, Any] = {}
    fake_module = types.ModuleType("fastvideo_kernel")

    def fake_video_sparse_attn(q, k, v, **kwargs):
        calls["q_shape"] = tuple(q.shape)
        calls["block_size"] = kwargs["block_size"]
        calls["vbs"] = kwargs["variable_block_sizes"].tolist()
        return q + k + v

    setattr(fake_module, "video_sparse_attn", fake_video_sparse_attn)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", fake_module)
    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        backend_kwargs={"topk": 1, "min_seq_len": 1, "disable_when_sp_active": False},
    )
    query = torch.randn(1, 160, 2, 8)
    metadata = AttentionMetadata(
        packed_padding=PackedPaddingMetadata(
            q_length=130,
            kv_length=130,
            cu_seqlens_q=torch.tensor([0, 130], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 130], dtype=torch.int32),
        ),
        extra={"vsa_dit_seq_shape": (1, 1, 130), "vsa_block_size": (1, 1, 64)},
    )
    monkeypatch.setattr(impl, "_fallback_reason", lambda *args, **kwargs: None)

    output = impl.forward_cuda(query, query, query, metadata)

    assert output.shape == query.shape
    assert calls == {"q_shape": (1, 2, 192, 8), "block_size": (1, 1, 64), "vbs": [64, 64, 2]}
    assert torch.count_nonzero(output[:, 130:]) == 0


def test_h3_geometry_keeps_prefix_segments_pure_and_tiles_video_3d():
    partition, sizes, non_pad, untile, prefix_blocks, video_blocks = _get_h3_tile_metadata(
        (5, 70, 9), (5, 6, 6), torch.device("cpu")
    )
    assert sizes[:4].tolist() == [5, 64, 6, 9]
    assert prefix_blocks == 4
    assert video_blocks == 8
    assert int(sizes.sum()) == 5 + 70 + 9 + 5 * 6 * 6
    assert partition.numel() == non_pad.numel() == untile.numel() == int(sizes.sum())


def test_h3_block_map_makes_prefix_queries_dense_and_prefix_keys_exempt():
    scores = torch.arange(1 * 2 * 5 * 5, dtype=torch.float32).reshape(1, 2, 5, 5)
    block_map = _build_h3_block_map(scores, num_prefix_blocks=2, num_video_blocks=3, topk=1)
    assert block_map[:, :, :2].all()
    assert block_map[..., :2].all()
    assert (block_map[:, :, 2:, 2:].sum(dim=-1) == 1).all()
