# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the SubBlock sparse + Sage backend (plan building, admission, config surface)."""

import pytest
import torch

from vllm_omni.diffusion.attention.backends import subblock_sage_attn as sb
from vllm_omni.diffusion.data import AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_first_sparse_query_block_rounds_up():
    assert sb.first_sparse_query_block(0) == 0
    assert sb.first_sparse_query_block(1) == 1
    assert sb.first_sparse_query_block(64) == 1
    assert sb.first_sparse_query_block(65) == 2


def test_mixed_plan_keeps_routed_rows_and_dense_rows():
    gq, gk, topk = 5, 12, 4
    index = torch.randint(0, gk, (1, 2, gq, topk), dtype=torch.int32)
    mask = sb.sparse_query_block_mask(live_rows=gq * 64, dense_prefix_rows=100, device=torch.device("cpu"))
    assert mask.tolist() == [False, False, True, True, True]
    indices, counts = sb.build_mixed_plan(index, topk, gk, mask)
    assert indices.shape == (1, 2, gq, gk)
    assert counts.dtype == torch.int32
    assert (counts[..., :2] == gk).all() and (counts[..., 2:] == topk).all()
    assert torch.equal(indices[:, :, 2:, :topk], index[:, :, 2:])
    # Dense rows enumerate every key block exactly once.
    for row in indices[0, 0, :2]:
        assert sorted(row.tolist()) == list(range(gk))


def test_attention_spec_gates_subblock_to_its_backend():
    spec = AttentionSpec(backend="SUBBLOCK_SAGE_ATTN")
    kw = spec.backend_kwargs()["subblock"]
    assert kw["sparsity"] == 0.75 and kw["start_step"] == 10 and kw["min_seq_len"] == 4096
    with pytest.raises(ValueError, match="subblock is only supported"):
        AttentionSpec(backend="FLASH_ATTN", subblock={"sparsity": 0.5})
    with pytest.raises(ValueError, match="sparsity"):
        AttentionSpec(backend="SUBBLOCK_SAGE_ATTN", subblock={"sparsity": 1.0})
    assert AttentionSpec(backend="SUBBLOCK_SAGE_ATTN", subblock={"skip_layers": "0,3"}).backend_kwargs()["subblock"][
        "skip_layers"
    ] == [0, 3]


def _impl(**subblock):
    return sb.SubBlockSageAttentionImpl(
        num_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        prefix="blocks.4.attn",
        backend_kwargs={"subblock": subblock},
    )


def test_dense_steps_and_skipped_layers_decline():
    q = torch.empty(1, 8192, 2, 128, dtype=torch.bfloat16, device="meta")
    assert (
        _impl(start_step=0)._decline_reason(q, q, q, None)
        == "no packed video layout (the dense prefix rows are unknown)"
    )
    assert _impl(skip_layers=[4])._decline_reason(q, q, q, None) == "layer is in skip_layers"
    # Outside a forward context the schedule only enables step 0 specs.
    assert _impl(start_step=10)._decline_reason(q, q, q, None) == "dense denoise step"


def test_non_bf16_and_cross_attention_decline():
    impl = _impl(start_step=0)
    q = torch.empty(1, 8192, 2, 128, dtype=torch.float16, device="meta")
    assert "BF16" in impl._decline_reason(q, q, q, None)
    qb = q.to(torch.bfloat16)
    kv = torch.empty(1, 512, 2, 128, dtype=torch.bfloat16, device="meta")
    assert "cross" in impl._decline_reason(qb, kv, kv, None)


def test_subblock_is_accepted_in_per_role_config():
    from vllm_omni.diffusion.data import AttentionConfig

    cfg = AttentionConfig(
        per_role={"self": {"backend": "SUBBLOCK_SAGE_ATTN", "subblock": {"start_step": 20, "skip_layers": "15"}}}
    )
    spec = cfg.per_role["self"]
    assert spec.backend == "SUBBLOCK_SAGE_ATTN"
    assert spec.subblock.start_step == 20 and spec.subblock.skip_layer_indices == {15}


def test_live_rows_prefers_packed_padding_then_layout():
    from vllm_omni.diffusion.attention.backends.abstract import (
        AttentionMetadata,
        PackedPaddingMetadata,
        VideoTokenLayout,
    )

    layout = VideoTokenLayout(prefix_len=100, latent_grid=(2, 3, 4))
    impl_cls = sb.SubBlockSageAttentionImpl
    assert impl_cls._live_rows(AttentionMetadata(video_layout=layout), 1000) == 124
    cu = torch.tensor([0, 90], dtype=torch.int32)
    packed = PackedPaddingMetadata(q_length=90, kv_length=90, cu_seqlens_q=cu, cu_seqlens_k=cu)
    assert impl_cls._live_rows(AttentionMetadata(video_layout=layout, packed_padding=packed), 1000) == 90
    assert impl_cls._dense_prefix_rows(AttentionMetadata(video_layout=layout)) == 100
