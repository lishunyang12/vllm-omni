# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SM120 check of the SubBlock sparse + Sage backend against the exact BF16 attention it replaces."""

import math

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "RTX5090"})]


@pytest.fixture(autouse=True)
def sm120_runtime():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires an SM120 GPU")
    pytest.importorskip("flashinfer.cute_dsl.sparse.bsa_attn_sm120")


def _local_qkv(rows: int, heads: int, seed: int = 0):
    """Token-local structure shared by q and k, so block routing has something to find."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def draw():
        return torch.randn(1, rows, heads, 128, device="cuda", dtype=torch.float32, generator=g)

    def smooth():
        window = 96
        c = torch.cumsum(draw(), 1)
        s = (c[:, window:] - c[:, :-window]) / math.sqrt(window)
        return torch.cat([s, s[:, -window:]], 1)

    base = smooth()
    q, k, v = 1.2 * base + 0.5 * draw(), 1.2 * base + 0.5 * draw(), draw()
    return [x.to(torch.bfloat16).contiguous() for x in (q, k, v)]


def _metadata(prefix: int, live: int):
    from vllm_omni.diffusion.attention.backends.abstract import (
        AttentionMetadata,
        PackedPaddingMetadata,
        VideoTokenLayout,
    )

    cu = torch.tensor([0, live], dtype=torch.int32, device="cuda")
    return AttentionMetadata(
        video_layout=VideoTokenLayout(prefix_len=prefix, latent_grid=(1, 1, live - prefix)),
        packed_padding=PackedPaddingMetadata(q_length=live, kv_length=live, cu_seqlens_q=cu, cu_seqlens_k=cu),
    )


def _impl(**subblock):
    from vllm_omni.diffusion.attention.backends.subblock_sage_attn import SubBlockSageAttentionImpl

    return SubBlockSageAttentionImpl(
        num_heads=4,
        head_size=128,
        softmax_scale=128**-0.5,
        prefix="blocks.3.attn",
        backend_kwargs={"subblock": {"start_step": 0, **subblock}},
    )


def test_sparse_output_tracks_dense_and_restores_zero_pad_rows():
    rows, live, prefix = 8200, 8193, 100
    q, k, v = _local_qkv(rows, heads=4)
    md = _metadata(prefix, live)
    impl = _impl()
    assert impl._decline_reason(q, k, v, md) is None
    sparse = impl.forward_cuda(q, k, v, md)
    dense = impl._fallback(q, k, v, md, "reference")
    assert sparse.shape == q.shape and torch.isfinite(sparse).all()
    assert (sparse[:, live:] == 0).all()
    a, b = sparse[:, :live].float(), dense[:, :live].float()
    rel_l2 = float((a - b).norm() / b.norm())
    cosine = float((a * b).sum() / (a.norm() * b.norm()))
    assert rel_l2 < 0.15 and cosine > 0.98, (rel_l2, cosine)


def test_dense_prefix_rows_are_exact_up_to_quantization():
    rows, live, prefix = 8192, 8192, 640
    q, k, v = _local_qkv(rows, heads=4, seed=1)
    md = _metadata(prefix, live)
    impl = _impl()
    sparse = impl.forward_cuda(q, k, v, md)[:, :prefix].float()
    dense = impl._fallback(q, k, v, md, "reference")[:, :prefix].float()
    # Prefix query blocks attend to every key block: only INT8/FP8 rounding separates them from dense.
    assert float((sparse - dense).norm() / dense.norm()) < 0.05


def test_dense_steps_match_the_platform_attention_bit_for_bit():
    q, k, v = _local_qkv(4224, heads=4, seed=2)
    md = _metadata(64, 4224)
    impl = _impl(start_step=10)  # outside a forward context the schedule keeps everything dense
    assert impl._decline_reason(q, k, v, md) == "dense denoise step"
    assert torch.equal(impl.forward_cuda(q, k, v, md), impl._fallback(q, k, v, md, "reference"))
