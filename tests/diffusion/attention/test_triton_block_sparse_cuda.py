# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Portable consumer VSA against independent token-level FP32 attention."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": ["L4", "H100", "RTX5090"]}),
]


@pytest.fixture(autouse=True)
def runtime():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires SM80+")
    pytest.importorskip("fastvideo_kernel.block_sparse_attn")
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("empty,scale", [(False, 1.0), (False, 4.0), (True, 1.0)])
@torch.inference_mode()
def test_sparse_ragged_tiles_and_graph_replay(tile, empty, scale):
    torch.manual_seed(1919)
    q = torch.randn(2, 3 * tile, 3, 128, device="cuda", dtype=torch.bfloat16) * scale
    k = torch.randn(2, 5 * tile, 3, 128, device="cuda", dtype=torch.bfloat16) * scale
    v = torch.randn_like(k)
    sizes = torch.tensor([tile, 17, 0, tile - 1, 1], device="cuda", dtype=torch.int32)
    if empty:
        sizes.zero_()
    mask = torch.rand(2, 3, 3, 5, device="cuda") < 0.5
    mask[..., 0] = True
    mask[..., -1, :] = False

    def reference():
        valid = (torch.arange(tile, device="cuda")[None] < sizes[:, None]).flatten()
        token_mask = mask.repeat_interleave(tile, -2).repeat_interleave(tile, -1) & valid[None, None, None, :]
        scores = q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2) / 128**0.5
        probs = scores.masked_fill(~token_mask, -torch.inf).softmax(-1).nan_to_num()
        return (probs @ v.float().transpose(1, 2)).transpose(1, 2)

    def run():
        return block_sparse_attn_bshd(q, k, v, mask, sizes, 128**-0.5, provider="triton", block_size=tile)

    actual = run()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference(), atol=0.02, rtol=0.02)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    mask.zero_()
    graph.replay()
    assert torch.count_nonzero(captured).item() == 0
    mask[..., 0] = True
    sizes.copy_(torch.tensor([17, tile, 1, 0, tile - 1], device="cuda", dtype=torch.int32))
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.testing.assert_close(captured.float(), reference(), atol=0.02, rtol=0.02)
