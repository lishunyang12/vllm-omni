# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check the integrated cuDNN provider against an independent FP32 reference."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd
from vllm_omni.diffusion.attention.ops.cudnn_block_sparse import require_cudnn_sparse

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": ["H100", "B200", "RTX5090"]}),
]


@pytest.fixture(autouse=True)
def cudnn_runtime():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10, 11, 12):
        pytest.skip("requires a cuDNN BSA supported GPU")
    pytest.importorskip("cudnn", minversion="1.31.0")
    require_cudnn_sparse(torch.device("cuda", torch.accelerator.current_device_index()))
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("empty", [False, True])
@torch.inference_mode()
def test_cudnn_sparse_mask_partial_blocks_and_graph_replay(block_size, empty):
    torch.manual_seed(43)
    blocks = 5
    q, k, v = [torch.randn(1, blocks * block_size, 7, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    sizes = torch.tensor([block_size, 17, 0, block_size - 1, 1], device="cuda", dtype=torch.int32)
    if empty:
        sizes.zero_()
    mask = torch.rand(1, 7, blocks, blocks, device="cuda") < 0.4
    mask[..., 0] = True
    mask[..., -1, :] = False
    valid = (torch.arange(block_size, device="cuda")[None] < sizes[:, None]).flatten()
    v[:, ~valid] = 100

    def run():
        return block_sparse_attn_bshd(q, k, v, mask, sizes, 128**-0.5, provider="cudnn", block_size=block_size)

    def reference():
        token_mask = mask.repeat_interleave(block_size, -2).repeat_interleave(block_size, -1)
        token_mask &= valid[None, None, None, :]
        scores = q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2) / 128**0.5
        probabilities = scores.masked_fill(~token_mask, -torch.inf).softmax(-1).nan_to_num()
        return (probabilities @ v.float().transpose(1, 2)).transpose(1, 2)

    torch.testing.assert_close(run().float(), reference(), rtol=0.01, atol=0.01)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    q.copy_(torch.randn_like(q))
    mask.zero_()
    graph.replay()
    assert torch.count_nonzero(captured).item() == 0
    mask[..., 0] = True
    graph.replay()
    torch.testing.assert_close(captured.float(), reference(), rtol=0.01, atol=0.01)
