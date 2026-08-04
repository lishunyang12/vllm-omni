# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

_DEVICE = torch.device("cuda")
_HEAD_DIM = 128
_ROTARY_DIM = 96
_EPS = 1e-5


def _reference(q, k, q_weight, k_weight, rope_table):
    q = F.rms_norm(q, (_HEAD_DIM,), q_weight, _EPS)
    k = F.rms_norm(k, (_HEAD_DIM,), k_weight, _EPS)
    cos = rope_table[..., : _ROTARY_DIM // 2].to(q.dtype).unsqueeze(1)
    sin = rope_table[..., _ROTARY_DIM // 2 :].to(q.dtype).unsqueeze(1)

    def apply(x):
        first = x[..., : _ROTARY_DIM // 2]
        second = x[..., _ROTARY_DIM // 2 : _ROTARY_DIM]
        rotated = torch.cat(
            (
                first * cos - second * sin,
                second * cos + first * sin,
                x[..., _ROTARY_DIM:],
            ),
            dim=-1,
        )
        return rotated

    return apply(q), apply(k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("seq_len", [1, 2, 7, 128, 257, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_minimax_h3_fused_qk_norm_rope_matches_reference(seq_len, dtype):
    from vllm_omni.diffusion.models.minimax_h3.fused_qk_norm_rope import (
        fused_qk_rmsnorm_rope,
    )

    torch.manual_seed(17)
    q_heads = 14
    k_heads = 14
    qkv = torch.randn(
        seq_len,
        (q_heads + k_heads + k_heads) * _HEAD_DIM,
        device=_DEVICE,
        dtype=dtype,
    )
    q = qkv[:, : q_heads * _HEAD_DIM].view(seq_len, q_heads, _HEAD_DIM)
    k_start = q_heads * _HEAD_DIM
    k = qkv[:, k_start : k_start + k_heads * _HEAD_DIM].view(seq_len, k_heads, _HEAD_DIM)
    q_weight = torch.randn(_HEAD_DIM, device=_DEVICE, dtype=dtype)
    k_weight = torch.randn(_HEAD_DIM, device=_DEVICE, dtype=dtype)
    raw_freqs = torch.randn(seq_len, _ROTARY_DIM, device=_DEVICE, dtype=torch.float32)
    half = _ROTARY_DIM // 2
    rope_table = torch.cat(
        (torch.cos(raw_freqs[:, :half]), torch.sin(raw_freqs[:, :half])),
        dim=-1,
    ).to(dtype)

    ref_q, ref_k = _reference(q, k, q_weight, k_weight, rope_table)
    out_q, out_k = fused_qk_rmsnorm_rope(q, k, q_weight, k_weight, rope_table, _EPS)

    if dtype == torch.bfloat16:
        atol = rtol = 3e-2
    else:
        atol = rtol = 1e-5
    torch.testing.assert_close(out_q, ref_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_k, ref_k, atol=atol, rtol=rtol)
