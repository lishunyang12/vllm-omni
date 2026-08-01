# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.attention.backends import flashinfer_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _impl(*, causal: bool = False):
    return flashinfer_attn.FlashInferAttentionImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=0.5,
        causal=causal,
    )


def test_flashinfer_rejects_float_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.zeros(2, 2))

    with pytest.raises(ValueError, match="boolean-only"):
        _impl().forward_cuda(query, query, query, metadata)


def test_flashinfer_rejects_causal_custom_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    with pytest.raises(ValueError, match="causal=True"):
        _impl(causal=True).forward_cuda(query, query, query, metadata)
