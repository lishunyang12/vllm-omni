# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import pytest
import torch
from torch.nn.attention import SDPBackend

import vllm_omni.diffusion.attention.backends.cudnn_attn as cudnn_backend
from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_cudnn_backend_does_not_fallback_for_unsupported_shape(monkeypatch):
    selected_backends = []

    @contextmanager
    def fake_sdpa_kernel(backends):
        selected_backends.append(tuple(backends))
        yield

    def reject_shape(*args, **kwargs):
        raise RuntimeError("No available kernel. Aborting execution.")

    monkeypatch.setattr(cudnn_backend, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", reject_shape)

    impl = CuDNNAttentionImpl(num_heads=2, head_size=8, softmax_scale=0.5)
    query = torch.randn(1, 2, 2, 8)
    singleton_kv = torch.randn(1, 1, 2, 8)

    with pytest.raises(RuntimeError, match="No available kernel"):
        impl.forward_cuda(query, singleton_kv, singleton_kv)

    assert selected_backends == [(SDPBackend.CUDNN_ATTENTION,)]
