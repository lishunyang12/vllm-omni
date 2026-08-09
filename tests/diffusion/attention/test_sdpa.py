# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm_omni.diffusion.attention.backends.sdpa as sdpa_backend
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_sdpa_slices_valid_kv_prefix_without_padding_mask(monkeypatch):
    observed = {}

    def fake_sdpa(query, key, value, **kwargs):
        observed.update(query=query, key=key, value=value, kwargs=kwargs)
        return query

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    impl = SDPAImpl(num_heads=2, head_size=4, softmax_scale=0.5)
    query = torch.randn(1, 8, 2, 4)

    output = impl.forward_cuda(
        query,
        query,
        query,
        AttentionMetadata(extra={"valid_kv_length": 5}),
    )

    assert SDPABackend.supports_prefix_kv_slicing
    assert output.shape == query.shape
    assert observed["query"].shape == (1, 2, 8, 4)
    assert observed["key"].shape == (1, 2, 5, 4)
    assert observed["value"].shape == (1, 2, 5, 4)
    assert observed["kwargs"]["attn_mask"] is None


def test_sdpa_rejects_invalid_valid_kv_length():
    impl = SDPAImpl(num_heads=2, head_size=4, softmax_scale=0.5)
    query = torch.randn(1, 8, 2, 4)

    with pytest.raises(ValueError, match="valid_kv_length"):
        impl.forward_cuda(
            query,
            query,
            query,
            AttentionMetadata(extra={"valid_kv_length": 9}),
        )


def test_sdpa_valid_prefix_matches_padding_mask():
    impl = SDPAImpl(num_heads=2, head_size=4, softmax_scale=0.5)
    query = torch.randn(1, 8, 2, 4)
    key = torch.randn(1, 8, 2, 4)
    value = torch.randn(1, 8, 2, 4)

    sliced = impl.forward_cuda(
        query,
        key,
        value,
        AttentionMetadata(extra={"valid_kv_length": 5}),
    )
    masked = impl.forward_cuda(
        query,
        key,
        value,
        AttentionMetadata(attn_mask=torch.tensor([[True] * 5 + [False] * 3])),
    )

    torch.testing.assert_close(sliced, masked)


def test_sdpa_expands_kv_when_native_gqa_kernel_is_unavailable(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: False)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=8, softmax_scale=0.5)
    output = impl.forward_cuda(
        torch.randn(1, 3, 4, 8),
        torch.randn(1, 3, 2, 8),
        torch.randn(1, 3, 2, 8),
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == key_shape == value_shape == (1, 4, 3, 8)
    assert kwargs["enable_gqa"] is False
    assert output.shape == (1, 3, 4, 8)


def test_sdpa_keeps_compressed_kv_when_native_gqa_kernel_is_available(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: True)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=8, softmax_scale=0.5)
    output = impl.forward_cuda(
        torch.randn(1, 3, 4, 8),
        torch.randn(1, 3, 2, 8),
        torch.randn(1, 3, 2, 8),
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == (1, 4, 3, 8)
    assert key_shape == value_shape == (1, 2, 3, 8)
    assert kwargs["enable_gqa"] is True
    assert output.shape == (1, 3, 4, 8)
