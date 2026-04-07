# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP8 Q/K/V quantization utilities."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def test_qkv_roundtrip_preserves_values():
    """quantize_qkv_fp8 -> dequantize_fp8 should preserve values within FP8 tolerance."""
    from vllm_omni.quantization.kv_quant import (
        dequantize_fp8,
        quantize_qkv_fp8,
    )

    torch.manual_seed(42)
    query = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16)
    key = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16)
    value = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16)

    fp8_q, fp8_k, fp8_v, q_scale, k_scale, v_scale = quantize_qkv_fp8(
        query, key, value
    )

    assert fp8_q.dtype == torch.float8_e4m3fn
    assert fp8_k.dtype == torch.float8_e4m3fn
    assert fp8_v.dtype == torch.float8_e4m3fn
    assert q_scale.numel() == 1
    assert k_scale.numel() == 1
    assert v_scale.numel() == 1

    query_rt = dequantize_fp8(fp8_q, q_scale, torch.bfloat16)
    key_rt = dequantize_fp8(fp8_k, k_scale, torch.bfloat16)
    value_rt = dequantize_fp8(fp8_v, v_scale, torch.bfloat16)

    # FP8 e4m3 has ~0.1% relative error for typical values
    torch.testing.assert_close(query_rt, query, rtol=0.05, atol=0.05)
    torch.testing.assert_close(key_rt, key, rtol=0.05, atol=0.05)
    torch.testing.assert_close(value_rt, value, rtol=0.05, atol=0.05)


def test_kv_only_roundtrip():
    """quantize_kv_fp8 for joint attention path."""
    from vllm_omni.quantization.kv_quant import (
        dequantize_fp8,
        quantize_kv_fp8,
    )

    torch.manual_seed(42)
    key = torch.randn(1, 64, 4, 32, dtype=torch.bfloat16)
    value = torch.randn(1, 64, 4, 32, dtype=torch.bfloat16)

    fp8_k, fp8_v, k_scale, v_scale = quantize_kv_fp8(key, value)

    assert fp8_k.dtype == torch.float8_e4m3fn
    assert k_scale > 0
    assert v_scale > 0

    key_rt = dequantize_fp8(fp8_k, k_scale, torch.bfloat16)
    torch.testing.assert_close(key_rt, key, rtol=0.05, atol=0.05)


def test_scales_are_positive():
    from vllm_omni.quantization.kv_quant import quantize_qkv_fp8

    q = torch.randn(1, 64, 4, 32, dtype=torch.bfloat16)
    k = torch.randn(1, 64, 4, 32, dtype=torch.bfloat16)
    v = torch.randn(1, 64, 4, 32, dtype=torch.bfloat16)

    _, _, _, q_scale, k_scale, v_scale = quantize_qkv_fp8(q, k, v)
    assert q_scale > 0
    assert k_scale > 0
    assert v_scale > 0


def test_zero_tensor():
    """All-zero input should not produce NaN or Inf."""
    from vllm_omni.quantization.kv_quant import (
        dequantize_fp8,
        quantize_qkv_fp8,
    )

    q = torch.zeros(1, 16, 4, 32, dtype=torch.bfloat16)
    k = torch.zeros(1, 16, 4, 32, dtype=torch.bfloat16)
    v = torch.zeros(1, 16, 4, 32, dtype=torch.bfloat16)

    fp8_q, fp8_k, fp8_v, q_s, k_s, v_s = quantize_qkv_fp8(q, k, v)
    q_rt = dequantize_fp8(fp8_q, q_s, torch.bfloat16)
    k_rt = dequantize_fp8(fp8_k, k_s, torch.bfloat16)

    assert not torch.isnan(q_rt).any()
    assert not torch.isnan(k_rt).any()
    assert torch.allclose(q_rt, q)
    assert torch.allclose(k_rt, k)


def test_fp16_input():
    """Should work with float16 input as well."""
    from vllm_omni.quantization.kv_quant import quantize_qkv_fp8

    q = torch.randn(1, 32, 4, 64, dtype=torch.float16)
    k = torch.randn(1, 32, 4, 64, dtype=torch.float16)
    v = torch.randn(1, 32, 4, 64, dtype=torch.float16)

    fp8_q, fp8_k, fp8_v, _, _, _ = quantize_qkv_fp8(q, k, v)
    assert fp8_q.dtype == torch.float8_e4m3fn
    assert fp8_k.dtype == torch.float8_e4m3fn
    assert fp8_v.dtype == torch.float8_e4m3fn


def test_kv_cache_dtype_config_field():
    """OmniDiffusionConfig should accept kv_cache_dtype field."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test", kv_cache_dtype="fp8")
    assert config.kv_cache_dtype == "fp8"

    config_default = OmniDiffusionConfig(model="test")
    assert config_default.kv_cache_dtype is None


def test_is_quantized_kv_cache():
    """is_quantized_kv_cache should detect FP8 dtype strings."""
    from vllm_omni.quantization.kv_quant import is_quantized_kv_cache

    assert is_quantized_kv_cache("fp8") is True
    assert is_quantized_kv_cache("fp8_e4m3") is True
    assert is_quantized_kv_cache(None) is False
    assert is_quantized_kv_cache("auto") is False
    assert is_quantized_kv_cache("bfloat16") is False


def test_attention_metadata_kv_cache_dtype():
    """AttentionMetadata should have kv_cache_dtype field."""
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

    meta = AttentionMetadata()
    assert meta.kv_cache_dtype is None

    meta.kv_cache_dtype = "fp8"
    assert meta.kv_cache_dtype == "fp8"


def test_fast_qkv_quantization():
    """quantize_qkv_fp8_fast should use scale=1.0 (direct cast)."""
    from vllm_omni.quantization.kv_quant import quantize_qkv_fp8_fast

    q = torch.randn(1, 32, 4, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 32, 4, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 32, 4, 64, dtype=torch.bfloat16)

    fp8_q, fp8_k, fp8_v, q_s, k_s, v_s = quantize_qkv_fp8_fast(q, k, v)

    assert fp8_q.dtype == torch.float8_e4m3fn
    assert fp8_k.dtype == torch.float8_e4m3fn
    assert fp8_v.dtype == torch.float8_e4m3fn
    # Fast path uses scale=1.0
    assert q_s.item() == 1.0
    assert k_s.item() == 1.0
    assert v_s.item() == 1.0


def test_fast_kv_quantization():
    """quantize_kv_fp8_fast for joint attention path."""
    from vllm_omni.quantization.kv_quant import quantize_kv_fp8_fast

    k = torch.randn(1, 64, 4, 32, dtype=torch.bfloat16)
    v = torch.randn(1, 64, 4, 32, dtype=torch.bfloat16)

    fp8_k, fp8_v, k_s, v_s = quantize_kv_fp8_fast(k, v)

    assert fp8_k.dtype == torch.float8_e4m3fn
    assert fp8_v.dtype == torch.float8_e4m3fn
    assert k_s.item() == 1.0
    assert v_s.item() == 1.0


def test_flash_backend_supports_kv_cache_dtype():
    """FlashAttentionBackend should declare FP8 support."""
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend

    assert FlashAttentionBackend.supports_kv_cache_dtype(None) is True
    assert FlashAttentionBackend.supports_kv_cache_dtype("fp8") is True
    assert FlashAttentionBackend.supports_kv_cache_dtype("fp8_e4m3") is True
    assert FlashAttentionBackend.supports_kv_cache_dtype("mxfp8") is False


def test_sdpa_backend_does_not_support_fp8():
    """SDPABackend should not declare FP8 support."""
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend

    assert SDPABackend.supports_kv_cache_dtype(None) is True
    assert SDPABackend.supports_kv_cache_dtype("fp8") is False


def test_handle_kv_cache_dtype_clears_unsupported():
    """_handle_kv_cache_dtype should clear unsupported dtype to None."""
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

    impl = SDPAImpl(num_heads=4, head_size=64, softmax_scale=0.125)
    meta = AttentionMetadata(kv_cache_dtype="fp8")

    # SDPA has empty _supported_kv_cache_dtypes, should clear fp8
    impl._handle_kv_cache_dtype(meta, "cuda")
    assert meta.kv_cache_dtype is None


def test_handle_kv_cache_dtype_preserves_supported():
    """_handle_kv_cache_dtype should preserve supported dtype."""
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

    impl = FlashAttentionImpl(num_heads=4, head_size=64, softmax_scale=0.125)
    meta = AttentionMetadata(kv_cache_dtype="fp8")

    impl._handle_kv_cache_dtype(meta, "cuda")
    assert meta.kv_cache_dtype == "fp8"


def test_handle_kv_cache_dtype_clears_unsupported_platform():
    """FP8 on FlashAttention should be cleared for non-CUDA platforms."""
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

    impl = FlashAttentionImpl(num_heads=4, head_size=64, softmax_scale=0.125)
    meta = AttentionMetadata(kv_cache_dtype="fp8")

    # NPU not in FlashAttentionImpl._supported_kv_cache_dtypes
    impl._handle_kv_cache_dtype(meta, "npu")
    assert meta.kv_cache_dtype is None
