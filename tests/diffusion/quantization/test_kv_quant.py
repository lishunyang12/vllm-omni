# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP8 KV quantization utilities."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def test_roundtrip_preserves_values():
    """quantize_kv_fp8 -> dequantize_fp8 should preserve values within FP8 tolerance."""
    from vllm_omni.quantization.kv_quant import dequantize_fp8, quantize_kv_fp8

    torch.manual_seed(42)
    key = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16)
    value = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16)

    fp8_key, fp8_value, k_scale, v_scale = quantize_kv_fp8(key, value)

    assert fp8_key.dtype == torch.float8_e4m3fn
    assert fp8_value.dtype == torch.float8_e4m3fn
    assert k_scale.numel() == 1
    assert v_scale.numel() == 1

    key_rt = dequantize_fp8(fp8_key, k_scale, torch.bfloat16)
    value_rt = dequantize_fp8(fp8_value, v_scale, torch.bfloat16)

    assert key_rt.shape == key.shape
    assert value_rt.shape == value.shape

    # FP8 e4m3 has ~0.1% relative error for typical values
    torch.testing.assert_close(key_rt, key, rtol=0.05, atol=0.05)
    torch.testing.assert_close(value_rt, value, rtol=0.05, atol=0.05)


def test_scales_are_positive():
    from vllm_omni.quantization.kv_quant import quantize_kv_fp8

    key = torch.randn(1, 64, 4, 32, dtype=torch.bfloat16)
    value = torch.randn(1, 64, 4, 32, dtype=torch.bfloat16)

    _, _, k_scale, v_scale = quantize_kv_fp8(key, value)
    assert k_scale > 0
    assert v_scale > 0


def test_zero_tensor():
    """All-zero input should not produce NaN or Inf."""
    from vllm_omni.quantization.kv_quant import dequantize_fp8, quantize_kv_fp8

    key = torch.zeros(1, 16, 4, 32, dtype=torch.bfloat16)
    value = torch.zeros(1, 16, 4, 32, dtype=torch.bfloat16)

    fp8_key, fp8_value, k_scale, v_scale = quantize_kv_fp8(key, value)
    key_rt = dequantize_fp8(fp8_key, k_scale, torch.bfloat16)
    value_rt = dequantize_fp8(fp8_value, v_scale, torch.bfloat16)

    assert not torch.isnan(key_rt).any()
    assert not torch.isnan(value_rt).any()
    assert torch.allclose(key_rt, key)


def test_fp16_input():
    """Should work with float16 input as well."""
    from vllm_omni.quantization.kv_quant import quantize_kv_fp8

    key = torch.randn(1, 32, 4, 64, dtype=torch.float16)
    value = torch.randn(1, 32, 4, 64, dtype=torch.float16)

    fp8_key, fp8_value, k_scale, v_scale = quantize_kv_fp8(key, value)
    assert fp8_key.dtype == torch.float8_e4m3fn
    assert fp8_value.dtype == torch.float8_e4m3fn


def test_kv_quantization_config_field():
    """OmniDiffusionConfig should accept kv_quantization field."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test", kv_quantization=True)
    assert config.kv_quantization is True

    config_default = OmniDiffusionConfig(model="test")
    assert config_default.kv_quantization is False


def test_attention_metadata_scales():
    """AttentionMetadata should have k_scale and v_scale fields."""
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

    meta = AttentionMetadata()
    assert meta.k_scale is None
    assert meta.v_scale is None

    scale = torch.tensor(0.5)
    meta.k_scale = scale
    meta.v_scale = scale
    assert meta.k_scale is scale
