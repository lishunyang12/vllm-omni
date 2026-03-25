# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MXFP4 quantization config."""

from unittest.mock import MagicMock

import pytest
import torch
from torch.nn import Module, Parameter
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.factory import SUPPORTED_QUANTIZATION_METHODS

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def test_mxfp4_config_creation():
    config = build_quant_config("mxfp4")
    assert config is not None
    assert config.get_name() == "mxfp4"


def test_mxfp4_in_supported_methods():
    assert "mxfp4" in SUPPORTED_QUANTIZATION_METHODS


def test_mxfp4_config_with_custom_params():
    config = build_quant_config(
        "mxfp4",
        ignored_layers=["proj_out", "t_embedder"],
    )
    assert config.activation_scheme == "dynamic"
    assert "proj_out" in config.ignored_layers
    assert "t_embedder" in config.ignored_layers


def test_mxfp4_config_dict():
    config = build_quant_config({"method": "mxfp4", "ignored_layers": ["norm"]})
    assert config.get_name() == "mxfp4"
    assert "norm" in config.ignored_layers


def test_mxfp4_static_activation_rejected():
    with pytest.raises(ValueError, match="dynamic"):
        build_quant_config("mxfp4", activation_scheme="static")


def test_integration_string():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test", quantization_config="mxfp4")
    assert config.quantization_config.get_name() == "mxfp4"


def test_integration_dict():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(
        model="test",
        quantization_config={"method": "mxfp4", "ignored_layers": ["vae"]},
    )
    assert config.quantization_config.get_name() == "mxfp4"
    assert "vae" in config.quantization_config.ignored_layers


def test_integration_per_component():
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.quantization import ComponentQuantizationConfig

    config = OmniDiffusionConfig(
        model="test",
        quantization_config={
            "transformer": {"method": "mxfp4"},
            "vae": None,
        },
    )
    assert isinstance(config.quantization_config, ComponentQuantizationConfig)
    assert config.quantization_config.component_configs["transformer"].get_name() == "mxfp4"
    assert config.quantization_config.component_configs["vae"] is None


def test_get_quant_method_returns_mxfp4():
    from vllm_omni.quantization.mxfp4_config import Mxfp4OnlineLinearMethod

    config = build_quant_config("mxfp4")
    layer = MagicMock(spec=LinearBase)
    method = config.get_quant_method(layer, "transformer.blocks.0.attn.to_q")
    assert isinstance(method, Mxfp4OnlineLinearMethod)


def test_get_quant_method_skips_ignored():
    config = build_quant_config("mxfp4", ignored_layers=["proj_out"])
    layer = MagicMock(spec=LinearBase)
    method = config.get_quant_method(layer, "proj_out")
    assert isinstance(method, UnquantizedLinearMethod)


def test_get_quant_method_non_linear_returns_none():
    config = build_quant_config("mxfp4")
    layer = MagicMock(spec=Module)
    assert config.get_quant_method(layer, "some_norm") is None


def test_quantize_dequant_round_trip():
    from vllm_omni.quantization.mxfp4_config import dequant_mxfp4, quantize_weight_mxfp4

    torch.manual_seed(42)
    weight = torch.randn(64, 128)

    packed, scales = quantize_weight_mxfp4(weight)
    assert packed.dtype == torch.uint8
    assert packed.shape == (64, 64)
    assert scales.shape == (64, 4)

    reconstructed = dequant_mxfp4(packed, scales, torch.float32)
    assert reconstructed.shape == weight.shape

    rel_error = (reconstructed - weight).abs() / (weight.abs() + 1e-6)
    assert rel_error.mean() < 0.5


def test_quantize_preserves_zeros():
    from vllm_omni.quantization.mxfp4_config import dequant_mxfp4, quantize_weight_mxfp4

    weight = torch.zeros(32, 64)
    packed, scales = quantize_weight_mxfp4(weight)
    reconstructed = dequant_mxfp4(packed, scales, torch.float32)
    assert torch.allclose(reconstructed, weight, atol=1e-6)


def test_qdq_activation_preserves_shape():
    from vllm_omni.quantization.mxfp4_config import qdq_mxfp4

    x = torch.randn(4, 16, 128, dtype=torch.bfloat16)
    result = qdq_mxfp4(x)
    assert result.shape == x.shape
    assert result.dtype == x.dtype


def test_qdq_activation_handles_non_aligned_dim():
    from vllm_omni.quantization.mxfp4_config import qdq_mxfp4

    x = torch.randn(2, 50)
    result = qdq_mxfp4(x)
    assert result.shape == x.shape


def test_process_weights_after_loading():
    from vllm_omni.quantization.mxfp4_config import DiffusionMxfp4Config, Mxfp4OnlineLinearMethod

    config = DiffusionMxfp4Config()
    method = Mxfp4OnlineLinearMethod(config)

    layer = Module()
    layer.weight = Parameter(torch.randn(64, 128), requires_grad=False)
    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.uint8
    assert layer.weight.shape == (64, 64)
    assert layer.weight_scale.dtype == torch.uint8
    assert layer.weight_scale.shape == (64, 4)


def test_apply_forward():
    from vllm_omni.quantization.mxfp4_config import DiffusionMxfp4Config, Mxfp4OnlineLinearMethod

    config = DiffusionMxfp4Config()
    method = Mxfp4OnlineLinearMethod(config)

    layer = Module()
    layer.weight = Parameter(torch.randn(64, 128), requires_grad=False)
    method.process_weights_after_loading(layer)

    x = torch.randn(2, 8, 128)
    output = method.apply(layer, x, bias=None)
    assert output.shape == (2, 8, 64)


def test_apply_forward_with_bias():
    from vllm_omni.quantization.mxfp4_config import DiffusionMxfp4Config, Mxfp4OnlineLinearMethod

    config = DiffusionMxfp4Config()
    method = Mxfp4OnlineLinearMethod(config)

    layer = Module()
    layer.weight = Parameter(torch.randn(64, 128), requires_grad=False)
    method.process_weights_after_loading(layer)

    output = method.apply(layer, torch.randn(2, 8, 128), bias=torch.randn(64))
    assert output.shape == (2, 8, 64)


def test_apply_bfloat16():
    from vllm_omni.quantization.mxfp4_config import DiffusionMxfp4Config, Mxfp4OnlineLinearMethod

    config = DiffusionMxfp4Config()
    method = Mxfp4OnlineLinearMethod(config)

    layer = Module()
    layer.weight = Parameter(torch.randn(64, 128, dtype=torch.bfloat16), requires_grad=False)
    method.process_weights_after_loading(layer)

    output = method.apply(layer, torch.randn(4, 128, dtype=torch.bfloat16))
    assert output.shape == (4, 64)
    assert output.dtype == torch.bfloat16


def test_dequant_handles_float8_e8m0fnu_scales():
    from vllm_omni.quantization.mxfp4_config import dequant_mxfp4, quantize_weight_mxfp4

    torch.manual_seed(42)
    weight = torch.randn(32, 64)
    packed, scales_uint8 = quantize_weight_mxfp4(weight)

    result_uint8 = dequant_mxfp4(packed, scales_uint8, torch.float32)

    if hasattr(torch, "float8_e8m0fnu"):
        scales_e8m0 = scales_uint8.view(torch.float8_e8m0fnu)
        result_e8m0 = dequant_mxfp4(packed, scales_e8m0, torch.float32)
        assert torch.allclose(result_uint8, result_e8m0)
