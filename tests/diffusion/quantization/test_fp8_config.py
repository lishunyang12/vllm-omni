# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FP8 quantization config."""

import pytest


def test_fp8_config_creation():
    """Test that FP8 config can be created."""
    from vllm_omni.diffusion.quantization import get_diffusion_quant_config

    config = get_diffusion_quant_config("fp8")
    assert config is not None
    assert config.get_name() == "fp8"


def test_vllm_config_extraction():
    """Test that vLLM config can be extracted from diffusion config."""
    from vllm_omni.diffusion.quantization import (
        get_diffusion_quant_config,
        get_vllm_quant_config_for_layers,
    )

    diff_config = get_diffusion_quant_config("fp8")
    vllm_config = get_vllm_quant_config_for_layers(diff_config)
    assert vllm_config is not None
    assert vllm_config.activation_scheme == "dynamic"


def test_none_quantization():
    """Test that None quantization returns None config."""
    from vllm_omni.diffusion.quantization import (
        get_diffusion_quant_config,
        get_vllm_quant_config_for_layers,
    )

    config = get_diffusion_quant_config(None)
    assert config is None
    vllm_config = get_vllm_quant_config_for_layers(config)
    assert vllm_config is None


def test_invalid_quantization():
    """Test that invalid quantization method raises error."""
    from vllm_omni.diffusion.quantization import get_diffusion_quant_config

    with pytest.raises(ValueError, match="Unknown quantization method"):
        get_diffusion_quant_config("invalid_method")


def test_fp8_config_with_custom_params():
    """Test FP8 config with custom parameters."""
    from vllm_omni.diffusion.quantization import get_diffusion_quant_config

    config = get_diffusion_quant_config(
        "fp8",
        activation_scheme="static",
        ignored_layers=["proj_out"],
    )
    assert config is not None
    assert config.activation_scheme == "static"
    assert "proj_out" in config.ignored_layers


def test_supported_methods():
    """Test that supported methods list is correct."""
    from vllm_omni.diffusion.quantization import SUPPORTED_QUANTIZATION_METHODS

    assert "fp8" in SUPPORTED_QUANTIZATION_METHODS
