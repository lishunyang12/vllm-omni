# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVFP4 (ModelOpt FP4) quantization support."""

import json
import struct

import pytest
import torch

from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
    Flux2Transformer2DModel,
)
from vllm_omni.diffusion.utils.bfl_mapping import BFL_WEIGHTS_MAPPER as _BFL_WEIGHTS_MAPPER

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# Factory / config tests
# ---------------------------------------------------------------------------


def test_modelopt_fp4_in_supported_methods():
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    assert "modelopt_fp4" in SUPPORTED_QUANTIZATION_METHODS


def test_build_quant_config_modelopt_fp4():
    """vllm_omni.quantization.build_quant_config dispatches "modelopt_fp4"
    through ModelOptNvFp4Config.from_config — mirrors the on-disk flow where
    transformer/config.json carries the quantization_config block."""
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        "modelopt_fp4",
        quant_algo="NVFP4",
        kv_cache_quant_algo=None,
        exclude_modules=["*norm_out*", "*proj_out*"],
        group_size=16,
    )
    assert config.get_name() == "modelopt_fp4"
    assert config.is_checkpoint_nvfp4_serialized
    assert config.group_size == 16
    assert "*norm_out*" in config.exclude_modules


def test_build_quant_config_modelopt_fp4_flux_installs_nibble_swap():
    """The FLUX.2 variant adds a per-instance LinearMethodCls that swaps
    FP4 nibbles before delegating to the upstream process_weights_after_loading.

    We don't run the GPU kernel here — we just confirm that the returned
    config's LinearMethodCls is a subclass of the stock one, so that the
    runtime ``get_quant_method`` path will construct our variant.
    """
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4LinearMethod,
    )

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        "modelopt_fp4_flux",
        quant_algo="NVFP4",
        kv_cache_quant_algo=None,
        exclude_modules=["*norm_out*"],
        group_size=16,
    )
    assert config.is_checkpoint_nvfp4_serialized
    assert issubclass(config.LinearMethodCls, ModelOptNvFp4LinearMethod)
    assert config.LinearMethodCls is not ModelOptNvFp4LinearMethod


def test_modelopt_nvfp4_from_config_dict():
    """Upstream ModelOptNvFp4Config.from_config must accept the dict produced
    by parse_nvfp4_quant_metadata — sanity check for the prep script."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
    )

    quant_dict = {
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "exclude_modules": ["*norm_out*", "*proj_out*"],
            "group_size": 16,
        }
    }
    config = ModelOptNvFp4Config.from_config(quant_dict)
    assert config.get_name() == "modelopt_fp4"
    assert config.is_checkpoint_nvfp4_serialized
    assert config.group_size == 16
    assert "*norm_out*" in config.exclude_modules


# ---------------------------------------------------------------------------
# BFL weight name mapping tests
# ---------------------------------------------------------------------------


class TestBflWeightMapping:
    """Test BFL checkpoint name remapping."""

    def test_double_block_qkv_mapping(self):
        """BFL img_attn.qkv should map to attn.to_qkv."""
        weights = [("double_blocks.0.img_attn.qkv.weight", torch.zeros(1))]
        remapped = list(_BFL_WEIGHTS_MAPPER.apply(weights))
        assert len(remapped) == 1
        name, _ = remapped[0]
        assert "transformer_blocks.0.attn.to_qkv.weight" == name

    def test_double_block_txt_qkv_mapping(self):
        """BFL txt_attn.qkv should map to attn.add_kv_proj."""
        weights = [("double_blocks.3.txt_attn.qkv.weight", torch.zeros(1))]
        remapped = list(_BFL_WEIGHTS_MAPPER.apply(weights))
        name, _ = remapped[0]
        assert "transformer_blocks.3.attn.add_kv_proj.weight" == name

    def test_single_block_linear1_mapping(self):
        """BFL single_blocks.N.linear1 should map to attn.to_qkv_mlp_proj."""
        weights = [("single_blocks.5.linear1.weight", torch.zeros(1))]
        remapped = list(_BFL_WEIGHTS_MAPPER.apply(weights))
        name, _ = remapped[0]
        assert "single_transformer_blocks.5.attn.to_qkv_mlp_proj.weight" == name

    def test_single_block_linear2_mapping(self):
        """BFL single_blocks.N.linear2 should map to attn.to_out."""
        weights = [("single_blocks.10.linear2.weight", torch.zeros(1))]
        remapped = list(_BFL_WEIGHTS_MAPPER.apply(weights))
        name, _ = remapped[0]
        assert "single_transformer_blocks.10.attn.to_out.weight" == name

    def test_embedder_mappings(self):
        """BFL img_in / txt_in should map to x_embedder / context_embedder."""
        weights = [
            ("img_in.weight", torch.zeros(1)),
            ("txt_in.weight", torch.zeros(1)),
        ]
        remapped = dict(_BFL_WEIGHTS_MAPPER.apply(weights))
        assert "x_embedder.weight" in remapped
        assert "context_embedder.weight" in remapped

    def test_time_guidance_embed_mapping(self):
        """BFL time_in / guidance_in should map to time_guidance_embed."""
        weights = [
            ("time_in.in_layer.weight", torch.zeros(1)),
            ("guidance_in.out_layer.weight", torch.zeros(1)),
        ]
        remapped = dict(_BFL_WEIGHTS_MAPPER.apply(weights))
        assert "time_guidance_embed.timestep_embedder.linear_1.weight" in remapped
        assert "time_guidance_embed.guidance_embedder.linear_2.weight" in remapped

    def test_modulation_mapping(self):
        """BFL modulation .lin should map to .linear."""
        weights = [("double_stream_modulation_img.lin.weight", torch.zeros(1))]
        remapped = dict(_BFL_WEIGHTS_MAPPER.apply(weights))
        assert "double_stream_modulation_img.linear.weight" in remapped

    def test_final_layer_mapping(self):
        """BFL final_layer.linear should map to proj_out."""
        weights = [("final_layer.linear.weight", torch.zeros(1))]
        remapped = dict(_BFL_WEIGHTS_MAPPER.apply(weights))
        assert "proj_out.weight" in remapped

    def test_quantization_suffix_preserved(self):
        """Quantization-specific suffixes (.weight_scale etc.) survive remapping."""
        weights = [
            ("double_blocks.0.img_attn.qkv.weight_scale", torch.zeros(1)),
            ("double_blocks.0.img_attn.qkv.input_scale", torch.zeros(1)),
        ]
        remapped = dict(_BFL_WEIGHTS_MAPPER.apply(weights))
        assert "transformer_blocks.0.attn.to_qkv.weight_scale" in remapped
        assert "transformer_blocks.0.attn.to_qkv.input_scale" in remapped


class TestBflFormatDetection:
    """Test BFL checkpoint format auto-detection in load_weights."""

    def test_detects_bfl_format(self):
        weights = iter(
            [
                ("double_blocks.0.img_attn.qkv.weight", torch.zeros(1)),
                ("single_blocks.0.linear1.weight", torch.zeros(1)),
            ]
        )
        is_bfl, chained = Flux2Transformer2DModel._peek_bfl_format(weights)
        assert is_bfl is True
        # The iterator should still yield all original entries.
        assert len(list(chained)) == 2

    def test_detects_diffusers_format(self):
        weights = iter(
            [
                ("transformer_blocks.0.attn.to_q.weight", torch.zeros(1)),
                ("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight", torch.zeros(1)),
            ]
        )
        is_bfl, chained = Flux2Transformer2DModel._peek_bfl_format(weights)
        assert is_bfl is False
        assert len(list(chained)) == 2

    def test_peek_does_not_drain_large_iterator(self):
        """Peek path must only consume a handful of entries, not the whole stream."""

        call_count = {"n": 0}

        def gen():
            for i in range(1000):
                call_count["n"] += 1
                yield (f"double_blocks.{i}.img_attn.qkv.weight", torch.zeros(1))

        is_bfl, _ = Flux2Transformer2DModel._peek_bfl_format(gen())
        assert is_bfl is True
        # Should have only pulled a single entry to decide — certainly fewer than 1000.
        assert call_count["n"] < 10

    def test_bfl_mapping_handles_scale_rename(self):
        """BFL .scale should be renamed to .weight."""
        weights = [("transformer_blocks.0.attn.norm_q.scale", torch.zeros(1))]
        remapped = list(Flux2Transformer2DModel._apply_bfl_mapping(weights))
        name, _ = remapped[0]
        assert name.endswith(".weight")
        assert not name.endswith(".scale")

    def test_bfl_mapping_swaps_adaln_modulation(self):
        """adaLN modulation weight should have shift/scale swapped."""
        # Create [shift(4), scale(4)] tensor
        shift = torch.ones(4)
        scale = torch.ones(4) * 2
        weight = torch.cat([shift, scale], dim=0)

        weights = [("norm_out.linear.weight", weight)]
        remapped = list(Flux2Transformer2DModel._apply_bfl_mapping(weights))
        _, remapped_weight = remapped[0]

        # After swap: [scale(4), shift(4)]
        assert torch.equal(remapped_weight[:4], scale)
        assert torch.equal(remapped_weight[4:], shift)


# ---------------------------------------------------------------------------
# NVFP4 auto-detection utility tests
# ---------------------------------------------------------------------------


class TestNvfp4Detection:
    """Test NVFP4 auto-detection from safetensors files."""

    @staticmethod
    def _create_fake_safetensors(path: str, header: dict) -> None:
        """Create a minimal safetensors file with the given header dict."""
        header_bytes = json.dumps(header).encode("utf-8")
        header_len = len(header_bytes)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_bytes)

    def test_detects_nvfp4_from_uint8_weights(self, tmp_path):
        from vllm_omni.diffusion.utils.nvfp4_utils import detect_nvfp4_from_safetensors

        header = {
            "model.layer.0.weight": {"dtype": "U8", "shape": [128, 64], "data_offsets": [0, 8192]},
            "model.layer.0.weight_scale": {"dtype": "F8_E4M3", "shape": [128, 4], "data_offsets": [8192, 8704]},
        }
        self._create_fake_safetensors(str(tmp_path / "model.safetensors"), header)
        assert detect_nvfp4_from_safetensors(str(tmp_path)) is True

    def test_rejects_non_nvfp4_checkpoint(self, tmp_path):
        from vllm_omni.diffusion.utils.nvfp4_utils import detect_nvfp4_from_safetensors

        header = {
            "model.layer.0.weight": {"dtype": "BF16", "shape": [128, 128], "data_offsets": [0, 32768]},
        }
        self._create_fake_safetensors(str(tmp_path / "model.safetensors"), header)
        assert detect_nvfp4_from_safetensors(str(tmp_path)) is False

    def test_detects_from_quantization_metadata_key(self, tmp_path):
        """FLUX.2-dev-NVFP4 stores metadata under ``_quantization_metadata``."""
        from vllm_omni.diffusion.utils.nvfp4_utils import detect_nvfp4_from_safetensors

        qmeta = {"format_version": "1.0", "layers": {"block.0.qkv": {"format": "nvfp4"}}}
        header = {
            "__metadata__": {"_quantization_metadata": json.dumps(qmeta)},
            "model.layer.0.weight": {"dtype": "BF16", "shape": [128, 128], "data_offsets": [0, 32768]},
        }
        self._create_fake_safetensors(str(tmp_path / "model.safetensors"), header)
        assert detect_nvfp4_from_safetensors(str(tmp_path)) is True

    def test_parse_quant_metadata_builds_modelopt_dict(self, tmp_path):
        from vllm_omni.diffusion.utils.nvfp4_utils import parse_nvfp4_quant_metadata

        qmeta = {
            "format_version": "1.0",
            "layers": {
                "double_blocks.0.img_attn.qkv": {"format": "nvfp4"},
                "single_blocks.0.linear1": {"format": "nvfp4"},
            },
        }
        header = {
            "__metadata__": {"_quantization_metadata": json.dumps(qmeta)},
            "double_blocks.0.img_attn.qkv.weight": {"dtype": "U8", "shape": [1, 1], "data_offsets": [0, 1]},
            "double_blocks.0.img_attn.qkv.weight_scale": {
                "dtype": "F8_E4M3",
                "shape": [1, 1],
                "data_offsets": [1, 2],
            },
        }
        self._create_fake_safetensors(str(tmp_path / "model.safetensors"), header)
        parsed = parse_nvfp4_quant_metadata(str(tmp_path))
        assert parsed is not None
        assert parsed["quantization"]["quant_algo"] == "NVFP4"
        assert parsed["quantization"]["group_size"] == 16
        assert parsed["quantization"]["kv_cache_quant_algo"] is None
        # Exclude list must cover the non-quantized Flux2 layers.
        excl = parsed["quantization"]["exclude_modules"]
        assert any("proj_out" in p for p in excl)
        assert any("add_kv_proj" in p for p in excl)

    def test_resolve_nvfp4_checkpoint_file_prefers_plain(self, tmp_path):
        from vllm_omni.diffusion.utils.nvfp4_utils import resolve_nvfp4_checkpoint_file

        header = {"w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
        self._create_fake_safetensors(str(tmp_path / "flux2-dev-nvfp4.safetensors"), header)
        self._create_fake_safetensors(str(tmp_path / "flux2-dev-nvfp4-mixed.safetensors"), header)

        assert resolve_nvfp4_checkpoint_file(str(tmp_path)) == "flux2-dev-nvfp4.safetensors"
        assert resolve_nvfp4_checkpoint_file(str(tmp_path), prefer_mixed=True) == "flux2-dev-nvfp4-mixed.safetensors"

    def test_returns_false_for_empty_dir(self, tmp_path):
        from vllm_omni.diffusion.utils.nvfp4_utils import detect_nvfp4_from_safetensors

        assert detect_nvfp4_from_safetensors(str(tmp_path)) is False

    def test_returns_false_for_nonexistent_path(self):
        from vllm_omni.diffusion.utils.nvfp4_utils import detect_nvfp4_from_safetensors

        assert detect_nvfp4_from_safetensors("/nonexistent/path") is False
