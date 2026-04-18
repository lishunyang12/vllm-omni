# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config

from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.gguf_adapters import get_gguf_adapter

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _DummyPipelineModel(nn.Module):
    def __init__(self, *, source_prefix: str):
        super().__init__()
        self.transformer = nn.Linear(2, 2, bias=False)
        self.vae = nn.Linear(2, 2, bias=False)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path="dummy",
                subfolder="transformer",
                revision=None,
                prefix=source_prefix,
                fall_back_to_pt=True,
            )
        ]

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name not in params:
                continue
            params[name].data.copy_(tensor.to(dtype=params[name].dtype))
            loaded.add(name)
        return loaded


def _make_loader_with_weights(weight_names: list[str]) -> DiffusersPipelineLoader:
    loader = object.__new__(DiffusersPipelineLoader)
    loader.counter_before_loading_weights = 0.0
    loader.counter_after_loading_weights = 0.0

    def _iter_weights(_model):
        for name in weight_names:
            yield name, torch.zeros((2, 2))

    loader.get_all_weights = _iter_weights  # type: ignore[assignment]
    return loader


def test_strict_check_only_validates_source_prefix_parameters():
    model = _DummyPipelineModel(source_prefix="transformer.")
    loader = _make_loader_with_weights(["transformer.weight"])

    # Should not require VAE parameters because they are outside weights_sources.
    loader.load_weights(model)


def test_strict_check_raises_when_source_parameters_are_missing():
    model = _DummyPipelineModel(source_prefix="transformer.")
    loader = _make_loader_with_weights([])

    with pytest.raises(ValueError, match="transformer.weight"):
        loader.load_weights(model)


def test_empty_source_prefix_keeps_full_model_strict_check():
    model = _DummyPipelineModel(source_prefix="")
    loader = _make_loader_with_weights(["transformer.weight"])

    with pytest.raises(ValueError, match="vae.weight"):
        loader.load_weights(model)


def test_qwen_model_class_selects_qwen_gguf_adapter():
    od_config = type(
        "Config",
        (),
        {
            "model_class_name": "QwenImagePipeline",
            "tf_model_config": {"model_type": "qwen_image"},
        },
    )()
    source = DiffusersPipelineLoader.ComponentSource(
        model_or_path="dummy",
        subfolder="transformer",
        revision=None,
        prefix="transformer.",
    )

    adapter = get_gguf_adapter("dummy.gguf", object(), source, od_config)

    assert adapter.__class__.__name__ == "QwenImageGGUFAdapter"


def test_loader_auto_detects_quant_config_from_transformer_config():
    od_config = type(
        "Config",
        (),
        {
            "quantization_config": None,
            "tf_model_config": type(
                "TransformerConfig",
                (),
                {
                    "quant_config": ModelOptFp8Config.from_config(
                        {
                            "quant_method": "modelopt",
                            "quant_algo": "FP8",
                            "ignore": [],
                        }
                    ),
                    "quant_method": "modelopt",
                },
            )(),
            "set_tf_model_config": lambda self, tf_model_config: setattr(
                self,
                "quantization_config",
                tf_model_config.quant_config,
            ),
        },
    )()

    DiffusersPipelineLoader._auto_detect_quant_config(od_config)

    assert od_config.quantization_config is od_config.tf_model_config.quant_config


class _PackedModelOptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.block = nn.Module()
        self.transformer.block.to_qkv = nn.Linear(2, 2, bias=False)


def test_modelopt_adapter_dequantizes_fp8_weight_for_full_precision_target():
    loader = object.__new__(DiffusersPipelineLoader)
    model = _PackedModelOptModel()
    source = DiffusersPipelineLoader.ComponentSource(
        model_or_path="dummy",
        subfolder="transformer",
        revision=None,
        prefix="transformer.",
    )
    fp8_weight = torch.tensor([[2.0, -4.0], [1.0, 3.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], dtype=torch.float32)

    adapted = list(
        loader._adapt_modelopt_fp8_weights(
            model,
            source,
            iter(
                [
                    ("transformer.block.to_q.weight_scale", scale),
                    ("transformer.block.to_q.input_scale", torch.tensor([1.0])),
                    ("transformer.block.to_q.weight", fp8_weight),
                ]
            ),
        )
    )

    assert [name for name, _ in adapted] == ["transformer.block.to_q.weight"]
    assert adapted[0][1].dtype == model.transformer.block.to_qkv.weight.dtype
    assert torch.allclose(adapted[0][1], fp8_weight.to(torch.float32) * scale)


def test_modelopt_adapter_dequantizes_fp8_weight_when_scale_arrives_late():
    loader = object.__new__(DiffusersPipelineLoader)
    model = _PackedModelOptModel()
    source = DiffusersPipelineLoader.ComponentSource(
        model_or_path="dummy",
        subfolder="transformer",
        revision=None,
        prefix="transformer.",
    )
    fp8_weight = torch.tensor([[2.0, -4.0], [1.0, 3.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], dtype=torch.float32)

    adapted = list(
        loader._adapt_modelopt_fp8_weights(
            model,
            source,
            iter(
                [
                    ("transformer.block.to_q.weight", fp8_weight),
                    ("transformer.block.to_q.weight_scale", scale),
                    ("transformer.block.to_q.input_scale", torch.tensor([1.0])),
                ]
            ),
        )
    )

    assert [name for name, _ in adapted] == ["transformer.block.to_q.weight"]
    assert adapted[0][1].dtype == model.transformer.block.to_qkv.weight.dtype
    assert torch.allclose(adapted[0][1], fp8_weight.to(torch.float32) * scale)


class _QuantizedPackedModelOptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.block = nn.Module()
        self.transformer.block.to_qkv = nn.Module()
        self.transformer.block.to_qkv.register_parameter(
            "weight",
            nn.Parameter(torch.empty(2, 2, dtype=torch.float8_e4m3fn), requires_grad=False),
        )
        self.transformer.block.to_qkv.register_parameter(
            "weight_scale",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )
        self.transformer.block.to_qkv.register_parameter(
            "input_scale",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )


class _ChildPackedModelOptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.packed_modules_mapping = {"packed_proj": ["proj_a", "proj_b"]}
        self.transformer.block = nn.Module()
        self.transformer.block.packed_proj = nn.Linear(2, 2, bias=False)


def test_modelopt_adapter_uses_child_packed_modules_mapping():
    loader = object.__new__(DiffusersPipelineLoader)
    model = _ChildPackedModelOptModel()
    source = DiffusersPipelineLoader.ComponentSource(
        model_or_path="dummy",
        subfolder="transformer",
        revision=None,
        prefix="transformer.",
    )
    fp8_weight = torch.tensor([[2.0, -4.0], [1.0, 3.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], dtype=torch.float32)

    adapted = list(
        loader._adapt_modelopt_fp8_weights(
            model,
            source,
            iter(
                [
                    ("transformer.block.proj_a.weight", fp8_weight),
                    ("transformer.block.proj_a.weight_scale", scale),
                ]
            ),
        )
    )

    assert [name for name, _ in adapted] == ["transformer.block.proj_a.weight"]
    assert adapted[0][1].dtype == model.transformer.block.packed_proj.weight.dtype
    assert torch.allclose(adapted[0][1], fp8_weight.to(torch.float32) * scale)


def test_modelopt_adapter_keeps_scale_tensors_for_quantized_target():
    loader = object.__new__(DiffusersPipelineLoader)
    model = _QuantizedPackedModelOptModel()
    source = DiffusersPipelineLoader.ComponentSource(
        model_or_path="dummy",
        subfolder="transformer",
        revision=None,
        prefix="transformer.",
    )
    scale = torch.tensor([0.5], dtype=torch.float32)

    adapted = list(
        loader._adapt_modelopt_fp8_weights(
            model,
            source,
            iter(
                [
                    ("transformer.block.to_q.weight_scale", scale),
                    ("transformer.block.to_q.input_scale", torch.tensor([1.0])),
                ]
            ),
        )
    )

    assert [name for name, _ in adapted] == [
        "transformer.block.to_q.weight_scale",
        "transformer.block.to_q.input_scale",
    ]
