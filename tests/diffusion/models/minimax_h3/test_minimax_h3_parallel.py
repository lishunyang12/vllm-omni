# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_grouped_qkv_checkpoint_reorder():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        _reorder_grouped_qkv_to_qkv,
    )

    # Two groups with rows [q, k, v] become [q0, q1, k0, k1, v0, v1].
    grouped = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    reordered = _reorder_grouped_qkv_to_qkv(
        grouped,
        num_query_groups=2,
        heads_per_group=1,
        head_dim=1,
    )

    assert reordered[:, 0].tolist() == [0, 3, 1, 4, 2, 5]


def test_transformer_declares_cache_sp_layerwise_offload_and_hsdp():
    from cache_dit import ForwardPattern

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    assert MiniMaxH3DiTModel._repeated_blocks == ["MiniMaxH3DiTBlock"]
    assert MiniMaxH3DiTModel._layerwise_offload_blocks_attrs == ["blocks"]
    assert MiniMaxH3DiTModel._cache_dit_adapter_config.block_forward_patterns["blocks"] == ForwardPattern.Pattern_3
    assert not MiniMaxH3DiTModel._cache_dit_adapter_config.has_separate_cfg
    assert set(MiniMaxH3DiTModel._sp_plan) == {"sp_prepare", "sp_gather"}

    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.token_refiner = nn.Module()
    model.token_refiner.blocks = nn.ModuleList([nn.Linear(4, 4)])
    model.final_layer = nn.Linear(4, 4)

    matched = [
        name
        for name, module in model.named_modules()
        if any(condition(name, module) for condition in MiniMaxH3DiTModel._hsdp_shard_conditions)
    ]
    assert matched == ["blocks.0", "blocks.1"]


def test_packed_attention_is_a_regional_compile_boundary():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    assert getattr(MiniMaxH3Attention._run_packed_attention, "_torchdynamo_disable", False)


@pytest.mark.parametrize(
    ("tp_size", "message"),
    [
        (3, "num_attention_heads"),
        (5, "num_attention_heads"),
    ],
)
def test_tp_rejects_non_divisible_head_counts(tp_size, message):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    with pytest.raises(ValueError, match=message):
        model._validate_tp_config(
            arch=MiniMaxH3DiTArchConfig(),
            tp_size=tp_size,
        )


def test_tp_accepts_checkpoint_supported_sizes():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    arch = MiniMaxH3DiTArchConfig()
    for tp_size in (1, 2, 4, 7):
        model._validate_tp_config(arch=arch, tp_size=tp_size)


@pytest.fixture
def model_parallel():
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29519")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
        backend="gloo",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


def test_online_quantization_routes_supported_linears_and_preserves_fp32(model_parallel, monkeypatch):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    import vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer as h3_transformer
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MINIMAX_H3_FP32_PARAM_NAMES,
        MiniMaxH3DiTModel,
    )
    from vllm_omni.quantization.component_config import (
        ComponentQuantizationConfig,
    )

    class FakeAttention(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            del kwargs

    monkeypatch.setattr(h3_transformer, "Attention", FakeAttention)

    arch = {
        "num_layers": 1,
        "token_refiner_num_layers": 1,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "attention_head_dim": 16,
        "ffn_hidden_size": 128,
        "latents_dim": 4,
        "audio_latents_dim": 4,
        "patch_size": [1, 2, 2],
        "text_dim": 32,
        "timestep_input_dim": 16,
        "time_embed_hidden_size": 64,
        "time_embed_dim": 32,
        "adaln_out_features": 18 * 64,
        "final_adaln_out_features": 2 * 64,
        "rope_inv_freq_len": 4,
    }
    od_config = SimpleNamespace(
        tf_model_config=arch,
        parallel_config=SimpleNamespace(ulysses_degree=1),
    )
    transformer_quant_config = Mock()
    transformer_quant_config.weight_block_size = None
    transformer_quant_config.get_quant_method.return_value = UnquantizedLinearMethod()
    quant_config = ComponentQuantizationConfig(
        {"transformer": transformer_quant_config},
    )

    model = MiniMaxH3DiTModel(
        od_config,
        quant_config=quant_config,
        prefix="transformer",
    )

    quantized_prefixes = [
        call.args[1]
        for call in transformer_quant_config.get_quant_method.call_args_list
    ]
    assert quantized_prefixes == [
        "transformer.condition_proj",
        "transformer.token_refiner.blocks.0.attn.qkv_proj",
        "transformer.token_refiner.blocks.0.attn.out_proj",
        "transformer.token_refiner.blocks.0.mlp.fc1",
        "transformer.token_refiner.blocks.0.mlp.fc2",
        "transformer.blocks.0.attn.qkv_proj",
        "transformer.blocks.0.attn.out_proj",
        "transformer.blocks.0.mlp.fc1",
        "transformer.blocks.0.mlp.fc2",
        "transformer.blocks.0.adaln_proj.linear",
        "transformer.final_layer.adaln_proj.linear",
    ]

    params = dict(model.named_parameters())
    assert MINIMAX_H3_FP32_PARAM_NAMES <= params.keys()
    assert all(params[name].dtype == torch.float32 for name in MINIMAX_H3_FP32_PARAM_NAMES)


def test_adaln_keeps_bf16_activations_with_fp8_weights():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3AdalnProj,
    )

    class CaptureLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(1, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.input_dtype = None

        def forward(self, x):
            self.input_dtype = x.dtype
            return torch.zeros((x.shape[0], 8), dtype=x.dtype), None

    proj = object.__new__(MiniMaxH3AdalnProj)
    nn.Module.__init__(proj)
    proj.expand_ratio = 2
    proj.modality_num = 1
    proj.hidden_size = 4
    proj.linear = CaptureLinear()

    outputs = proj(torch.randn(1, 4, dtype=torch.float32))

    assert proj.linear.input_dtype == torch.bfloat16
    assert len(outputs) == 2
    assert all(output.dtype == torch.bfloat16 for output in outputs)
