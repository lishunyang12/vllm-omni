# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for LayerwiseOffloadHook and LayerWiseOffloadBackend utilities."""

import gc
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate

import vllm_omni.diffusion.offloader.config as offload_config_module
import vllm_omni.diffusion.offloader.layerwise_backend as layerwise_backend_module
from tests.diffusion.offloader.helpers import _DummyBlock, _PlainEncoder, _SingleBlockModel, _StagedEncoder, _StagedVAE
from tests.helpers.runtime import get_distributed_init_method
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.layerwise_backend import (
    LayerWiseOffloadBackend,
    LayerwiseOffloadHook,
)
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class DummyStream:
    def wait_stream(self, _stream) -> None:
        return None

    def wait_event(self, _event) -> None:
        return None


class DummyEvent:
    def record(self, _stream) -> None:
        return None


@contextmanager
def dummy_stream(_stream):
    yield None


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

    gc.collect()
    if current_omni_platform.is_available():
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()


@pytest.fixture(scope="module")
def dist_group():
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=get_distributed_init_method())
    try:
        yield
    finally:
        _cleanup_distributed()


@pytest.fixture
def patched_offload_runtime(monkeypatch):
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "Stream", DummyStream)
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "Event", DummyEvent)
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "current_stream", lambda: DummyStream())
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "stream", dummy_stream)


class TinyBlock(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        mesh = DeviceMesh("cpu", [0])
        dtensor = DTensor.from_local(values, mesh, [Replicate()])
        self.weight = nn.Parameter(dtensor)


def _make_values(start: float) -> torch.Tensor:
    return torch.arange(start, start + 4, dtype=torch.float32)


class TestLayerwiseOffloadHook:
    def test_dtensor_wrapper_is_preserved_across_prefetch_and_offload(self, dist_group, patched_offload_runtime):
        current_block = TinyBlock(_make_values(1.0))
        next_block = TinyBlock(_make_values(10.0))

        hook = LayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            stream=DummyStream(),
            pin_memory=False,
        )

        hook.initialize_hook(current_block)

        assert isinstance(next_block.weight, DTensor)
        assert next_block.weight.to_local().is_meta
        assert next_block.weight.to_local().shape == torch.Size([4])
        assert hook.dtype_metadata[next_block.weight.dtype][0]["shape"] == torch.Size([4])

        hook.prefetch_layer(non_blocking=False)
        assert isinstance(next_block.weight, DTensor)
        assert torch.equal(next_block.weight.to_local(), _make_values(10.0))
        assert next_block.weight.to_local().shape == torch.Size([4])

        hook.offload_layer()
        assert isinstance(current_block.weight, DTensor)
        assert current_block.weight.to_local().is_meta
        assert current_block.weight.to_local().shape == torch.Size([4])
        assert not hook.is_materialized

    def test_prefetch_preserves_transposed_weight_stride(self, patched_offload_runtime):
        """Online-FP8 Cutlass weights must retain their transposed layout."""

        class StridedBlock(nn.Module):
            def __init__(self, start: float):
                super().__init__()
                base = torch.arange(start, start + 12).reshape(3, 4)
                self.weight = nn.Parameter(base.t(), requires_grad=False)

        current_block = StridedBlock(1.0)
        next_block = StridedBlock(20.0)
        expected = next_block.weight.detach().clone()
        expected_stride = next_block.weight.stride()
        assert expected_stride == (1, 4)

        hook = LayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            stream=DummyStream(),
            pin_memory=False,
        )
        hook.initialize_hook(current_block)
        hook.prefetch_layer(non_blocking=False)

        assert next_block.weight.stride() == expected_stride
        assert torch.equal(next_block.weight, expected)


class _MultiBlockModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]

    def __init__(self, num_transformer: int = 2, num_single: int = 2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_DummyBlock() for _ in range(num_transformer)])
        self.single_transformer_blocks = nn.ModuleList([_DummyBlock() for _ in range(num_single)])


class _EmptyBlocksModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([])


class _InvalidAttrModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["nonexistent_blocks", "blocks"]

    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class _DeprecatedSingleAttrModel(nn.Module):
    _layerwise_offload_blocks_attr = "blocks"

    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class _NoAttrsModel(nn.Module):
    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class TestGetBlocksFromDit:
    def test_get_blocks_from_dit_single_block_attr(self):
        model = _SingleBlockModel(num_blocks=3)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == ["blocks"]
        assert len(blocks) == 3
        assert all(isinstance(b, _DummyBlock) for b in blocks)

    def test_get_blocks_from_dit_multi_block_attrs(self):
        model = _MultiBlockModel(num_transformer=2, num_single=3)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert set(attr_names) == {"transformer_blocks", "single_transformer_blocks"}
        assert len(blocks) == 5
        assert all(isinstance(b, _DummyBlock) for b in blocks)

    def test_get_blocks_from_dit_empty_blocks(self):
        model = _EmptyBlocksModel()
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == []
        assert blocks == []

    def test_get_blocks_from_dit_invalid_attr_name(self):
        model = _InvalidAttrModel(num_blocks=2)
        with pytest.raises(
            AttributeError,
            match="Attribute 'nonexistent_blocks' declared in _layerwise_offload_blocks_attrs does not exist",
        ):
            LayerWiseOffloadBackend.get_blocks_from_dit(model)

    def test_get_blocks_from_dit_no_attrs_defined(self):
        model = _NoAttrsModel(num_blocks=3)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == []
        assert blocks == []

    def test_get_blocks_from_dit_deprecated_single_attr(self):
        model = _DeprecatedSingleAttrModel(num_blocks=2)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == ["blocks"]
        assert len(blocks) == 2


class TestGetBlocksAttrNames:
    def test_get_blocks_attr_names_new_format(self):
        model = _MultiBlockModel()
        attrs = LayerWiseOffloadBackend.get_blocks_attr_names(model)
        assert attrs == ["transformer_blocks", "single_transformer_blocks"]

    def test_get_blocks_attr_names_no_attrs(self):
        model = _NoAttrsModel()
        attrs = LayerWiseOffloadBackend.get_blocks_attr_names(model)
        assert attrs == []

    def test_set_blocks_attr_names(self):
        model = _NoAttrsModel()
        LayerWiseOffloadBackend.set_blocks_attr_names(model, ["new_blocks"])
        assert hasattr(model.__class__, "_layerwise_offload_blocks_attrs")
        assert model.__class__._layerwise_offload_blocks_attrs == ["new_blocks"]


class _ComponentPipeline(nn.Module):
    _offload_plan = OffloadPlan(
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("vision.blocks", "text_model.layers")},
        on_demand_component_paths=frozenset({"text_encoder", "vae"}),
    )

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel()
        self.text_encoder = _StagedEncoder()
        self.vae = _StagedVAE()


class _LegacyComponentPipeline(nn.Module):
    """Pipeline with DiT block metadata but no auxiliary OffloadPlan."""

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel()
        self.text_encoder = _StagedEncoder()
        self.vae = _StagedVAE()


class _LegacyEncoderOnlyPipeline(nn.Module):
    _offload_plan = _ComponentPipeline._offload_plan

    def __init__(self):
        super().__init__()
        self.text_encoder = _StagedEncoder()
        self.vae = _StagedVAE()


class _GenericEncoderPipeline(nn.Module):
    _offload_plan = OffloadPlan(
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("encoder.block",)},
    )

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel()
        self.text_encoder = _PlainEncoder()


class TestLayerwiseComponentSelection:
    def test_legacy_missing_dit_preserves_noop(self, patched_offload_runtime):
        pipeline = _LegacyEncoderOnlyPipeline()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        assert not backend.enabled
        assert pipeline.text_encoder.offload_calls == 0
        assert pipeline.vae.offload_calls == 0

    def test_encoder_only_streams_planned_blocks(self, patched_offload_runtime):
        pipeline = _ComponentPipeline()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"text_encoder"}),
                components_explicit=True,
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        assert pipeline.text_encoder._omni_layerwise_enabled
        assert len(pipeline.text_encoder._omni_layerwise_hooks) == 4
        assert pipeline.text_encoder.offload_calls == 1
        assert pipeline.vae.offload_calls == 0
        assert pipeline.vae.to_calls == 1
        assert not hasattr(pipeline.transformer.blocks[0], "_hook_registry")
        assert backend.enabled

        backend.disable()
        assert not pipeline.text_encoder._omni_layerwise_enabled

    def test_shutdown_skips_encoder_weight_reconstruction(self, patched_offload_runtime, mocker):
        pipeline = _ComponentPipeline()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"text_encoder"}),
                components_explicit=True,
            ),
            torch.device("cpu"),
        )
        backend.enable(pipeline)
        for hook in pipeline.text_encoder._omni_layerwise_hooks:
            hook.prefetch_layer = mocker.Mock(side_effect=AssertionError("shutdown rebuilt encoder weights"))

        backend.shutdown()

        assert not pipeline.text_encoder._omni_layerwise_enabled
        assert not backend.enabled

    def test_single_gpu_dit_only_keeps_encoder_and_vae_resident(self, patched_offload_runtime):
        pipeline = _ComponentPipeline()
        expected_weights = [block.weight.detach().clone() for block in pipeline.transformer.blocks]
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"dit"}),
                components_explicit=True,
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        assert not hasattr(pipeline.text_encoder, "_omni_layerwise_enabled")
        assert pipeline.text_encoder.to_calls == 1
        assert pipeline.text_encoder.offload_calls == 0
        assert pipeline.vae.to_calls == 1
        assert pipeline.vae.offload_calls == 0
        assert hasattr(pipeline.transformer.blocks[0], "_hook_registry")

        backend.disable()

        for block, expected in zip(pipeline.transformer.blocks, expected_weights, strict=True):
            torch.testing.assert_close(block.weight, expected)

        backend.enable(pipeline)
        backend.disable()
        for block, expected in zip(pipeline.transformer.blocks, expected_weights, strict=True):
            torch.testing.assert_close(block.weight, expected)

    def test_partial_dit_enable_failure_restores_weights_and_hooks(self, patched_offload_runtime, monkeypatch):
        pipeline = _ComponentPipeline()
        expected_weights = [block.weight.detach().clone() for block in pipeline.transformer.blocks]
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"dit"}),
                components_explicit=True,
            ),
            torch.device("cpu"),
        )
        original_apply = layerwise_backend_module.apply_block_hook
        calls = 0

        def fail_second_hook(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected hook failure")
            return original_apply(*args, **kwargs)

        monkeypatch.setattr(layerwise_backend_module, "apply_block_hook", fail_second_hook)

        with pytest.raises(RuntimeError, match="injected hook failure"):
            backend.enable(pipeline)

        assert not backend.enabled
        for block, expected in zip(pipeline.transformer.blocks, expected_weights, strict=True):
            registry = getattr(block, "_hook_registry", None)
            assert registry is None or registry.get_hook("layerwise_offload") is None
            torch.testing.assert_close(block.weight, expected)

    def test_encoder_only_requires_streamable_offload_plan(self, patched_offload_runtime):
        pipeline = nn.Module()
        pipeline.transformer = _SingleBlockModel()
        pipeline.text_encoder = _StagedEncoder()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"text_encoder"}),
                components_explicit=True,
            ),
            torch.device("cpu"),
        )

        with pytest.raises(ValueError, match="Selected text_encoder layerwise offload requires"):
            backend.enable(pipeline)

    def test_default_selection_preserves_unplanned_auxiliaries(self, patched_offload_runtime):
        pipeline = _LegacyComponentPipeline()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        assert hasattr(pipeline.transformer.blocks[0], "_hook_registry")
        assert not hasattr(pipeline.text_encoder, "_omni_layerwise_enabled")
        assert pipeline.text_encoder.to_calls == 1
        assert pipeline.text_encoder.offload_calls == 0
        assert pipeline.vae.to_calls == 1
        assert pipeline.vae.offload_calls == 0

        backend.disable()

    def test_legacy_selector_preserves_planned_encoder_and_vae_lifecycle(self, patched_offload_runtime):
        pipeline = _ComponentPipeline()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        assert pipeline.text_encoder._omni_layerwise_enabled
        assert pipeline.text_encoder.offload_calls == 1
        assert pipeline.vae.offload_calls == 1

        backend.disable()

    def test_standard_encoder_needs_only_declared_block_paths(self, patched_offload_runtime):
        pipeline = _GenericEncoderPipeline()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"text_encoder"}),
                components_explicit=True,
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        blocks = pipeline.text_encoder.encoder.block
        assert pipeline.text_encoder._omni_layerwise_enabled
        assert all(block._hook_registry.get_hook("layerwise_offload") is not None for block in blocks)
        assert pipeline.text_encoder.final_norm.weight.numel() == 4

        backend.disable()

        assert not pipeline.text_encoder._omni_layerwise_enabled
        assert all(block._hook_registry.get_hook("layerwise_offload") is None for block in blocks)
        assert all(block.weight.numel() == 100 for block in blocks)


def _offload_od_config(**overrides):
    values = {
        "diffusion_offload_config": None,
        "enable_cpu_offload": False,
        "enable_layerwise_offload": False,
        "enable_distributed_layerwise_offload": False,
        "dlo_use_allgather": True,
        "dlo_resident_layers": 0,
        "pin_cpu_memory": True,
        "parallel_config": None,
        "model": "/fake/model",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestLayerwiseComponentConfig:
    def test_public_config_is_parsed_once_per_config_object(self, monkeypatch):
        od_config = _offload_od_config(
            diffusion_offload_config={
                "mode": "layer",
                "components": ["dit", "text_encoder"],
            }
        )
        parse = offload_config_module.parse_diffusion_offload_config
        calls = 0

        def counted_parse(value):
            nonlocal calls
            calls += 1
            return parse(value)

        monkeypatch.setattr(offload_config_module, "parse_diffusion_offload_config", counted_parse)

        assert offload_config_module.selected_offload_components(od_config) == {"dit", "text_encoder"}
        assert not offload_config_module.component_uses_allgather(od_config, "dit")
        assert offload_config_module.resolve_offload_strategy(od_config) is OffloadStrategy.LAYER_WISE
        assert calls == 1

    def test_replacing_public_config_invalidates_parsed_cache(self):
        od_config = _offload_od_config(
            diffusion_offload_config={
                "mode": "layer",
                "components": ["dit"],
            }
        )

        first = offload_config_module.get_diffusion_offload_config(od_config)
        od_config.diffusion_offload_config = {
            "mode": "layer",
            "components": ["text_encoder"],
        }
        second = offload_config_module.get_diffusion_offload_config(od_config)

        assert first is not second
        assert second is not None
        assert second.components == {"text_encoder"}

    def test_layer_mode_selects_components_without_backend_jargon(self):
        config = OffloadConfig.from_od_config(
            _offload_od_config(
                diffusion_offload_config={
                    "mode": "layer",
                    "components": ["dit", "text_encoder"],
                }
            )
        )

        assert config.strategy is OffloadStrategy.LAYER_WISE
        assert config.components == frozenset({"dit", "text_encoder"})
        assert not config.uses_allgather("dit")
        assert not config.uses_allgather("text_encoder")
        plan = OffloadPlan(encoder_component_types={"mllm": "text_encoder"})
        assert config.offloads_encoder("mllm", plan)

    @pytest.mark.parametrize("component", ["image_encoder", "vae", "scheduler", "text-encoder"])
    def test_unknown_or_noncanonical_component_is_rejected(self, component):
        with pytest.raises(ValueError, match="Unknown diffusion offload component"):
            OffloadConfig.from_od_config(
                _offload_od_config(diffusion_offload_config={"mode": "layer", "components": [component]})
            )

    def test_components_is_selection_only_not_an_options_mapping(self):
        with pytest.raises(TypeError, match="components must be a non-empty list"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={
                        "mode": "layer",
                        "components": {"dit": {}},
                    }
                )
            )

    def test_layer_options_requires_component_name_keys(self):
        with pytest.raises(TypeError, match="layer_options keys must be component names"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={
                        "mode": "layer",
                        "components": ["dit"],
                        "layer_options": {0: {}},
                    }
                )
            )

    @pytest.mark.parametrize(
        "config,match",
        [
            ({"components": ["dit"]}, "requires 'mode'"),
            ({"mode": "layer"}, "requires 'components'"),
            ({"mode": "layer", "components": []}, "must not be empty"),
            ({"mode": "layerwise", "components": ["dit"]}, "Unknown diffusion offload mode"),
            (
                {
                    "mode": "layer",
                    "components": ["dit"],
                    "layer_options": {"dit": {"weight_transfer": "rank_local"}},
                },
                "Unknown offload transfer",
            ),
            (
                {
                    "mode": "layer",
                    "components": ["dit"],
                    "layer_options": {"dit": {"prefetch": 2}},
                },
                "Unknown diffusion offload setting",
            ),
            (
                {
                    "mode": "layer",
                    "components": ["text_encoder"],
                    "layer_options": {"text_encoder": {"resident_layers": 1}},
                },
                "supports only the 'dit'",
            ),
            (
                {
                    "mode": "layer",
                    "components": ["dit"],
                    "layer_options": {"text_encoder": {}},
                },
                "requires selecting the same component",
            ),
        ],
    )
    def test_schema_rejects_ambiguous_or_unsupported_values(self, config, match):
        with pytest.raises(ValueError, match=match):
            OffloadConfig.from_od_config(_offload_od_config(diffusion_offload_config=config))

    def test_module_mode_accepts_component_selection(self):
        config = OffloadConfig.from_od_config(
            _offload_od_config(
                diffusion_offload_config={
                    "mode": "module",
                    "components": ["dit", "text_encoder"],
                }
            )
        )

        assert config.strategy is OffloadStrategy.MODEL_LEVEL
        assert config.components == frozenset({"dit", "text_encoder"})
        assert config.components_explicit is True

    @pytest.mark.parametrize("setting", [{"weight_transfer": "rank-local"}, {"resident_layers": 1}])
    def test_module_mode_rejects_layer_settings(self, setting):
        with pytest.raises(ValueError, match="layer_options requires mode='layer'"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={
                        "mode": "module",
                        "components": ["dit"],
                        "layer_options": {"dit": setting},
                    }
                )
            )

    def test_each_component_can_choose_its_transfer(self):
        config = OffloadConfig.from_od_config(
            _offload_od_config(
                diffusion_offload_config={
                    "mode": "layer",
                    "components": ["dit", "text_encoder"],
                    "layer_options": {
                        "dit": {"weight_transfer": "rank-local"},
                        "text_encoder": {"weight_transfer": "allgather"},
                    },
                }
            )
        )

        assert config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
        assert not config.uses_allgather("dit")
        assert config.uses_allgather("text_encoder")
        assert config.dlo_use_allgather is False

    def test_dit_residency_selects_capable_backend(self):
        config = OffloadConfig.from_od_config(
            _offload_od_config(
                diffusion_offload_config={
                    "mode": "layer",
                    "components": ["dit"],
                    "layer_options": {"dit": {"resident_layers": 20}},
                }
            )
        )

        assert config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
        assert config.dlo_resident_layers == 20
        assert not config.uses_allgather("dit")

    def test_dit_residency_rejects_allgather(self):
        with pytest.raises(ValueError, match="resident_layers requires dit.weight_transfer='rank-local'"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={
                        "mode": "layer",
                        "components": ["dit"],
                        "layer_options": {"dit": {"weight_transfer": "allgather", "resident_layers": 1}},
                    }
                )
            )

    @pytest.mark.parametrize(
        "offload_config",
        [
            {"mode": "module", "components": ["dit"]},
            {"mode": "layer", "components": ["text_encoder"]},
            {
                "mode": "layer",
                "components": ["dit"],
                "layer_options": {"dit": {"weight_transfer": "allgather"}},
            },
        ],
    )
    def test_host_weight_runtime_rejects_ineligible_offload_config(self, offload_config):
        with pytest.raises(ValueError, match="Host Weight Runtime requires"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config=offload_config,
                    host_weight_runtime_mode="preferred",
                )
            )

    def test_host_weight_runtime_selects_rank_local_dlo_backend(self):
        config = OffloadConfig.from_od_config(
            _offload_od_config(
                diffusion_offload_config={
                    "mode": "layer",
                    "components": ["dit"],
                    "layer_options": {"dit": {"weight_transfer": "rank-local"}},
                },
                host_weight_runtime_mode="preferred",
            )
        )

        assert config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
        assert not config.uses_allgather("dit")

    def test_pin_memory_override_is_scoped_to_offload_config(self):
        config = OffloadConfig.from_od_config(
            _offload_od_config(
                pin_cpu_memory=True,
                diffusion_offload_config={
                    "mode": "layer",
                    "components": ["dit"],
                    "pin_memory": False,
                },
            )
        )

        assert config.pin_cpu_memory is False

    def test_legacy_omitted_selector_preserves_dit_only_behavior(self):
        config = OffloadConfig.from_od_config(_offload_od_config(enable_layerwise_offload=True))

        assert config.strategy is OffloadStrategy.LAYER_WISE
        assert config.components == frozenset({"dit"})

    def test_legacy_distributed_settings_still_work(self):
        config = OffloadConfig.from_od_config(
            _offload_od_config(
                enable_distributed_layerwise_offload=True,
                dlo_use_allgather=False,
                dlo_resident_layers=3,
            )
        )

        assert config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
        assert config.dlo_resident_layers == 3
        assert not config.uses_allgather("dit")
        assert not config.uses_allgather("text_encoder")

    def test_legacy_allgather_remains_dit_only(self):
        config = OffloadConfig.from_od_config(
            _offload_od_config(
                enable_distributed_layerwise_offload=True,
                dlo_use_allgather=True,
            )
        )

        assert config.uses_allgather("dit")
        assert not config.uses_allgather("text_encoder")

    @pytest.mark.parametrize(
        ("legacy_flags", "expected_strategy"),
        [
            (
                {"enable_cpu_offload": True, "enable_layerwise_offload": True},
                OffloadStrategy.LAYER_WISE,
            ),
            (
                {
                    "enable_cpu_offload": True,
                    "enable_layerwise_offload": True,
                    "enable_distributed_layerwise_offload": True,
                },
                OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            ),
        ],
    )
    def test_legacy_aliases_preserve_existing_priority(self, legacy_flags, expected_strategy):
        config = OffloadConfig.from_od_config(_offload_od_config(**legacy_flags))

        assert config.strategy is expected_strategy

    def test_compact_config_cannot_mix_with_legacy_alias(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={"mode": "layer", "components": ["dit"]},
                    enable_cpu_offload=True,
                )
            )

    @pytest.mark.parametrize(
        "legacy_options",
        [
            {"dlo_use_allgather": False},
            {"dlo_resident_layers": 12},
        ],
    )
    def test_compact_config_cannot_mix_with_legacy_dlo_options(self, legacy_options):
        with pytest.raises(ValueError, match="cannot be combined with legacy DLO option"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    diffusion_offload_config={"mode": "layer", "components": ["dit"]},
                    **legacy_options,
                )
            )

    def test_legacy_allgather_resident_layers_fail_during_config_resolution(self):
        with pytest.raises(ValueError, match="requires the DiT DLO transfer to be rank-local"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    enable_distributed_layerwise_offload=True,
                    dlo_use_allgather=True,
                    dlo_resident_layers=12,
                )
            )
