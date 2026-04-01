"""Unit tests for vllm_omni.entrypoints.utils module."""

import os
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints.utils import (
    _convert_dataclasses_to_dict,
    _filter_dict_like_object,
    filter_dataclass_kwargs,
    resolve_model_config_path,
)
from vllm_omni.model_executor.models.voxcpm.native_config import ensure_hf_compatible_voxcpm_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestFilterDictLikeObject:
    """Test suite for _filter_dict_like_object function."""

    def test_simple_dict(self):
        """Test filtering a simple dictionary with no callables."""
        input_dict = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        result = _filter_dict_like_object(input_dict)

        assert result == input_dict
        assert isinstance(result, dict)

    def test_dict_with_nested_values(self):
        """Test filtering dict with nested dict and list values."""
        input_dict = {
            "level1": {
                "level2": {"key": "value"},
                "list": [1, 2, 3],
            },
            "simple": "string",
        }

        result = _filter_dict_like_object(input_dict)

        # Nested dicts and lists should be recursively processed
        assert result["simple"] == "string"
        assert isinstance(result["level1"], dict)

    def test_dict_with_dataclass_values(self):
        """Test filtering dict containing dataclass values."""

        @dataclass
        class TestDataclass:
            field1: str
            field2: int

        obj = TestDataclass(field1="test", field2=42)
        input_dict = {"data": obj, "normal": "value"}

        result = _filter_dict_like_object(input_dict)

        # Dataclass should be converted to dict by recursive _convert_dataclasses_to_dict
        assert "data" in result
        assert "normal" in result
        assert result["normal"] == "value"

    def test_dict_with_counter_values(self):
        """Test filtering dict containing Counter objects."""
        counter_obj = Counter({"a": 1, "b": 2})
        input_dict = {"counter": counter_obj, "normal": "value"}

        result = _filter_dict_like_object(input_dict)

        # Counter should be converted to regular dict
        assert "counter" in result
        assert "normal" in result
        assert result["normal"] == "value"

    def test_empty_dict(self):
        """Test filtering an empty dictionary."""
        result = _filter_dict_like_object({})
        assert result == {}
        assert isinstance(result, dict)

    def test_dict_with_set_values(self):
        """Test filtering dict with set values."""
        input_dict = {"set_key": {1, 2, 3}, "normal": "value"}

        result = _filter_dict_like_object(input_dict)

        assert "set_key" in result
        assert "normal" in result
        # Set should be converted to list by _convert_dataclasses_to_dict
        assert result["normal"] == "value"

    def test_dict_with_none_values(self):
        """Test filtering dict with None values."""
        input_dict = {"key1": None, "key2": "value", "key3": 0}

        result = _filter_dict_like_object(input_dict)

        assert result == input_dict

    def test_dict_with_mixed_types(self):
        """Test filtering dict with mixed value types."""
        input_dict = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
            "dict": {"nested": "value"},
        }

        result = _filter_dict_like_object(input_dict)

        assert "string" in result
        assert "int" in result
        assert "float" in result
        assert "bool" in result
        assert "none" in result
        assert "list" in result
        assert "tuple" in result
        assert "set" in result
        assert "dict" in result

    def test_dict_preserves_key_types(self):
        """Test that dict key types are preserved."""
        input_dict = {
            "string_key": "value1",
            42: "value2",
            (1, 2): "value3",  # tuple as key
        }

        result = _filter_dict_like_object(input_dict)

        # Keys should remain the same
        assert "string_key" in result
        assert 42 in result
        assert (1, 2) in result

    def test_dict_with_recursive_structure(self, mocker: MockerFixture):
        """Test filtering dict with recursive/complex nested structure."""
        input_dict = {
            "level1": {
                "level2": {
                    "level3": {"key": "value"},
                    "callable": lambda x: x,
                }
            },
            "normal": "value",
        }

        mocker.patch("vllm_omni.entrypoints.utils.logger")
        result = _filter_dict_like_object(input_dict)

        # Normal key should exist
        assert "normal" in result
        # Level1 should exist
        assert "level1" in result

    def test_integration_with_convert_dataclasses(self, mocker: MockerFixture):
        """Test that _filter_dict_like_object integrates properly with _convert_dataclasses_to_dict."""

        @dataclass
        class Config:
            name: str
            count: int

        input_dict = {
            "config": Config(name="test", count=5),
            "func": lambda x: x,
            "normal": "value",
        }

        mocker.patch("vllm_omni.entrypoints.utils.logger")
        result = _filter_dict_like_object(input_dict)

        # Callable should be filtered
        assert "func" not in result
        # Config should be converted to dict
        assert "config" in result
        assert "normal" in result


class TestConvertDataclassesToDict:
    """Test suite for _convert_dataclasses_to_dict function."""

    def test_uses_filter_dict_like_object(self, mocker: MockerFixture):
        """Test that _convert_dataclasses_to_dict uses _filter_dict_like_object for dicts."""
        input_dict = {
            "normal": "value",
            "callable": lambda x: x,
        }

        mocker.patch("vllm_omni.entrypoints.utils.logger")
        result = _convert_dataclasses_to_dict(input_dict)

        # Callable should be filtered out by _filter_dict_like_object
        assert "normal" in result
        assert "callable" not in result


class TestFilterDataclassKwargs:
    """Test basic functionality of filter_dataclass_kwargs."""

    def test_simple_filtering(self):
        """Test basic dataclass kwargs filtering."""

        @dataclass
        class SimpleConfig:
            name: str
            count: int

        kwargs = {"name": "test", "count": 42, "invalid": "should_be_removed"}
        result = filter_dataclass_kwargs(SimpleConfig, kwargs)

        assert "name" in result
        assert "count" in result
        assert "invalid" not in result

    def test_invalid_dataclass_raises_error(self):
        """Test that non-dataclass raises ValueError."""
        with pytest.raises(ValueError, match="is not a dataclass"):
            filter_dataclass_kwargs(dict, {})

    def test_invalid_kwargs_type_raises_error(self):
        """Test that non-dict kwargs raises ValueError."""

        @dataclass
        class SimpleConfig:
            name: str

        with pytest.raises(ValueError, match="kwargs must be a dictionary"):
            filter_dataclass_kwargs(SimpleConfig, "invalid")

    def test_filters_omni_engine_args_unknown_fields(self):
        """Test that OmniEngineArgs kwargs are filtered to valid fields only."""
        kwargs = {
            "model": "dummy",
            "stage_id": 1,
            "engine_output_type": "image",
            "unknown_field": "drop_me",
        }

        result = filter_dataclass_kwargs(OmniEngineArgs, kwargs)

        assert "model" in result
        assert "stage_id" in result
        assert "engine_output_type" in result
        assert "unknown_field" not in result

    def test_filters_omni_diffusion_config_union_dataclass(self):
        """Test that OmniDiffusionConfig filters nested dataclass in Union fields."""
        kwargs = {
            "model": "dummy",
            "cache_config": {
                "rel_l1_thresh": 0.3,
                "extra_param": "should_drop",
            },
            "unknown_top": "drop_me",
        }

        result = filter_dataclass_kwargs(OmniDiffusionConfig, kwargs)

        assert "model" in result
        assert "cache_config" in result
        assert "unknown_top" not in result
        assert result["cache_config"]["rel_l1_thresh"] == 0.3
        assert "extra_param" not in result["cache_config"]

def _build_native_voxcpm_config() -> dict:
    return {
        "lm_config": {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "vocab_size": 32000,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "max_position_embeddings": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
        },
        "encoder_config": {"hidden_dim": 1024, "ffn_dim": 4096, "num_heads": 16, "num_layers": 4},
        "dit_config": {
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 16,
            "num_layers": 4,
            "cfm_config": {"inference_cfg_rate": 10},
        },
        "patch_size": 2,
        "feat_dim": 64,
    }


class TestVoxCPMNativeConfig:
    def test_ensure_hf_compatible_voxcpm_config(self, tmp_path: Path):
        model_dir = tmp_path / "voxcpm-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(_build_native_voxcpm_config()))

        hf_config_path = ensure_hf_compatible_voxcpm_config(model_dir)

        assert hf_config_path is not None
        rendered_path = Path(hf_config_path) / "config.json"
        assert rendered_path.exists()
        rendered = json.loads(rendered_path.read_text())
        assert rendered["model_type"] == "voxcpm"
        assert rendered["architectures"] == ["VoxCPMForConditionalGeneration"]
        assert rendered["patch_size"] == 2
        assert rendered["feat_dim"] == 64


class TestResolveModelConfigPath:
    def test_resolves_native_voxcpm_to_stage_yaml(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        config_dict = _build_native_voxcpm_config()
        model_dir = tmp_path / "voxcpm-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(config_dict))

        def _raise_get_config(*args, **kwargs):
            raise ValueError("native VoxCPM config is not HF-compatible")

        monkeypatch.setattr("vllm_omni.entrypoints.utils.get_config", _raise_get_config)
        monkeypatch.setattr(
            "vllm_omni.entrypoints.utils.file_or_path_exists",
            lambda model, filename, revision=None: filename == "config.json",
        )
        monkeypatch.setattr(
            "vllm_omni.entrypoints.utils.get_hf_file_to_dict",
            lambda filename, model, revision=None: config_dict,
        )

        config_path = resolve_model_config_path(str(model_dir))

        assert config_path is not None
        assert config_path.endswith("vllm_omni/model_executor/stage_configs/voxcpm.yaml")

    def test_resolves_native_voxcpm_to_npu_stage_yaml(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        config_dict = _build_native_voxcpm_config()
        model_dir = tmp_path / "voxcpm-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(config_dict))

        class _FakePlatform:
            @staticmethod
            def get_default_stage_config_path() -> str:
                return "vllm_omni/platforms/npu/stage_configs"

        def _raise_get_config(*args, **kwargs):
            raise ValueError("native VoxCPM config is not HF-compatible")

        monkeypatch.setattr("vllm_omni.entrypoints.utils.current_omni_platform", _FakePlatform())
        monkeypatch.setattr("vllm_omni.entrypoints.utils.get_config", _raise_get_config)
        monkeypatch.setattr(
            "vllm_omni.entrypoints.utils.file_or_path_exists",
            lambda model, filename, revision=None: filename == "config.json",
        )
        monkeypatch.setattr(
            "vllm_omni.entrypoints.utils.get_hf_file_to_dict",
            lambda filename, model, revision=None: config_dict,
        )

        config_path = resolve_model_config_path(str(model_dir))

        assert config_path is not None
        assert config_path.endswith("vllm_omni/platforms/npu/stage_configs/voxcpm.yaml")

    def test_glm_image_diffusers_format_resolution(self, mocker: MockerFixture):
        """Test GlmImagePipeline diffusers class resolves to glm_image config."""
        mocker.patch(
            "vllm_omni.entrypoints.utils.file_or_path_exists",
            return_value=True,
        )
        mocker.patch(
            "vllm_omni.entrypoints.utils._try_get_class_name_from_diffusers_config",
            return_value="GlmImagePipeline",
        )
        mocker.patch(
            "vllm_omni.entrypoints.utils.current_omni_platform.get_default_stage_config_path",
            return_value="vllm_omni/model_executor/stage_configs",
        )

        original_exists = os.path.exists

        def mock_exists(path):
            if "glm_image.yaml" in str(path):
                return True
            return original_exists(path)

        mocker.patch("os.path.exists", side_effect=mock_exists)

        result = resolve_model_config_path("zai-org/GLM-Image")

        assert result is not None
        assert "glm_image.yaml" in result
