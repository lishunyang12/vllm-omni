# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for StageConfigFactory and related classes.
"""

import pytest

from vllm_omni.config.stage_config import (
    StageConfig,
    StageConfigFactory,
    StageTopology,
    StageType,
)


class TestStageType:
    """Tests for StageType enum."""

    def test_stage_type_values(self):
        """Test StageType enum values."""
        assert StageType.LLM.value == "llm"
        assert StageType.DIFFUSION.value == "diffusion"

    def test_stage_type_from_string(self):
        """Test creating StageType from string."""
        assert StageType("llm") == StageType.LLM
        assert StageType("diffusion") == StageType.DIFFUSION


class TestStageConfig:
    """Tests for StageConfig dataclass."""

    def test_minimal_config(self):
        """Test creating StageConfig with minimal required fields."""
        config = StageConfig(stage_id=0, model_stage="thinker")
        assert config.stage_id == 0
        assert config.model_stage == "thinker"
        assert config.stage_type == StageType.LLM
        assert config.input_sources == []
        assert config.final_output is False
        assert config.worker_type is None

    def test_full_config(self):
        """Test creating StageConfig with all fields."""
        config = StageConfig(
            stage_id=1,
            model_stage="talker",
            stage_type=StageType.LLM,
            input_sources=[0],
            custom_process_input_func="module.path.func",
            final_output=True,
            final_output_type="audio",
            worker_type="ar",
            scheduler_cls="path.to.Scheduler",
            hf_config_name="talker_config",
            is_comprehension=False,
        )
        assert config.stage_id == 1
        assert config.model_stage == "talker"
        assert config.input_sources == [0]
        assert config.final_output_type == "audio"
        assert config.worker_type == "ar"

    def test_to_omegaconf_basic(self):
        """Test converting StageConfig to OmegaConf format."""
        config = StageConfig(
            stage_id=0,
            model_stage="thinker",
            stage_type=StageType.LLM,
            worker_type="ar",
            final_output=True,
            final_output_type="text",
        )
        omega_config = config.to_omegaconf()

        assert omega_config.stage_id == 0
        assert omega_config.stage_type == "llm"
        assert omega_config.engine_args.model_stage == "thinker"
        assert omega_config.engine_args.worker_type == "ar"
        assert omega_config.final_output is True
        assert omega_config.final_output_type == "text"
        # Legacy field name for backward compatibility
        assert omega_config.engine_input_source == []

    def test_to_omegaconf_with_runtime_overrides(self):
        """Test that runtime overrides are applied to OmegaConf output."""
        config = StageConfig(
            stage_id=0,
            model_stage="thinker",
            runtime_overrides={
                "gpu_memory_utilization": 0.9,
                "tensor_parallel_size": 2,
                "devices": "0,1",
                "max_batch_size": 64,
            },
        )
        omega_config = config.to_omegaconf()

        assert omega_config.engine_args.gpu_memory_utilization == 0.9
        assert omega_config.engine_args.tensor_parallel_size == 2
        assert omega_config.runtime.devices == "0,1"
        assert omega_config.runtime.max_batch_size == 64


class TestStageTopology:
    """Tests for StageTopology class."""

    def test_valid_linear_dag(self):
        """Test validation of a valid linear DAG."""
        stages = [
            StageConfig(stage_id=0, model_stage="thinker", input_sources=[]),
            StageConfig(stage_id=1, model_stage="talker", input_sources=[0]),
            StageConfig(stage_id=2, model_stage="code2wav", input_sources=[1]),
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_dag()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_valid_branching_dag(self):
        """Test validation of a valid branching DAG."""
        stages = [
            StageConfig(stage_id=0, model_stage="input", input_sources=[]),
            StageConfig(stage_id=1, model_stage="branch_a", input_sources=[0]),
            StageConfig(stage_id=2, model_stage="branch_b", input_sources=[0]),
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_dag()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_entry_point(self):
        """Test that missing entry point is detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="stage_a", input_sources=[1]),
            StageConfig(stage_id=1, model_stage="stage_b", input_sources=[0]),
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_dag()
        assert any("entry point" in e.lower() for e in errors)

    def test_missing_dependency(self):
        """Test that missing stage reference is detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="input", input_sources=[]),
            StageConfig(stage_id=1, model_stage="output", input_sources=[99]),  # Invalid
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_dag()
        assert any("non-existent" in e.lower() for e in errors)

    def test_duplicate_stage_ids(self):
        """Test that duplicate stage IDs are detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="stage_a", input_sources=[]),
            StageConfig(stage_id=0, model_stage="stage_b", input_sources=[]),  # Duplicate
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_dag()
        assert any("duplicate" in e.lower() for e in errors)

    def test_cycle_detection_simple(self):
        """Test that simple cycles are detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="stage_a", input_sources=[1]),
            StageConfig(stage_id=1, model_stage="stage_b", input_sources=[0]),
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_dag()
        # Should detect both missing entry point and cycle
        assert len(errors) >= 1

    def test_self_reference(self):
        """Test that self-references are detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="entry", input_sources=[]),
            StageConfig(stage_id=1, model_stage="self_ref", input_sources=[1]),  # Self
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_dag()
        assert any("itself" in e.lower() for e in errors)

    def test_get_stage_by_id(self):
        """Test getting stage by ID."""
        stages = [
            StageConfig(stage_id=0, model_stage="thinker", input_sources=[]),
            StageConfig(stage_id=1, model_stage="talker", input_sources=[0]),
        ]
        topology = StageTopology(model_type="test", stages=stages)

        stage = topology.get_stage(1)
        assert stage is not None
        assert stage.model_stage == "talker"

        missing = topology.get_stage(99)
        assert missing is None

    def test_empty_topology(self):
        """Test validation of empty topology."""
        topology = StageTopology(model_type="test", stages=[])
        errors = topology.validate_dag()
        assert any("no stages" in e.lower() for e in errors)


class TestStageConfigFactory:
    """Tests for StageConfigFactory class."""

    def test_default_diffusion_no_yaml(self):
        """Test single-stage diffusion works without YAML config (@ZJY0516)."""
        kwargs = {
            "cache_backend": "none",
            "cache_config": None,
            "dtype": "bfloat16",
        }
        configs = StageConfigFactory.create_default_diffusion(kwargs)

        assert len(configs) == 1
        cfg = configs[0]
        assert cfg["stage_id"] == 0
        assert cfg["stage_type"] == "diffusion"
        assert cfg["final_output"] is True
        assert cfg["final_output_type"] == "image"

    def test_default_diffusion_with_parallel_config(self):
        """Test diffusion config calculates devices from parallel_config."""

        class MockParallelConfig:
            world_size = 4

        kwargs = {
            "parallel_config": MockParallelConfig(),
            "cache_backend": "tea_cache",
        }
        configs = StageConfigFactory.create_default_diffusion(kwargs)

        assert configs[0]["runtime"]["devices"] == "0,1,2,3"

    def test_cli_tp_size_applies_to_stage(self):
        """Test that global --tensor-parallel-size applies to stages (@fake0fan)."""
        stages = [
            StageConfig(stage_id=0, model_stage="thinker", input_sources=[]),
        ]
        cli_overrides = {"tensor_parallel_size": 4}

        overrides = StageConfigFactory._merge_cli_overrides(stages[0], cli_overrides)

        assert overrides["tensor_parallel_size"] == 4

    def test_cli_gpu_mem_applies_to_stage(self):
        """Test that global --gpu-memory-utilization applies to stages (@ZJY0516)."""
        stages = [
            StageConfig(stage_id=0, model_stage="thinker", input_sources=[]),
        ]
        cli_overrides = {"gpu_memory_utilization": 0.8}

        overrides = StageConfigFactory._merge_cli_overrides(stages[0], cli_overrides)

        assert overrides["gpu_memory_utilization"] == 0.8

    def test_per_stage_override_precedence(self):
        """Test that --stage-0-gpu-memory-utilization overrides global."""
        stage = StageConfig(stage_id=0, model_stage="thinker", input_sources=[])
        cli_overrides = {
            "gpu_memory_utilization": 0.5,  # Global
            "stage_0_gpu_memory_utilization": 0.9,  # Per-stage override
        }

        overrides = StageConfigFactory._merge_cli_overrides(stage, cli_overrides)

        # Per-stage should override global
        assert overrides["gpu_memory_utilization"] == 0.9

    def test_memory_auto_allocation(self):
        """Test safe defaults when stages share GPU."""
        stages = [
            StageConfig(
                stage_id=0,
                model_stage="stage_a",
                runtime_overrides={"devices": "0"},
            ),
            StageConfig(
                stage_id=1,
                model_stage="stage_b",
                runtime_overrides={"devices": "0"},  # Same device
            ),
        ]

        allocation = StageConfigFactory.auto_allocate_memory(stages)

        # Two stages sharing one device should each get 0.45 (0.9 / 2)
        assert allocation[0] == pytest.approx(0.45)
        assert allocation[1] == pytest.approx(0.45)

    def test_memory_auto_allocation_separate_devices(self):
        """Test allocation when stages use different devices."""
        stages = [
            StageConfig(
                stage_id=0,
                model_stage="stage_a",
                runtime_overrides={"devices": "0"},
            ),
            StageConfig(
                stage_id=1,
                model_stage="stage_b",
                runtime_overrides={"devices": "1"},  # Different device
            ),
        ]

        allocation = StageConfigFactory.auto_allocate_memory(stages)

        # Each stage gets full allocation on its device
        assert allocation[0] == pytest.approx(0.9)
        assert allocation[1] == pytest.approx(0.9)

    def test_arch_mapping(self):
        """Test that model architecture mapping is correct."""
        assert StageConfigFactory.ARCH_MAPPING["qwen3_omni_moe"] == "Qwen3OmniMoeForConditionalGeneration"
        assert StageConfigFactory.ARCH_MAPPING["qwen2_5_omni"] == "Qwen2_5OmniForConditionalGeneration"
        assert StageConfigFactory.ARCH_MAPPING["bagel"] == "BagelForConditionalGeneration"


class TestExtractRuntimeOverrides:
    """Tests for extract_runtime_overrides helper function."""

    def test_extract_basic_params(self):
        """Test extracting basic runtime parameters."""
        from vllm_omni.entrypoints.utils import extract_runtime_overrides

        kwargs = {
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 2,
            "devices": "0,1",
            "unrelated_param": "ignored",
            "none_param": None,  # Should be excluded
        }

        result = extract_runtime_overrides(kwargs)

        assert result["gpu_memory_utilization"] == 0.9
        assert result["tensor_parallel_size"] == 2
        assert result["devices"] == "0,1"
        assert "unrelated_param" not in result
        assert "none_param" not in result

    def test_extract_per_stage_overrides(self):
        """Test extracting per-stage override parameters."""
        from vllm_omni.entrypoints.utils import extract_runtime_overrides

        kwargs = {
            "gpu_memory_utilization": 0.5,
            "stage_0_gpu_memory_utilization": 0.9,
            "stage_1_tensor_parallel_size": 4,
            "stage_2_devices": "2,3",
        }

        result = extract_runtime_overrides(kwargs)

        assert result["gpu_memory_utilization"] == 0.5
        assert result["stage_0_gpu_memory_utilization"] == 0.9
        assert result["stage_1_tensor_parallel_size"] == 4
        assert result["stage_2_devices"] == "2,3"


class TestCleanedTopologyNoEngineParams:
    """Tests verifying Tier-1 YAML has NO engine params (@hsliuustc0106)."""

    def test_stage_config_no_engine_params(self):
        """Test that StageConfig doesn't have engine params like gpu_mem/tp_size/devices."""
        config = StageConfig(stage_id=0, model_stage="test")

        # These should NOT be direct attributes of StageConfig
        assert not hasattr(config, "gpu_memory_utilization")
        assert not hasattr(config, "tensor_parallel_size")
        assert not hasattr(config, "devices")
        assert not hasattr(config, "max_batch_size")
        assert not hasattr(config, "enforce_eager")
        assert not hasattr(config, "trust_remote_code")

    def test_stage_config_has_topology_fields(self):
        """Test that StageConfig has only Tier-1 topology fields."""
        config = StageConfig(stage_id=0, model_stage="test")

        # These ARE valid Tier-1 fields
        assert hasattr(config, "stage_id")
        assert hasattr(config, "model_stage")
        assert hasattr(config, "stage_type")
        assert hasattr(config, "input_sources")
        assert hasattr(config, "custom_process_input_func")
        assert hasattr(config, "final_output")
        assert hasattr(config, "final_output_type")
        assert hasattr(config, "worker_type")
        assert hasattr(config, "scheduler_cls")
        assert hasattr(config, "hf_config_name")


class TestRequestParamsNotInConfig:
    """Tests verifying request params (height/width/seed) NOT in config (@fake0fan)."""

    def test_stage_config_no_request_params(self):
        """Test that StageConfig doesn't have request-level params."""
        config = StageConfig(stage_id=0, model_stage="test")

        # These should NOT be in config - they're request-level
        assert not hasattr(config, "height")
        assert not hasattr(config, "width")
        assert not hasattr(config, "num_inference_steps")
        assert not hasattr(config, "seed")
        assert not hasattr(config, "prompt")
        assert not hasattr(config, "output_path")

    def test_topology_no_request_params(self):
        """Test that StageTopology doesn't have request-level params."""
        topology = StageTopology(model_type="test", stages=[])

        assert not hasattr(topology, "height")
        assert not hasattr(topology, "width")
        assert not hasattr(topology, "seed")
