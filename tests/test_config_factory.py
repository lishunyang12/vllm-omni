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
        errors = topology.validate_topology()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_valid_branching_dag(self):
        """Test validation of a valid branching DAG."""
        stages = [
            StageConfig(stage_id=0, model_stage="input", input_sources=[]),
            StageConfig(stage_id=1, model_stage="branch_a", input_sources=[0]),
            StageConfig(stage_id=2, model_stage="branch_b", input_sources=[0]),
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_topology()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_entry_point(self):
        """Test that missing entry point is detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="stage_a", input_sources=[1]),
            StageConfig(stage_id=1, model_stage="stage_b", input_sources=[0]),
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_topology()
        assert any("entry point" in e.lower() for e in errors)

    def test_missing_dependency(self):
        """Test that missing stage reference is detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="input", input_sources=[]),
            StageConfig(stage_id=1, model_stage="output", input_sources=[99]),  # Invalid
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_topology()
        assert any("non-existent" in e.lower() for e in errors)

    def test_duplicate_stage_ids(self):
        """Test that duplicate stage IDs are detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="stage_a", input_sources=[]),
            StageConfig(stage_id=0, model_stage="stage_b", input_sources=[]),  # Duplicate
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_topology()
        assert any("duplicate" in e.lower() for e in errors)

    def test_mutual_dependency_detected_as_missing_entry(self):
        """Test that mutual dependencies are caught (no entry point)."""
        stages = [
            StageConfig(stage_id=0, model_stage="stage_a", input_sources=[1]),
            StageConfig(stage_id=1, model_stage="stage_b", input_sources=[0]),
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_topology()
        assert any("entry point" in e.lower() for e in errors)

    def test_self_reference(self):
        """Test that self-references are detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="entry", input_sources=[]),
            StageConfig(stage_id=1, model_stage="self_ref", input_sources=[1]),  # Self
        ]
        topology = StageTopology(model_type="test", stages=stages)
        errors = topology.validate_topology()
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
        errors = topology.validate_topology()
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

    def test_cli_override_forwards_engine_registered_args(self):
        """Test that any engine-registered CLI arg is forwarded (@wuhang2014)."""
        stage = StageConfig(stage_id=0, model_stage="thinker", input_sources=[])
        cli_overrides = {
            "gpu_memory_utilization": 0.9,  # Well-known param
            "custom_engine_flag": True,  # Engine-registered but not in RUNTIME_PARAMS
        }

        overrides = StageConfigFactory._merge_cli_overrides(stage, cli_overrides)

        assert overrides["gpu_memory_utilization"] == 0.9
        assert overrides["custom_engine_flag"] is True

    def test_cli_override_excludes_internal_keys(self):
        """Test that internal/orchestrator keys are not forwarded."""
        stage = StageConfig(stage_id=0, model_stage="thinker", input_sources=[])
        cli_overrides = {
            "gpu_memory_utilization": 0.9,
            "model": "some_model",  # Internal
            "stage_configs_path": "/path",  # Internal
            "batch_timeout": 10,  # Internal
        }

        overrides = StageConfigFactory._merge_cli_overrides(stage, cli_overrides)

        assert overrides["gpu_memory_utilization"] == 0.9
        assert "model" not in overrides
        assert "stage_configs_path" not in overrides
        assert "batch_timeout" not in overrides

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
            "custom_engine_param": "forwarded",
            "none_param": None,  # Should be excluded
            "model": "some_model",  # Internal key, should be excluded
        }

        result = extract_runtime_overrides(kwargs)

        assert result["gpu_memory_utilization"] == 0.9
        assert result["tensor_parallel_size"] == 2
        assert result["devices"] == "0,1"
        # Non-internal, non-None params are forwarded (extensible overrides)
        assert result["custom_engine_param"] == "forwarded"
        assert "none_param" not in result
        # Internal keys are excluded
        assert "model" not in result

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


class TestTopologyYamlParsing:
    """Tests for stage topology YAML file parsing (@ZJY0516)."""

    def test_parse_qwen3_omni_moe_yaml(self, tmp_path):
        """Test parsing the qwen3_omni_moe topology YAML."""
        yaml_content = """\
model_type: qwen3_omni_moe

stages:
  - stage_id: 0
    model_stage: thinker
    stage_type: llm
    input_sources: []
    worker_type: ar
    scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
    hf_config_name: thinker_config
    final_output: true
    final_output_type: text
    is_comprehension: true

  - stage_id: 1
    model_stage: talker
    stage_type: llm
    input_sources: [0]
    worker_type: ar
    scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
    hf_config_name: talker_config
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker

  - stage_id: 2
    model_stage: code2wav
    stage_type: llm
    input_sources: [1]
    worker_type: generation
    scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
    hf_config_name: thinker_config
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav
    final_output: true
    final_output_type: audio
"""
        yaml_file = tmp_path / "qwen3_omni_moe.yaml"
        yaml_file.write_text(yaml_content)

        topology = StageConfigFactory._parse_topology_yaml(yaml_file, "qwen3_omni_moe")

        assert topology.model_type == "qwen3_omni_moe"
        assert len(topology.stages) == 3

        # Stage 0: thinker
        s0 = topology.stages[0]
        assert s0.stage_id == 0
        assert s0.model_stage == "thinker"
        assert s0.stage_type == StageType.LLM
        assert s0.input_sources == []
        assert s0.worker_type == "ar"
        assert s0.final_output is True
        assert s0.final_output_type == "text"
        assert s0.is_comprehension is True

        # Stage 1: talker
        s1 = topology.stages[1]
        assert s1.stage_id == 1
        assert s1.input_sources == [0]
        assert s1.custom_process_input_func == (
            "vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker"
        )
        assert s1.final_output is False

        # Stage 2: code2wav
        s2 = topology.stages[2]
        assert s2.stage_id == 2
        assert s2.input_sources == [1]
        assert s2.worker_type == "generation"
        assert s2.final_output_type == "audio"

    def test_parse_yaml_with_legacy_engine_input_source(self, tmp_path):
        """Test backward compatibility with engine_input_source field."""
        yaml_content = """\
model_type: legacy_model

stages:
  - stage_id: 0
    model_stage: entry
    stage_type: llm
  - stage_id: 1
    model_stage: downstream
    stage_type: llm
    engine_input_source: [0]
"""
        yaml_file = tmp_path / "legacy.yaml"
        yaml_file.write_text(yaml_content)

        topology = StageConfigFactory._parse_topology_yaml(yaml_file, "legacy_model")
        assert topology.stages[1].input_sources == [0]

    def test_parse_yaml_with_connectors_and_edges(self, tmp_path):
        """Test parsing topology with optional connectors and edges."""
        yaml_content = """\
model_type: test_model

stages:
  - stage_id: 0
    model_stage: entry
    stage_type: llm
    input_sources: []

connectors:
  type: ray

edges:
  - from: 0
    to: 1
"""
        yaml_file = tmp_path / "with_connectors.yaml"
        yaml_file.write_text(yaml_content)

        topology = StageConfigFactory._parse_topology_yaml(yaml_file, "test_model")
        assert topology.connectors == {"type": "ray"}
        assert topology.edges == [{"from": 0, "to": 1}]

    def test_parsed_topology_passes_validation(self, tmp_path):
        """Test that a well-formed YAML produces a valid topology."""
        yaml_content = """\
model_type: valid_model

stages:
  - stage_id: 0
    model_stage: entry
    stage_type: llm
    input_sources: []
    final_output: true
    final_output_type: text
  - stage_id: 1
    model_stage: next
    stage_type: llm
    input_sources: [0]
"""
        yaml_file = tmp_path / "valid.yaml"
        yaml_file.write_text(yaml_content)

        topology = StageConfigFactory._parse_topology_yaml(yaml_file, "valid_model")
        errors = topology.validate_topology()
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_parse_diffusion_stage_type(self, tmp_path):
        """Test parsing a diffusion stage type from YAML."""
        yaml_content = """\
model_type: diff_model

stages:
  - stage_id: 0
    model_stage: dit
    stage_type: diffusion
    input_sources: []
    final_output: true
    final_output_type: image
"""
        yaml_file = tmp_path / "diffusion.yaml"
        yaml_file.write_text(yaml_content)

        topology = StageConfigFactory._parse_topology_yaml(yaml_file, "diff_model")
        assert topology.stages[0].stage_type == StageType.DIFFUSION


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
