# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for frozen PipelineConfig dataclasses (RFC #2072 2/N)."""

from __future__ import annotations

import pytest

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    _EXECUTION_TYPE_TO_SCHEDULER,
    _PIPELINE_REGISTRY,
    register_pipeline,
)


class TestStagePipelineConfig:
    def test_frozen(self):
        s = StagePipelineConfig(stage_id=0, model_stage="a")
        with pytest.raises(AttributeError):
            s.model_stage = "changed"

    def test_defaults(self):
        s = StagePipelineConfig(stage_id=0, model_stage="a")
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.input_sources == ()
        assert s.final_output is False
        assert s.is_comprehension is False
        assert s.requires_multimodal_data is False
        assert s.sampling_constraints == {}
        assert s.engine_output_type is None
        assert s.custom_process_next_stage_input_func is None


class TestPipelineConfig:
    def test_frozen(self):
        p = PipelineConfig(model_type="t", model_arch="A")
        with pytest.raises(AttributeError):
            p.model_type = "changed"

    def test_validate_valid(self):
        p = PipelineConfig(
            model_type="t", model_arch="A",
            stages=(
                StagePipelineConfig(stage_id=0, model_stage="a"),
                StagePipelineConfig(stage_id=1, model_stage="b", input_sources=(0,)),
            ),
        )
        assert p.validate() == []

    def test_validate_no_stages(self):
        p = PipelineConfig(model_type="t", model_arch="A")
        assert any("no stages" in e.lower() for e in p.validate())

    def test_validate_duplicate_ids(self):
        p = PipelineConfig(
            model_type="t", model_arch="A",
            stages=(
                StagePipelineConfig(stage_id=0, model_stage="a"),
                StagePipelineConfig(stage_id=0, model_stage="b"),
            ),
        )
        assert any("duplicate" in e.lower() for e in p.validate())

    def test_validate_bad_source(self):
        p = PipelineConfig(
            model_type="t", model_arch="A",
            stages=(StagePipelineConfig(stage_id=0, model_stage="a", input_sources=(99,)),),
        )
        assert any("non-existent" in e.lower() for e in p.validate())

    def test_validate_self_ref(self):
        p = PipelineConfig(
            model_type="t", model_arch="A",
            stages=(StagePipelineConfig(stage_id=0, model_stage="a", input_sources=(0,)),),
        )
        assert any("itself" in e.lower() for e in p.validate())

    def test_get_stage(self):
        s = StagePipelineConfig(stage_id=0, model_stage="a")
        p = PipelineConfig(model_type="t", model_arch="A", stages=(s,))
        assert p.get_stage(0) is s
        assert p.get_stage(99) is None

    def test_get_scheduler_cls(self):
        p = PipelineConfig(
            model_type="t", model_arch="A",
            stages=(
                StagePipelineConfig(stage_id=0, model_stage="a",
                                    execution_type=StageExecutionType.LLM_AR),
                StagePipelineConfig(stage_id=1, model_stage="b",
                                    execution_type=StageExecutionType.LLM_GENERATION,
                                    input_sources=(0,)),
            ),
        )
        assert "OmniARScheduler" in p.get_scheduler_cls(0)
        assert "OmniGenerationScheduler" in p.get_scheduler_cls(1)


class TestExecutionTypeToScheduler:
    def test_all_types_mapped(self):
        for et in StageExecutionType:
            assert et in _EXECUTION_TYPE_TO_SCHEDULER


class TestRegistry:
    def test_register_and_lookup(self):
        p = PipelineConfig(
            model_type="__test_only__", model_arch="A",
            stages=(StagePipelineConfig(stage_id=0, model_stage="a"),),
        )
        register_pipeline(p)
        assert _PIPELINE_REGISTRY["__test_only__"] is p
        del _PIPELINE_REGISTRY["__test_only__"]


class TestDeployConfigLoading:
    def test_load_deploy_config(self):
        from pathlib import Path

        from vllm_omni.config.stage_config import load_deploy_config

        deploy_path = (
            Path(__file__).parent.parent
            / "vllm_omni"
            / "deploy"
            / "qwen3_omni_moe.yaml"
        )
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        assert len(deploy.stages) == 3
        assert deploy.stages[0].stage_id == 0
        assert deploy.async_chunk is True
        assert deploy.connectors is not None
        assert "connector_of_shared_memory" in deploy.connectors
        assert deploy.edges is not None
        assert len(deploy.edges) == 2
        assert deploy.stages[0].output_connectors is not None
        assert deploy.stages[0].default_sampling_params is not None
        assert deploy.platforms is not None
        assert "npu" in deploy.platforms

    def test_merge_pipeline_deploy(self):
        from pathlib import Path

        from vllm_omni.config.stage_config import (
            load_deploy_config,
            merge_pipeline_deploy,
        )

        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401

        pipeline = _PIPELINE_REGISTRY["qwen3_omni_moe"]
        deploy_path = (
            Path(__file__).parent.parent
            / "vllm_omni"
            / "deploy"
            / "qwen3_omni_moe.yaml"
        )
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        stages = merge_pipeline_deploy(pipeline, deploy)

        assert len(stages) == 3
        # Thinker — model_arch from pipeline, engine_output_type from pipeline
        s0 = stages[0]
        assert s0.model_stage == "thinker"
        assert s0.is_comprehension is True
        assert s0.yaml_engine_args["model_arch"] == "Qwen3OmniMoeForConditionalGeneration"
        assert s0.yaml_engine_args["engine_output_type"] == "latent"
        assert s0.yaml_engine_args.get("async_chunk") is True
        # sampling: deploy defaults + pipeline constraints merged
        assert "default_sampling_params" in s0.yaml_extras
        assert s0.yaml_extras["default_sampling_params"]["detokenize"] is True
        # Connectors from deploy
        assert "output_connectors" in s0.yaml_extras
        # Talker
        s1 = stages[1]
        assert s1.input_sources == [0]
        assert s1.yaml_extras["default_sampling_params"]["stop_token_ids"] == [2150]
        assert "input_connectors" in s1.yaml_extras
        # Code2wav
        s2 = stages[2]
        assert s2.final_output is True
        assert s2.final_output_type == "audio"


class TestQwen3OmniPipeline:
    def test_registered(self):
        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401

        p = _PIPELINE_REGISTRY.get("qwen3_omni_moe")
        assert p is not None
        assert p.model_arch == "Qwen3OmniMoeForConditionalGeneration"
        assert len(p.stages) == 3
        assert p.validate() == []

    def test_thinker(self):
        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen3_omni_moe"].get_stage(0)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.is_comprehension is True
        assert s.requires_multimodal_data is True
        assert s.final_output is True
        assert s.engine_output_type == "latent"
        assert s.sampling_constraints["detokenize"] is True
        # Both hooks defined
        assert s.custom_process_next_stage_input_func is not None

    def test_talker(self):
        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen3_omni_moe"].get_stage(1)
        assert s.input_sources == (0,)
        assert s.sampling_constraints["stop_token_ids"] == [2150]
        # Both sync and async hooks defined
        assert s.custom_process_input_func is not None
        assert s.custom_process_next_stage_input_func is not None

    def test_code2wav(self):
        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen3_omni_moe"].get_stage(2)
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"
        assert s.custom_process_input_func is not None
