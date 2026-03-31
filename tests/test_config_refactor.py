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
        assert s.sampling_constraints == {}


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
        assert s.final_output is True
        assert s.sampling_constraints["detokenize"] is True

    def test_talker(self):
        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen3_omni_moe"].get_stage(1)
        assert s.input_sources == (0,)
        assert s.sampling_constraints["stop_token_ids"] == [2150]

    def test_code2wav(self):
        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen3_omni_moe"].get_stage(2)
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.final_output_type == "audio"
