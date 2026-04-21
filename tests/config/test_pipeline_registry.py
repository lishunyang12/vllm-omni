# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the central pipeline registry (2.5/N)."""

from __future__ import annotations

import pytest

from vllm_omni.config.pipeline_registry import _DIFFUSION_PIPELINES, _OMNI_PIPELINES
from vllm_omni.config.stage_config import (
    _PIPELINE_REGISTRY,
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    register_pipeline,
)


class TestCentralRegistryDeclarations:
    """Every in-tree pipeline must be discoverable via the central registry."""

    def test_omni_entries_visible(self):
        for key in _OMNI_PIPELINES:
            assert key in _PIPELINE_REGISTRY

    def test_diffusion_entries_visible(self):
        for key in _DIFFUSION_PIPELINES:
            assert key in _PIPELINE_REGISTRY

    def test_no_duplicate_model_type_between_omni_and_diffusion(self):
        overlap = set(_OMNI_PIPELINES) & set(_DIFFUSION_PIPELINES)
        assert not overlap, f"Duplicate model_types across omni/diffusion: {overlap}"

    def test_expected_omni_pipelines_present(self):
        # Guard against accidental removal during future refactors.
        assert "qwen2_5_omni" in _OMNI_PIPELINES
        assert "qwen2_5_omni_thinker_only" in _OMNI_PIPELINES
        assert "qwen3_omni_moe" in _OMNI_PIPELINES
        assert "qwen3_tts" in _OMNI_PIPELINES


class TestLazyLoading:
    """Pipelines are imported only on first access."""

    def test_contains_without_import(self):
        # ``in`` hits the lazy map, not the loaded cache.
        assert "qwen3_omni_moe" in _PIPELINE_REGISTRY

    def test_getitem_loads_correct_pipeline(self):
        pipeline = _PIPELINE_REGISTRY["qwen3_omni_moe"]
        assert pipeline.model_type == "qwen3_omni_moe"
        assert pipeline.model_arch == "Qwen3OmniMoeForConditionalGeneration"

    def test_unknown_model_type_returns_none_via_get(self):
        assert _PIPELINE_REGISTRY.get("not_a_real_pipeline") is None

    def test_unknown_model_type_raises_keyerror_via_getitem(self):
        with pytest.raises(KeyError):
            _PIPELINE_REGISTRY["not_a_real_pipeline"]

    def test_iteration_yields_registered_pipelines(self):
        keys = set(_PIPELINE_REGISTRY)
        assert "qwen2_5_omni" in keys
        assert "qwen3_omni_moe" in keys


class TestSingleStageDiffusion:
    """Single-stage diffusion pipelines are pre-built in ``_DIFFUSION_PIPELINES``."""

    def test_uniform_topology(self):
        pipeline = _PIPELINE_REGISTRY["flux"]
        assert pipeline.model_type == "flux"
        assert pipeline.model_arch == "FluxPipeline"
        assert len(pipeline.stages) == 1
        stage = pipeline.stages[0]
        assert stage.execution_type == StageExecutionType.DIFFUSION
        assert stage.final_output is True
        assert stage.final_output_type == "image"
        assert stage.model_arch == "FluxPipeline"

    def test_lookup_returns_same_object(self):
        # Pre-built configs: registry hands back the same instance.
        first = _PIPELINE_REGISTRY["qwen_image"]
        second = _PIPELINE_REGISTRY["qwen_image"]
        assert first is second

    def test_dynamic_registration_overrides_diffusion_entry(self):
        custom = PipelineConfig(
            model_type="flux",
            model_arch="OverriddenFlux",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="dit",
                    execution_type=StageExecutionType.DIFFUSION,
                    final_output=True,
                ),
            ),
        )
        register_pipeline(custom)
        try:
            assert _PIPELINE_REGISTRY["flux"].model_arch == "OverriddenFlux"
        finally:
            if "flux" in _PIPELINE_REGISTRY._loaded:
                del _PIPELINE_REGISTRY["flux"]


class TestDynamicRegistration:
    """``register_pipeline()`` still works for plugins and tests."""

    def test_register_adds_to_registry(self):
        custom = PipelineConfig(
            model_type="_test_dynamic_registration",
            model_arch="DynamicTestModel",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="test",
                    execution_type=StageExecutionType.LLM_AR,
                    input_sources=(),
                    final_output=True,
                ),
            ),
        )
        register_pipeline(custom)
        try:
            assert "_test_dynamic_registration" in _PIPELINE_REGISTRY
            assert _PIPELINE_REGISTRY["_test_dynamic_registration"] is custom
        finally:
            # Don't leak the test registration into other tests.
            if "_test_dynamic_registration" in _PIPELINE_REGISTRY:
                del _PIPELINE_REGISTRY["_test_dynamic_registration"]

    def test_dynamic_registration_overrides_lazy_entry(self):
        # Build a substitute for qwen3_omni_moe that we can distinguish.
        original = _PIPELINE_REGISTRY["qwen3_omni_moe"]
        override = PipelineConfig(
            model_type="qwen3_omni_moe",
            model_arch="OverriddenArch",
            stages=original.stages,
        )
        register_pipeline(override)
        try:
            assert _PIPELINE_REGISTRY["qwen3_omni_moe"].model_arch == "OverriddenArch"
        finally:
            # Remove the dynamic override so later tests see the original.
            if "qwen3_omni_moe" in _PIPELINE_REGISTRY._loaded:
                del _PIPELINE_REGISTRY["qwen3_omni_moe"]
