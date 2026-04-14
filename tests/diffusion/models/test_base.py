# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast interface checks for all Diffusion pipelines."""

import pytest

from vllm_omni.diffusion.models.interface import VllmDiffusionPipeline
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.inputs.data import DiffusionParamOverrides, OmniDiffusionSamplingParams

# Pipelines to omit from common tests; this should be done sparingly
# as the tests are generic, and only added for
SKIP_PIPELINES = ["DreamIDOmniPipeline"]

# Instance variables that need to be mocked for sampling_param_defaults
INSTANCE_VAR_MOCKS = {
    "LTX2Pipeline": {"tokenizer_max_length": 512},
    "LTX2ImageToVideoPipeline": {"tokenizer_max_length": 512},
}

TEST_PIPELINES = [pipe for pipe in DiffusionModelRegistry.models.keys() if pipe not in SKIP_PIPELINES]


@pytest.mark.parametrize("pipeline_type", TEST_PIPELINES)
def test_pipelines_are_vllm_diffusion_pipeline(pipeline_type):
    """Ensure all pipelines are instances of VllmDiffusionPipeline"""
    pipe_class = DiffusionModelRegistry._try_load_model_cls(pipeline_type)
    assert pipe_class is not None
    assert issubclass(pipe_class, VllmDiffusionPipeline)


@pytest.mark.parametrize("pipeline_type", TEST_PIPELINES)
def test_pipeline_sampling_params_are_valid(pipeline_type):
    """Ensure all pipelines define sampling_param_defaults with valid param kwargs."""
    pipe_class = DiffusionModelRegistry._try_load_model_cls(pipeline_type)
    assert pipe_class is not None

    # Create an uninitialized instance; this is easier than going through init/model load
    # since the vast majority of models do not use instance vars in their default params
    pipe_instance = object.__new__(pipe_class)

    # Patch instance variables for any pipelines that do need it
    if pipeline_type in INSTANCE_VAR_MOCKS:
        for attr_name, attr_value in INSTANCE_VAR_MOCKS[pipeline_type].items():
            setattr(pipe_instance, attr_name, attr_value)

    # Verify sampling_param_defaults exists and has at least one key, since at a
    # minimum every class will inherit num_inference_steps from the base class
    defaults = pipe_instance.sampling_param_defaults
    assert isinstance(defaults, DiffusionParamOverrides)
    assert hasattr(defaults, "validated_overrides")
    assert len(defaults.validated_overrides) > 0

    # Ensure we can create a diffusion sampling params object (i.e., kwargs are valid)
    params = OmniDiffusionSamplingParams(**defaults.validated_overrides)
    for attr_name, val in defaults.validated_overrides.items():
        assert hasattr(params, attr_name)
        assert getattr(params, attr_name) == val
