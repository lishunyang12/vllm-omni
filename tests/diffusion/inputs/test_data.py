"""
Tests for sampling parameter override behaviors.
"""

from copy import deepcopy

from vllm_omni.inputs.data import DiffusionParamOverrides, OmniDiffusionSamplingParams


def test_merge_nothing():
    """Ensure that merging nothing doesn't break anything."""
    user_params = OmniDiffusionSamplingParams()
    overrides = DiffusionParamOverrides()
    orig_params = deepcopy(user_params)
    user_params.merge_with_def_params(overrides)
    assert user_params.__dict__ == orig_params.__dict__
    assert user_params._init_kwargs == set()


def test_merge_unset():
    """Ensure that we can override fields that are unset."""
    default_steps = 777
    user_params = OmniDiffusionSamplingParams()
    overrides = DiffusionParamOverrides(num_inference_steps=default_steps)
    user_params.merge_with_def_params(overrides)
    assert user_params.num_inference_steps == 777
    assert user_params._init_kwargs == set()


def test_merge_priority():
    """Ensure that explicitly passed values won't be overridden by pipelines."""
    user_steps = 888
    model_steps = 777
    user_params = OmniDiffusionSamplingParams(
        num_inference_steps=user_steps,
    )
    overrides = DiffusionParamOverrides(num_inference_steps=model_steps)
    user_params.merge_with_def_params(overrides)
    assert user_params.num_inference_steps == user_steps
    assert user_params._init_kwargs == {"num_inference_steps"}


def test_merge_multiple():
    """Ensure that we can merge over truthy or falsy default values."""
    model_steps = 888
    model_resolution = 320
    user_params = OmniDiffusionSamplingParams()
    overrides = DiffusionParamOverrides(
        num_inference_steps=model_steps,  # Falsy (None) by default
        resolution=model_resolution,  # 640 by default
    )
    user_params.merge_with_def_params(overrides)
    assert user_params.num_inference_steps == model_steps
    assert user_params.resolution == model_resolution
    assert user_params._init_kwargs == set()


def test_hierarchical_merge_complex():
    """Tests merge priority with multiple values."""
    user_steps = 100
    user_height = 100
    user_width = 100
    model_steps = 888  # clobbered by user steps
    model_resolution = 320

    user_params = OmniDiffusionSamplingParams(
        num_inference_steps=user_steps,
        height=user_height,
        width=user_width,
    )
    overrides = DiffusionParamOverrides(
        num_inference_steps=model_steps,  # lower priority than user param
        resolution=model_resolution,
    )
    user_params.merge_with_def_params(overrides)
    assert user_params.num_inference_steps == user_steps
    assert user_params.height == user_height
    assert user_params.width == user_width
    assert user_params.resolution == model_resolution
    assert user_params._init_kwargs == {"num_inference_steps", "height", "width"}


def test_can_pass_falsy_override():
    user_params = OmniDiffusionSamplingParams(num_inference_steps=None)
    overrides = DiffusionParamOverrides(
        num_inference_steps=100,
    )
    user_params.merge_with_def_params(overrides)
    assert user_params.num_inference_steps is None
    assert user_params._init_kwargs == {"num_inference_steps"}
