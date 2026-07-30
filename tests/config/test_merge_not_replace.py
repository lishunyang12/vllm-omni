# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RFC #4867: merge-not-replace — a user deploy config overlays the pipeline's
default deploy instead of replacing it, so a thin override keeps the base's knobs."""

import pytest

from vllm_omni.config.stage_config import DeployConfig, StageDeployConfig, merge_deploy_configs


def _base() -> DeployConfig:
    return DeployConfig(
        async_chunk=False,
        trust_remote_code=True,
        stages=[
            StageDeployConfig(
                stage_id=0,
                enforce_eager=True,
                max_num_seqs=1,
                gpu_memory_utilization=0.5,
                default_sampling_params={"seed": 42, "max_tokens": 256},
            )
        ],
    )


def test_thin_overlay_keeps_base_knobs():
    overlay = DeployConfig(stages=[StageDeployConfig(stage_id=0, gpu_memory_utilization=0.9)])
    merged = merge_deploy_configs(_base(), overlay)

    # base knobs the thin overlay did NOT set are preserved
    assert merged.async_chunk is False
    assert merged.trust_remote_code is True
    s = merged.stages[0]
    assert s.enforce_eager is True
    assert s.max_num_seqs == 1
    # the one knob the overlay set wins
    assert s.gpu_memory_utilization == 0.9


def test_dict_engine_fields_deep_merge():
    overlay = DeployConfig(stages=[StageDeployConfig(stage_id=0, default_sampling_params={"temperature": 0.7})])
    merged = merge_deploy_configs(_base(), overlay)
    # overlay adds a key without dropping the base's sampling params
    assert merged.stages[0].default_sampling_params == {"seed": 42, "max_tokens": 256, "temperature": 0.7}


def test_full_overlay_matches_replace_backward_compat():
    # a complete overlay yields exactly the overlay's values for the (None-default)
    # engine knobs — identical to the old replace semantics. (A scalar with a
    # meaningful non-None default, e.g. async_chunk, can't be set *back to* its
    # default via an overlay; that's handled by the pipeline / CLI, not here.)
    full = DeployConfig(
        trust_remote_code=False,
        stages=[StageDeployConfig(stage_id=0, enforce_eager=False, max_num_seqs=8, gpu_memory_utilization=0.8)],
    )
    merged = merge_deploy_configs(_base(), full)
    assert merged.trust_remote_code is False
    s = merged.stages[0]
    assert (s.enforce_eager, s.max_num_seqs, s.gpu_memory_utilization) == (False, 8, 0.8)


def test_base_is_not_mutated():
    base = _base()
    merge_deploy_configs(base, DeployConfig(stages=[StageDeployConfig(stage_id=0, gpu_memory_utilization=0.9)]))
    assert base.stages[0].gpu_memory_utilization == 0.5  # template untouched


def test_new_stage_in_overlay_is_added():
    overlay = DeployConfig(stages=[StageDeployConfig(stage_id=1, max_num_seqs=4)])
    merged = merge_deploy_configs(_base(), overlay)
    assert [s.stage_id for s in merged.stages] == [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
