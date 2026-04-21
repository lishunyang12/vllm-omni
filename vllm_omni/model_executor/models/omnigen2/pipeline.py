# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OmniGen2 (text + image conditioned) pipeline topology — single DiT stage."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

OMNIGEN2_PIPELINE = PipelineConfig(
    model_type="omnigen2",
    model_arch="OmniGen2Pipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            final_output=True,
            final_output_type="image",
            model_arch="OmniGen2Pipeline",
        ),
    ),
)
