# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLUX.2 Klein (distilled text-to-image) pipeline topology — single DiT stage."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

FLUX2_KLEIN_PIPELINE = PipelineConfig(
    model_type="flux2_klein",
    model_arch="Flux2KleinPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            final_output=True,
            final_output_type="image",
            model_arch="Flux2KleinPipeline",
        ),
    ),
)
