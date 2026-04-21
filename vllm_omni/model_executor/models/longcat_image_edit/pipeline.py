# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LongCat Image Edit (image editing) pipeline topology — single DiT stage."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

LONGCAT_IMAGE_EDIT_PIPELINE = PipelineConfig(
    model_type="longcat_image_edit",
    model_arch="LongCatImageEditPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            final_output=True,
            final_output_type="image",
            model_arch="LongCatImageEditPipeline",
        ),
    ),
)
