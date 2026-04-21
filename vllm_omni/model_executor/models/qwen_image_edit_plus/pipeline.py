# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen-Image Edit Plus (multi-image editing) pipeline topology — single DiT stage."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

QWEN_IMAGE_EDIT_PLUS_PIPELINE = PipelineConfig(
    model_type="qwen_image_edit_plus",
    model_arch="QwenImageEditPlusPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            final_output=True,
            final_output_type="image",
            model_arch="QwenImageEditPlusPipeline",
        ),
    ),
)
