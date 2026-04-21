# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helios (text-to-image) pipeline topology — single DiT stage.

The Pyramid variant shares the same pipeline class in ``_DIFFUSION_MODELS``,
so a single ``helios`` registration covers both.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

HELIOS_PIPELINE = PipelineConfig(
    model_type="helios",
    model_arch="HeliosPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            final_output=True,
            final_output_type="image",
            model_arch="HeliosPipeline",
        ),
    ),
)
