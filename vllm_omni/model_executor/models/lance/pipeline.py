# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lance single-stage pipeline.

Stage 0 is a self-contained diffusion stage that handles all modalities
internally via its own Qwen2-MoT LLM, Qwen2.5-VL ViT, Wan2.2 VAE and
tokenizer. Mirrors ``bagel_single_stage``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

LANCE_SINGLE_STAGE_PIPELINE = PipelineConfig(
    model_type="lance",
    model_arch="LancePipeline",
    hf_architectures=(),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
        ),
    ),
)
