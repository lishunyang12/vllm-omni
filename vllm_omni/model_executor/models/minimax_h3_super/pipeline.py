# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax H3 Turbo draft followed by LTX-2.5 refinement."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

MINIMAX_H3_SUPER_PIPELINE = PipelineConfig(
    model_type="minimax_h3_super",
    default_deploy_config_name="minimax_h3_super.yaml",
    model_arch="MiniMaxH3SuperDraftPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="h3_draft",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=False,
            final_output_type="video",
            model_arch="MiniMaxH3SuperDraftPipeline",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="ltx25_refine",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="video",
            model_arch="LTX25H3RefinerPipeline",
            custom_process_input_func=(
                "vllm_omni.model_executor.stage_input_processors.minimax_h3_super.h3_to_ltx25_refiner"
            ),
        ),
    ),
)

__all__ = ["MINIMAX_H3_SUPER_PIPELINE"]
