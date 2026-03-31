# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen3-Omni-MoE pipeline topology (frozen, immutable).

Stage 0: Thinker — multimodal understanding + text generation
Stage 1: Talker  — text embeddings → RVQ codec codes
Stage 2: Code2Wav — RVQ codes → audio waveform

Both sync and async processing hooks are defined per stage. The deploy
config's ``async_chunk`` flag controls which path the engine uses:
  - async_chunk=true  → custom_process_next_stage_input_func + connectors
  - async_chunk=false → custom_process_input_func (sync, direct)
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    register_pipeline,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.qwen3_omni"

QWEN3_OMNI_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_moe",
    model_arch="Qwen3OmniMoeForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            is_comprehension=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            # async path: thinker pushes chunks to talker
            custom_process_next_stage_input_func=(
                f"{_PROC}.thinker2talker_async_chunk"
            ),
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="talker_config",
            engine_output_type="latent",
            # sync path: talker receives full batch from thinker
            custom_process_input_func=f"{_PROC}.thinker2talker",
            # async path: talker pushes chunks to code2wav
            custom_process_next_stage_input_func=(
                f"{_PROC}.talker2code2wav_async_chunk"
            ),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2150],
            },
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="thinker_config",
            engine_output_type="audio",
            # sync path: code2wav receives full batch from talker
            custom_process_input_func=f"{_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

register_pipeline(QWEN3_OMNI_PIPELINE)
