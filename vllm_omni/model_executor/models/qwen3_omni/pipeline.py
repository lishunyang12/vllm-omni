# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen3-Omni-MoE pipeline topology (frozen, immutable).

Stage 0: Thinker — multimodal understanding + text generation
Stage 1: Talker  — text embeddings → RVQ codec codes
Stage 2: Code2Wav — RVQ codes → audio waveform

Sync processing hooks (custom_process_input_func) are defined here as part
of the fixed pipeline topology. Async hooks (custom_process_next_stage_input_func)
and engine_output_type live in deploy config since they depend on deployment mode.
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
    async_chunk=True,
    connectors={
        "connector_of_shared_memory": {
            "name": "SharedMemoryConnector",
            "extra": {
                "codec_chunk_frames": 25,
                "codec_left_context_frames": 25,
            },
        },
    },
    edges=[
        {"from": 0, "to": 1, "window_size": -1},
        {"from": 1, "to": 2, "window_size": -1},
    ],
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
            output_connectors={"to_stage_1": "connector_of_shared_memory"},
            default_sampling_params={
                "temperature": 0.4,
                "top_p": 0.9,
                "top_k": 1,
                "max_tokens": 2048,
                "seed": 42,
                "detokenize": True,
                "repetition_penalty": 1.05,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="talker_config",
            # sync path: talker receives full batch from thinker
            custom_process_input_func=f"{_PROC}.thinker2talker",
            input_connectors={"from_stage_0": "connector_of_shared_memory"},
            default_sampling_params={
                "temperature": 0.9,
                "top_k": 50,
                "max_tokens": 4096,
                "seed": 42,
                "detokenize": False,
                "repetition_penalty": 1.05,
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
            # sync path: code2wav receives full batch from talker
            custom_process_input_func=f"{_PROC}.talker2code2wav",
            default_sampling_params={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 65536,
                "seed": 42,
                "detokenize": True,
                "repetition_penalty": 1.1,
            },
        ),
    ),
)

register_pipeline(QWEN3_OMNI_PIPELINE)
