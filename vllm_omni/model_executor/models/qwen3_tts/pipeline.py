# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-TTS pipeline topology (frozen).

Stage 0: Talker   — text → 8-layer RVQ codec codes
Stage 1: Code2Wav — RVQ codes → audio waveform

The talker uses ``Qwen3TTSTalkerForConditionalGeneration`` and the
code2wav stage uses ``Qwen3TTSCode2Wav``. Both are AR-style stages.

A single pipeline registration handles both the async-chunk streaming
topology and the synchronous end-to-end topology; which processor runs
is dispatched at merge time from ``deploy.async_chunk``:

* ``async_chunk=True``:  Stage 0's ``talker2code2wav_async_chunk``
  processor streams per-chunk codec windows to Stage 1 via the
  ``SharedMemoryConnector`` wired in ``deploy/qwen3_tts.yaml``.
* ``async_chunk=False``: Stage 1's ``talker2code2wav`` processor runs
  on the full codec output at batch end; no shm connector is needed.

To run the synchronous variant, set ``async_chunk: false`` in your deploy
yaml (or pass ``--no-async-chunk``).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    register_pipeline,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.qwen3_tts"

QWEN3_TTS_PIPELINE = PipelineConfig(
    model_type="qwen3_tts",
    # Pipeline-level default; the code2wav stage overrides per-stage below.
    model_arch="Qwen3TTSTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="qwen3_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            # Per-chunk codec streaming processor — fires only when
            # deploy.async_chunk=True.
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2150],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            # Code2Wav has its own model class, distinct from the talker.
            model_arch="Qwen3TTSCode2Wav",
            # Batch-end codec processor — fires only when
            # deploy.async_chunk=False.
            sync_process_input_func=f"{_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
            extras={
                # tts_args block — passed through to the code2wav stage
                # at runtime; not part of StageDeployConfig.
                "tts_args": {"max_instructions_length": 500},
            },
        ),
    ),
)

register_pipeline(QWEN3_TTS_PIPELINE)
