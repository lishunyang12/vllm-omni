# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HunyuanImage-3.0 pipeline topologies (frozen).

Five variants share one HF model_arch (HunyuanImage3ForCausalMM) but expose
different stage graphs:

    t2i      Stage 0 AR (text → latent) + Stage 1 DiT, KV-transfer
    it2i     Stage 0 AR (image+text → latent) + Stage 1 DiT, KV-transfer
    dit_only Stage 0 DiT only (latent → image)
    i2t      Stage 0 AR only (image+text → text)
    t2t      Stage 0 AR only (text → text)

Variants are surfaced as separate model_types so the orchestrator picks the
right topology from deploy YAML alone (mirrors the qwen2_5_omni /
qwen2_5_omni_thinker_only split). Only ``t2i`` and ``it2i`` ship with a
default deploy yaml; ``dit_only`` / ``i2t`` / ``t2t`` are registered for
tests and bring-your-own deploy.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_MODEL_ARCH = "HunyuanImage3ForCausalMM"
_AR2DIT = "vllm_omni.model_executor.stage_input_processors.hunyuan_image3.ar2diffusion"

# AR-side KV transfer config: send cache once prefill is done so the DiT stage
# can splice it in without re-running attention over the prompt.
_AR_KV_SEND = {
    "need_send_cache": True,
    "kv_transfer_criteria": {"type": "prefill_finished"},
}
_DIT_KV_RECV = {"need_recv_cache": True}


# Only one variant carries the hf_architectures fallback so the deploy yaml's
# explicit ``pipeline:`` field stays the single source of truth for variant
# selection. T2I is the default because it's the headline modality.
HUNYUAN_IMAGE3_T2I_PIPELINE = PipelineConfig(
    model_type="hunyuan_image3_t2i",
    model_arch=_MODEL_ARCH,
    hf_architectures=("HunyuanImage3ForCausalMM",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
            omni_kv_config=_AR_KV_SEND,
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            engine_output_type="image",
            custom_process_input_func=_AR2DIT,
            omni_kv_config=_DIT_KV_RECV,
        ),
    ),
)


HUNYUAN_IMAGE3_IT2I_PIPELINE = PipelineConfig(
    model_type="hunyuan_image3_it2i",
    model_arch=_MODEL_ARCH,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            sampling_constraints={"stop_token_ids": [127957], "detokenize": False},
            omni_kv_config=_AR_KV_SEND,
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            engine_output_type="image",
            requires_multimodal_data=True,
            custom_process_input_func=_AR2DIT,
            omni_kv_config=_DIT_KV_RECV,
        ),
    ),
)


HUNYUAN_IMAGE3_DIT_ONLY_PIPELINE = PipelineConfig(
    model_type="hunyuan_image3_dit_only",
    model_arch=_MODEL_ARCH,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            engine_output_type="image",
            omni_kv_config=_DIT_KV_RECV,
        ),
    ),
)


# AR-only variants (no DiT). Registered for users bringing their own deploy
# yaml — no default deploy yaml ships because hardware sizing for I2T/T2T
# depends on the use case.
HUNYUAN_IMAGE3_I2T_PIPELINE = PipelineConfig(
    model_type="hunyuan_image3_i2t",
    model_arch=_MODEL_ARCH,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            sampling_constraints={"stop_token_ids": [127957, 128026], "detokenize": True},
        ),
    ),
)


HUNYUAN_IMAGE3_T2T_PIPELINE = PipelineConfig(
    model_type="hunyuan_image3_t2t",
    model_arch=_MODEL_ARCH,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            sampling_constraints={"stop_token_ids": [127957, 128026], "detokenize": True},
        ),
    ),
)
