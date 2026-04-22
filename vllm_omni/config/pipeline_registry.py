# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Central declarative registry of all vllm-omni pipelines.

Two registration shapes coexist:

* **Multi-stage pipelines** (``_OMNI_PIPELINES``) ship a per-model
  ``pipeline.py`` with custom topology / KV transfer / connectors.
  Entries are ``model_type -> (module_path, variable_name)``; the module
  is imported lazily on first lookup (see ``_LazyPipelineRegistry``).
* **Single-stage diffusion pipelines** (``_DIFFUSION_PIPELINES``) all
  share one topology (one ``DIFFUSION`` stage, ``final_output=True``),
  so the ``PipelineConfig`` is built inline via
  ``_single_stage_diffusion(...)`` instead of duplicating ~20 lines of
  per-model boilerplate. Entries are pre-built ``PipelineConfig`` objects
  — the registry just hands them back at lookup time.

Adding a pipeline:
    * Multi-stage: write ``vllm_omni/.../pipeline.py``, add an
      ``_OMNI_PIPELINES`` entry.
    * Single-stage diffusion: add a ``vllm_omni/deploy/<model>.yaml`` and
      one line to ``_DIFFUSION_PIPELINES`` below.

``register_pipeline(config)`` in ``stage_config`` is still supported for
out-of-tree plugins and tests; dynamic registrations override either
table.
"""

from __future__ import annotations

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# --- Multi-stage omni pipelines (LLM-centric; audio / video I/O) ---
# model_type -> (module_path, variable_name)
_OMNI_PIPELINES: dict[str, tuple[str, str]] = {
    "qwen2_5_omni": (
        "vllm_omni.model_executor.models.qwen2_5_omni.pipeline",
        "QWEN2_5_OMNI_PIPELINE",
    ),
    "qwen2_5_omni_thinker_only": (
        "vllm_omni.model_executor.models.qwen2_5_omni.pipeline",
        "QWEN2_5_OMNI_THINKER_ONLY_PIPELINE",
    ),
    "qwen3_omni_moe": (
        "vllm_omni.model_executor.models.qwen3_omni.pipeline",
        "QWEN3_OMNI_PIPELINE",
    ),
    "qwen3_tts": (
        "vllm_omni.model_executor.models.qwen3_tts.pipeline",
        "QWEN3_TTS_PIPELINE",
    ),
    "glm_image": (
        "vllm_omni.model_executor.models.glm_image.pipeline",
        "GLM_IMAGE_PIPELINE",
    ),
    "voxcpm2": (
        "vllm_omni.model_executor.models.voxcpm2.pipeline",
        "VOXCPM2_PIPELINE",
    ),
    "cosyvoice3": (
        "vllm_omni.model_executor.models.cosyvoice3.pipeline",
        "COSYVOICE3_PIPELINE",
    ),
    "mimo_audio": (
        "vllm_omni.model_executor.models.mimo_audio.pipeline",
        "MIMO_AUDIO_PIPELINE",
    ),
    "voxtral_tts": (
        "vllm_omni.model_executor.models.voxtral_tts.pipeline",
        "VOXTRAL_TTS_PIPELINE",
    ),
    "fish_qwen3_omni": (
        "vllm_omni.model_executor.models.fish_speech.pipeline",
        "FISH_SPEECH_PIPELINE",
    ),
}


def _single_stage_diffusion(model_type: str, model_arch: str, output: str) -> PipelineConfig:
    """Uniform single-stage DIFFUSION topology — every entry in
    ``_DIFFUSION_PIPELINES`` is built from this one helper.
    """
    return PipelineConfig(
        model_type=model_type,
        model_arch=model_arch,
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="dit",
                execution_type=StageExecutionType.DIFFUSION,
                final_output=True,
                final_output_type=output,
                model_arch=model_arch,
            ),
        ),
    )


# --- Single-stage diffusion pipelines (pre-built configs) ---
_DIFFUSION_PIPELINES: dict[str, PipelineConfig] = {
    "flux": _single_stage_diffusion("flux", "FluxPipeline", "image"),
    "flux_kontext": _single_stage_diffusion("flux_kontext", "FluxKontextPipeline", "image"),
    "flux2": _single_stage_diffusion("flux2", "Flux2Pipeline", "image"),
    "flux2_klein": _single_stage_diffusion("flux2_klein", "Flux2KleinPipeline", "image"),
    "qwen_image": _single_stage_diffusion("qwen_image", "QwenImagePipeline", "image"),
    "qwen_image_edit": _single_stage_diffusion("qwen_image_edit", "QwenImageEditPipeline", "image"),
    "qwen_image_edit_plus": _single_stage_diffusion("qwen_image_edit_plus", "QwenImageEditPlusPipeline", "image"),
    "qwen_image_layered": _single_stage_diffusion("qwen_image_layered", "QwenImageLayeredPipeline", "image"),
    "z_image": _single_stage_diffusion("z_image", "ZImagePipeline", "image"),
    "ovis_image": _single_stage_diffusion("ovis_image", "OvisImagePipeline", "image"),
    "longcat_image": _single_stage_diffusion("longcat_image", "LongCatImagePipeline", "image"),
    "longcat_image_edit": _single_stage_diffusion("longcat_image_edit", "LongCatImageEditPipeline", "image"),
    "sd3": _single_stage_diffusion("sd3", "StableDiffusion3Pipeline", "image"),
    "helios": _single_stage_diffusion("helios", "HeliosPipeline", "image"),
    "omnigen2": _single_stage_diffusion("omnigen2", "OmniGen2Pipeline", "image"),
    "nextstep_1_1": _single_stage_diffusion("nextstep_1_1", "NextStep11Pipeline", "image"),
}
