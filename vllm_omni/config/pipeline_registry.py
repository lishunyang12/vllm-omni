# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Central declarative registry of all vllm-omni pipelines.

Mirrors the pattern in ``vllm/model_executor/models/registry.py``: each entry
is ``model_type -> (module_path, variable_name)``, and the module is imported
lazily on first lookup (see ``_LazyPipelineRegistry`` in
``vllm_omni/config/stage_config.py``). Keeping every pipeline declared in one
file makes it easy to spot a missing registration, which was the original
motivation in https://github.com/vllm-project/vllm-omni/issues/2887 (item 4).

Per-model ``pipeline.py`` modules still define the ``PipelineConfig`` instance;
they just no longer need to self-register via ``register_pipeline(...)``.

Adding a new pipeline:
    1. Define the ``PipelineConfig`` instance as a module-level variable in
       ``vllm_omni/.../pipeline.py``.
    2. Add one line to ``_OMNI_PIPELINES`` or ``_DIFFUSION_PIPELINES`` below.

``register_pipeline(config)`` in ``stage_config`` is still supported for
out-of-tree plugins and tests that create pipelines at runtime; those override
the entries declared here.
"""

from __future__ import annotations

# --- Multi-stage omni pipelines (LLM-centric; audio / video I/O) ---
_OMNI_PIPELINES: dict[str, tuple[str, str]] = {
    # model_type -> (module_path, variable_name)
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

# --- Single-stage diffusion pipelines ---
_DIFFUSION_PIPELINES: dict[str, tuple[str, str]] = {
    # model_type -> (module_path, variable_name)
    "flux": (
        "vllm_omni.model_executor.models.flux.pipeline",
        "FLUX_PIPELINE",
    ),
    "flux_kontext": (
        "vllm_omni.model_executor.models.flux_kontext.pipeline",
        "FLUX_KONTEXT_PIPELINE",
    ),
    "flux2": (
        "vllm_omni.model_executor.models.flux2.pipeline",
        "FLUX2_PIPELINE",
    ),
    "flux2_klein": (
        "vllm_omni.model_executor.models.flux2_klein.pipeline",
        "FLUX2_KLEIN_PIPELINE",
    ),
    "qwen_image": (
        "vllm_omni.model_executor.models.qwen_image.pipeline",
        "QWEN_IMAGE_PIPELINE",
    ),
    "qwen_image_edit": (
        "vllm_omni.model_executor.models.qwen_image_edit.pipeline",
        "QWEN_IMAGE_EDIT_PIPELINE",
    ),
    "qwen_image_edit_plus": (
        "vllm_omni.model_executor.models.qwen_image_edit_plus.pipeline",
        "QWEN_IMAGE_EDIT_PLUS_PIPELINE",
    ),
    "qwen_image_layered": (
        "vllm_omni.model_executor.models.qwen_image_layered.pipeline",
        "QWEN_IMAGE_LAYERED_PIPELINE",
    ),
    "z_image": (
        "vllm_omni.model_executor.models.z_image.pipeline",
        "Z_IMAGE_PIPELINE",
    ),
    "ovis_image": (
        "vllm_omni.model_executor.models.ovis_image.pipeline",
        "OVIS_IMAGE_PIPELINE",
    ),
    "longcat_image": (
        "vllm_omni.model_executor.models.longcat_image.pipeline",
        "LONGCAT_IMAGE_PIPELINE",
    ),
    "longcat_image_edit": (
        "vllm_omni.model_executor.models.longcat_image_edit.pipeline",
        "LONGCAT_IMAGE_EDIT_PIPELINE",
    ),
    "sd3": (
        "vllm_omni.model_executor.models.sd3.pipeline",
        "SD3_PIPELINE",
    ),
    "helios": (
        "vllm_omni.model_executor.models.helios.pipeline",
        "HELIOS_PIPELINE",
    ),
    "omnigen2": (
        "vllm_omni.model_executor.models.omnigen2.pipeline",
        "OMNIGEN2_PIPELINE",
    ),
    "nextstep_1_1": (
        "vllm_omni.model_executor.models.nextstep_1_1.pipeline",
        "NEXTSTEP_1_1_PIPELINE",
    ),
}

# Union view used by ``_LazyPipelineRegistry``; don't mutate at runtime.
_VLLM_OMNI_PIPELINES: dict[str, tuple[str, str]] = {
    **_OMNI_PIPELINES,
    **_DIFFUSION_PIPELINES,
}
