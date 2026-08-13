# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the shared LTX pipeline runtime and public contracts."""

import json
import os
import tempfile
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.ltx2 import ltx2_components, ltx2_latents, ltx2_runtime
from vllm_omni.diffusion.models.ltx2.ltx2_components import (
    LTX2_COMPONENT_PROFILE,
    LTX2_DISTILLED_COMPONENT_PROFILE,
    LTX23_COMPONENT_PROFILE,
    LTX25_DISTILLED_COMPONENT_PROFILE,
    LTX25_FULL_COMPONENT_PROFILE,
    LTX25_FULL_TWO_STAGE_COMPONENT_PROFILE,
    _install_connector_attention,
    detect_ltx_model_version,
    resolve_ltx_checkpoint_variant,
)
from vllm_omni.diffusion.models.ltx2.ltx2_conditioning import LTXI2VConditioningMixin
from vllm_omni.diffusion.models.ltx2.ltx2_denoise import (
    LTXDenoiseExecutor,
    LTXPhaseResult,
    _first_frame_keyframes_mask,
    _official_ltx_sigmas,
    prepare_scheduler_stage,
)
from vllm_omni.diffusion.models.ltx2.ltx2_guidance import (
    LTX_GUIDANCE_EXECUTOR,
    LTXGuidancePlan,
    LTXGuidanceSpec,
    LTXModalityGuidance,
    build_perturbation_kwargs,
    combine_guided_x0,
    velocity_from_x0,
    x0_from_velocity,
)
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import (
    LTX2_DISTILLED_TWO_STAGE_RECIPE,
    LTX2_ONE_STAGE_RECIPE,
    LTX23_ONE_STAGE_RECIPE,
    LTX25_DISTILLED_TWO_STAGE_RECIPE,
    LTX25_FULL_RECIPE,
    LTX25_FULL_TWO_STAGE_RECIPE,
    LTX_DEFAULT_NEGATIVE_PROMPT,
    LTX_DISTILLED_SIGMAS,
    LTX_POSITIVE_ONLY_RECIPE,
)
from vllm_omni.diffusion.models.ltx2.ltx2_request import LTXRequestInputs, validate_pipeline_request
from vllm_omni.diffusion.models.ltx2.ltx2_runtime import LTXRuntime
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import (
    LTX2FullPipeline,
    LTX2I2VDMD2Pipeline,
    LTX2Pipeline,
    LTX2T2VDMD2Pipeline,
)
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_two_stage import (
    LTX2DistilledPipeline,
    LTX2TwoStagePipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("model_version", ["2", "2.3", "2.5"])
def test_ltx_timestep_adaln_expands_per_token_like_official_without_sp(model_version):
    pipe = object.__new__(LTX2Pipeline)
    pipe.model_version = model_version
    pipe.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1))
    ts = torch.tensor([750.0, 500.0])

    kwargs = LTXRuntime._denoise_timestep_kwargs(
        pipe,
        ts,
        SimpleNamespace(),
        SimpleNamespace(),
        video_token_count=4,
        audio_token_count=3,
    )

    torch.testing.assert_close(kwargs["timestep"], ts[:, None].expand(-1, 4))
    torch.testing.assert_close(kwargs["audio_timestep"], ts[:, None].expand(-1, 3))
    assert kwargs["sigma"] is ts
    assert kwargs["audio_sigma"] is ts


def test_ltx_ancestral_step_clears_sp_audio_padding_after_scheduler_update(monkeypatch):
    state = LTXAVState(
        video=torch.zeros(1, 2, 2),
        audio=torch.zeros(1, 4, 2),
    )
    forward_ctx = SimpleNamespace(
        original_audio_num_frames=3,
        video_audio_step_adapter=SimpleNamespace(_sampler="euler_ancestral"),
    )
    denoise_ctx = SimpleNamespace(latents=None, audio_latents=None)
    pipe = SimpleNamespace(
        _predict_noise_for_step=lambda *_args: (torch.zeros_like(state.video), torch.zeros_like(state.audio))
    )

    def fake_step(_pipeline, actual_forward_ctx, _denoise_ctx, *_args):
        assert actual_forward_ctx.video_audio_step_adapter._sampler == "euler_ancestral"
        updated_audio = torch.ones_like(state.audio)
        updated_audio[:, 3:] = 99.0
        return torch.ones_like(state.video), updated_audio

    monkeypatch.setattr(ltx2_runtime, "step_denoised_latents", fake_step)

    actual = LTXRuntime._denoise_step(
        pipe,
        0,
        torch.tensor(1.0),
        state,
        forward_ctx,
        denoise_ctx,
    )

    torch.testing.assert_close(actual.audio[:, :3], torch.ones_like(actual.audio[:, :3]))
    torch.testing.assert_close(actual.audio[:, 3:], torch.zeros_like(actual.audio[:, 3:]))


def test_ltx25_first_causal_latent_frame_is_marked_for_keyframe_embedding():
    reference = torch.zeros(2, 12, 8)

    mask = _first_frame_keyframes_mask(reference, latent_num_frames=3)

    assert mask.shape == (2, 12, 1)
    torch.testing.assert_close(mask[:, :4], torch.ones(2, 4, 1))
    torch.testing.assert_close(mask[:, 4:], torch.zeros(2, 8, 1))


def _make_ltx_request_pipe(cls):
    pipe = object.__new__(cls)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.tokenizer_max_length = 99
    pipe.vae_spatial_compression_ratio = 32
    pipe.vae_temporal_compression_ratio = 8
    return pipe


def _resolve_request_inputs_for_test(
    pipe,
    req,
    *,
    guidance_scale=4.0,
    negative_prompt=None,
):
    return pipe._resolve_request_inputs(
        req,
        prompt=None,
        negative_prompt=negative_prompt,
        height=None,
        width=None,
        num_frames=None,
        frame_rate=None,
        num_inference_steps=None,
        guidance_scale=guidance_scale,
        num_videos_per_prompt=1,
        generator=None,
        latents=None,
        audio_latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        decode_timestep=0.0,
        decode_noise_scale=None,
        output_type="np",
        max_sequence_length=None,
    )


def test_ltx_public_entries_share_runtime_and_keep_recipe_boundaries():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
        DistributedAutoencoderKLLTX2Video,
    )

    assert issubclass(LTX2Pipeline, LTXRuntime)
    assert issubclass(LTX2FullPipeline, LTXRuntime)
    assert issubclass(LTX2DistilledPipeline, LTXRuntime)
    assert issubclass(LTX2TwoStagePipeline, LTXRuntime)
    assert LTX2Pipeline.component_profile is LTX2_COMPONENT_PROFILE
    assert issubclass(LTX2DistilledPipeline, LTX2TwoStagePipeline)
    assert LTX2Pipeline.pipeline_recipe is LTX2_ONE_STAGE_RECIPE
    assert LTX2FullPipeline.component_profile is LTX25_FULL_COMPONENT_PROFILE
    assert LTX2FullPipeline.pipeline_recipe is LTX25_FULL_RECIPE
    assert LTX2TwoStagePipeline.component_profile is LTX2_DISTILLED_COMPONENT_PROFILE
    assert LTX2_COMPONENT_PROFILE.video_vae_cls is DistributedAutoencoderKLLTX2Video
    assert LTX23_COMPONENT_PROFILE.video_vae_cls is DistributedAutoencoderKLLTX2Video
    assert LTX25_FULL_COMPONENT_PROFILE.text_encoder_cls is ltx2_components.AutoModelForImageTextToText
    assert LTX25_DISTILLED_COMPONENT_PROFILE.text_encoder_cls is ltx2_components.AutoModelForImageTextToText


def test_ltx25_missing_gemma4_recommends_supported_transformers_range(monkeypatch):
    pipe = SimpleNamespace(component_profile=replace(LTX25_FULL_COMPONENT_PROFILE, text_encoder_cls=None))
    od_config = SimpleNamespace(model="Lightricks/LTX-2.5-Diffusers", dtype=torch.bfloat16)

    monkeypatch.setattr(ltx2_components, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ltx2_components, "prefetch_subfolders", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ltx2_components.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object())

    with pytest.raises(ImportError) as exc_info:
        ltx2_components.initialize_pipeline_components(pipe, od_config)

    assert str(exc_info.value) == (
        "LTX-2.5 requires Gemma4UnifiedForConditionalGeneration. Install transformers>=5.10.1,<5.15."
    )


def test_ltx_converted_component_loading_propagates_revision(monkeypatch):
    revision = "pinned-revision"
    calls = {"components": []}
    profile = replace(
        LTX25_FULL_TWO_STAGE_COMPONENT_PROFILE,
        text_encoder_cls=object,
        video_vae_cls=object,
        vocoder_cls=object,
        vocoder_fallback_cls=None,
        scheduler_use_dynamic_shifting=False,
    )
    pipeline = SimpleNamespace(component_profile=profile)
    od_config = SimpleNamespace(
        model="org/converted-ltx25",
        revision=revision,
        dtype=torch.bfloat16,
        quantization_config=None,
    )

    monkeypatch.setattr(ltx2_components, "is_ltx25_raw_checkpoint", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(ltx2_components, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ltx2_components, "_install_connector_attention", lambda _connectors: None)
    monkeypatch.setattr(ltx2_components, "_place_aux_components", lambda _pipeline: None)

    def fake_prefetch(model, subfolders, **kwargs):
        calls["prefetch"] = (model, tuple(subfolders), kwargs)

    def fake_tokenizer(*args, **kwargs):
        calls["tokenizer"] = (args, kwargs)
        return SimpleNamespace(model_max_length=1024)

    def fake_component(component_cls, model, subfolder, **kwargs):
        calls["components"].append((component_cls, model, subfolder, kwargs))
        if subfolder == "vae":
            return SimpleNamespace(spatial_compression_ratio=32, temporal_compression_ratio=8)
        if subfolder == "audio_vae":
            return SimpleNamespace(
                mel_compression_ratio=4,
                temporal_compression_ratio=4,
                config=SimpleNamespace(sample_rate=16_000, mel_hop_length=160),
            )
        return object()

    def fake_transformer_config(model, subfolder, local_files_only, *, revision):
        calls["transformer_config"] = (model, subfolder, local_files_only, revision)
        return {"component": "transformer"}

    monkeypatch.setattr(ltx2_components, "prefetch_subfolders", fake_prefetch)
    monkeypatch.setattr(ltx2_components.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(ltx2_components, "_load_component", fake_component)
    monkeypatch.setattr(ltx2_components, "load_transformer_config", fake_transformer_config)
    monkeypatch.setattr(
        ltx2_components,
        "create_transformer_from_config",
        lambda *_args, **_kwargs: SimpleNamespace(config=SimpleNamespace(patch_size=1, patch_size_t=1)),
    )

    def fake_scheduler(*args, **kwargs):
        calls["scheduler"] = (args, kwargs)
        return object()

    monkeypatch.setattr(ltx2_components.FlowMatchEulerDiscreteScheduler, "from_pretrained", fake_scheduler)

    ltx2_components.initialize_pipeline_components(pipeline, od_config)

    assert pipeline.weights_sources[0].revision == revision
    assert calls["prefetch"][2]["revision"] == revision
    assert calls["tokenizer"][1]["revision"] == revision
    assert {call[2] for call in calls["components"]} == {
        "text_encoder",
        "connectors",
        "vae",
        "audio_vae",
        "vocoder",
        "latent_upsampler",
    }
    assert all(call[3]["revision"] == revision for call in calls["components"])
    assert calls["transformer_config"] == (
        od_config.model,
        profile.transformer_subfolder,
        False,
        revision,
    )
    assert calls["scheduler"][1]["revision"] == revision


def test_ltx_checkpoint_version_detection_uses_metadata(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"]})
    )
    assert detect_ltx_model_version(str(tmp_path)) == "2.5"

    (tmp_path / "model_index.json").write_text(json.dumps({"vocoder": ["ltx2", "LTX2VocoderWithBWE"]}))
    assert detect_ltx_model_version(str(tmp_path)) == "2.3"

    (tmp_path / "model_index.json").write_text(json.dumps({"model_version": "2.3.1"}))
    assert detect_ltx_model_version(str(tmp_path)) == "2.3"

    (tmp_path / "model_index.json").write_text(json.dumps({"vocoder": ["ltx2", "LTX2Vocoder"]}))
    assert detect_ltx_model_version(str(tmp_path)) == "2"


def test_ltx_checkpoint_explicit_version_precedes_structural_heuristics(tmp_path):
    (tmp_path / "model_index.json").write_text(json.dumps({"model_version": "2.3.1"}))
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps({"ff_bias": False}))

    assert detect_ltx_model_version(str(tmp_path)) == "2.3"


def test_ltx25_checkpoint_selects_full_one_stage_profile(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    (tmp_path / "model_index.json").write_text(
        json.dumps({"text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"]})
    )

    def stub_components(pipe, od_config):
        pipe.od_config = od_config
        pipe.vae_spatial_compression_ratio = 32

    monkeypatch.setattr(ltx2_runtime, "initialize_pipeline_components", stub_components)
    monkeypatch.setattr(LTXRuntime, "setup_diffusion_pipeline_profiler", lambda *_args, **_kwargs: None)

    pipe = LTX2Pipeline(od_config=SimpleNamespace(model=str(tmp_path), enable_diffusion_pipeline_profiler=False))

    assert pipe.model_version == "2.5"
    assert pipe.checkpoint_variant == "full"
    assert pipe.component_profile is LTX25_FULL_COMPONENT_PROFILE
    assert pipe.pipeline_recipe is LTX25_FULL_RECIPE
    assert pipe.preserve_sp_padded_audio_duration
    assert pipe.reports_stage_durations


def test_ltx25_full_checkpoint_selects_sft_profile_and_recipe(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    (tmp_path / "model_index.json").write_text(
        json.dumps({"text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"]})
    )

    def stub_components(pipe, od_config):
        pipe.od_config = od_config
        pipe.vae_spatial_compression_ratio = 32

    monkeypatch.setattr(ltx2_runtime, "initialize_pipeline_components", stub_components)
    monkeypatch.setattr(LTXRuntime, "setup_diffusion_pipeline_profiler", lambda *_args, **_kwargs: None)

    pipe = LTX2FullPipeline(od_config=SimpleNamespace(model=str(tmp_path), enable_diffusion_pipeline_profiler=False))

    assert pipe.model_version == "2.5"
    assert pipe.component_profile is LTX25_FULL_COMPONENT_PROFILE
    assert pipe.pipeline_recipe is LTX25_FULL_RECIPE
    assert pipe.support_image_input
    assert pipe.unified_text_image_entry


def test_ltx25_full_recipe_matches_official_sft_defaults():
    (phase,) = LTX25_FULL_RECIPE.phases

    assert (LTX25_FULL_RECIPE.height, LTX25_FULL_RECIPE.width) == (544, 960)
    assert LTX25_FULL_RECIPE.num_frames == 121
    assert LTX25_FULL_RECIPE.frame_rate == 24.0
    assert LTX25_FULL_RECIPE.num_inference_steps == 30
    assert phase.guidance.video == LTXModalityGuidance(
        cfg_scale=3.0,
        stg_scale=1.0,
        modality_scale=3.0,
        rescale_scale=0.7,
        stg_blocks=(28,),
    )
    assert phase.guidance.audio == LTXModalityGuidance(
        cfg_scale=7.0,
        stg_scale=1.0,
        modality_scale=3.0,
        rescale_scale=0.7,
        stg_blocks=(28,),
    )
    assert phase.sigmas is None
    assert phase.use_official_sigma_schedule
    assert LTX25_FULL_COMPONENT_PROFILE.transformer_subfolder == "transformer_full"
    assert LTX25_FULL_COMPONENT_PROFILE.scheduler_use_dynamic_shifting
    assert LTX25_FULL_COMPONENT_PROFILE.scheduler_shift_terminal == 0.1


def test_ltx25_full_sigma_schedule_uses_official_default_sequence_anchor():
    scheduler = SimpleNamespace(
        config={
            "base_image_seq_len": 1024,
            "max_image_seq_len": 4096,
            "base_shift": 0.95,
            "max_shift": 2.05,
            "shift_terminal": 0.1,
        }
    )

    sigmas = _official_ltx_sigmas(
        scheduler,
        steps=30,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        sigmas[[0, 1, -2, -1]],
        torch.tensor([1.0, 0.9949570, 0.1, 0.0]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_ltx25_checkpoint_selects_distilled_two_stage_profile(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    (tmp_path / "model_index.json").write_text(
        json.dumps({"text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"]})
    )

    def stub_components(pipe, od_config):
        pipe.od_config = od_config
        pipe.vae_spatial_compression_ratio = 32

    monkeypatch.setattr(ltx2_runtime, "initialize_pipeline_components", stub_components)
    monkeypatch.setattr(LTXRuntime, "setup_diffusion_pipeline_profiler", lambda *_args, **_kwargs: None)

    pipe = LTX2TwoStagePipeline(
        od_config=SimpleNamespace(
            model=str(tmp_path),
            task_type="distilled",
            enable_diffusion_pipeline_profiler=False,
        )
    )

    assert pipe.checkpoint_variant == "distilled"
    assert pipe.component_profile is LTX25_DISTILLED_COMPONENT_PROFILE
    assert pipe.pipeline_recipe is LTX25_DISTILLED_TWO_STAGE_RECIPE


def test_ltx_checkpoint_variant_is_independent_from_topology():
    assert resolve_ltx_checkpoint_variant("one_stage", "2.5", "full") == "full"
    assert resolve_ltx_checkpoint_variant("one_stage", "2.5", "dev") == "full"
    assert resolve_ltx_checkpoint_variant("two_stage", "2.5", "distilled") == "distilled"
    assert resolve_ltx_checkpoint_variant("two_stage", "2.5", "full") == "full"

    with pytest.raises(ValueError, match="does not define a distilled one-stage"):
        resolve_ltx_checkpoint_variant("one_stage", "2.5", "distilled")
    with pytest.raises(ValueError, match="compatibility entry requires 'full'"):
        resolve_ltx_checkpoint_variant(
            "one_stage",
            "2.5",
            "distilled",
            compatibility_override="full",
        )


def test_ltx25_checkpoint_selects_full_two_stage_profile(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    (tmp_path / "model_index.json").write_text(
        json.dumps({"text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"]})
    )

    def stub_components(pipe, od_config):
        pipe.od_config = od_config
        pipe.vae_spatial_compression_ratio = 32

    monkeypatch.setattr(ltx2_runtime, "initialize_pipeline_components", stub_components)
    monkeypatch.setattr(LTXRuntime, "setup_diffusion_pipeline_profiler", lambda *_args, **_kwargs: None)

    pipe = LTX2TwoStagePipeline(
        od_config=SimpleNamespace(
            model=str(tmp_path),
            task_type="full",
            enable_diffusion_pipeline_profiler=False,
        )
    )

    assert pipe.checkpoint_variant == "full"
    assert pipe.component_profile is LTX25_FULL_TWO_STAGE_COMPONENT_PROFILE
    assert pipe.pipeline_recipe is LTX25_FULL_TWO_STAGE_RECIPE


def test_ltx25_full_two_stage_recipe_matches_official_handoff():
    stage1, stage2 = LTX25_FULL_TWO_STAGE_RECIPE.phases

    assert (LTX25_FULL_TWO_STAGE_RECIPE.height, LTX25_FULL_TWO_STAGE_RECIPE.width) == (1088, 1920)
    assert LTX25_FULL_TWO_STAGE_RECIPE.num_inference_steps == 30
    assert stage1.spatial_downscale == 2
    assert stage1.guidance.do_cfg
    assert stage1.sigmas is None
    assert stage1.use_official_sigma_schedule
    assert stage2.guidance == LTXGuidanceSpec.positive_only()
    assert stage2.sigmas == (0.909375, 0.725, 0.421875, 0.0)
    assert stage2.input_transform == "spatial_upsample"
    assert stage2.resident_adapter == "distilled_lora_450"
    assert LTX25_FULL_TWO_STAGE_RECIPE.video_output_phase == 1
    assert LTX25_FULL_TWO_STAGE_RECIPE.audio_output_phase == 0


def test_ltx25_distilled_two_stage_recipe_matches_model_card_resolution():
    stage1, stage2 = LTX25_DISTILLED_TWO_STAGE_RECIPE.phases

    assert (LTX25_DISTILLED_TWO_STAGE_RECIPE.height, LTX25_DISTILLED_TWO_STAGE_RECIPE.width) == (1088, 1920)
    assert stage1.spatial_downscale == 2
    assert stage1.sigmas == LTX_DISTILLED_SIGMAS
    assert stage1.sampler == "euler_ancestral"
    assert stage2.sampler == "euler"
    assert stage2.sigmas == (0.909375, 0.725, 0.421875, 0.0)
    assert stage2.input_transform == "spatial_upsample"

    assert LTX25_DISTILLED_TWO_STAGE_RECIPE.negative_prompt == ""
    assert not LTX25_DISTILLED_TWO_STAGE_RECIPE.allow_negative_prompt


@pytest.mark.parametrize(
    "recipe",
    [
        LTX2_ONE_STAGE_RECIPE,
        LTX23_ONE_STAGE_RECIPE,
        LTX25_FULL_RECIPE,
        LTX_POSITIVE_ONLY_RECIPE,
    ],
)
def test_ltx_one_stage_recipes_declare_cache_dit_supported(recipe):
    assert recipe.supports_cache_dit


@pytest.mark.parametrize(
    "recipe",
    [
        LTX2_DISTILLED_TWO_STAGE_RECIPE,
        LTX25_DISTILLED_TWO_STAGE_RECIPE,
        LTX25_FULL_TWO_STAGE_RECIPE,
    ],
)
def test_ltx_multistage_recipes_declare_cache_dit_unsupported(recipe):
    assert not recipe.supports_cache_dit


@pytest.mark.parametrize(
    "model_index",
    [
        {},
        {"model_version": "2.5"},
    ],
    ids=["ltx2", "ltx25"],
)
def test_ltx_multistage_rejects_cache_dit_before_component_initialization(tmp_path, monkeypatch, model_index):
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    (tmp_path / "model_index.json").write_text(json.dumps(model_index))
    initialized = False

    def stub_components(*_args, **_kwargs):
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(ltx2_runtime, "initialize_pipeline_components", stub_components)

    with pytest.raises(ValueError, match="only one-stage LTX recipes"):
        LTX2TwoStagePipeline(od_config=SimpleNamespace(model=str(tmp_path), cache_backend="cache_dit"))

    assert not initialized


@pytest.mark.parametrize("pipeline_cls", [LTX2Pipeline, LTX2TwoStagePipeline])
def test_ltx_rejects_advanced_uaa_before_component_initialization(pipeline_cls):
    od_config = SimpleNamespace(
        model="unused",
        parallel_config=SimpleNamespace(ulysses_mode="advanced_uaa"),
    )

    with pytest.raises(ValueError, match="does not support ulysses_mode='advanced_uaa'"):
        pipeline_cls(od_config=od_config)


def test_ltx23_checkpoint_selects_version_specific_one_stage_profile(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    (tmp_path / "model_index.json").write_text(json.dumps({"vocoder": ["ltx2", "LTX2VocoderWithBWE"]}))

    def stub_components(pipe, od_config):
        pipe.od_config = od_config
        pipe.vae_spatial_compression_ratio = 32

    monkeypatch.setattr(ltx2_runtime, "initialize_pipeline_components", stub_components)
    monkeypatch.setattr(LTXRuntime, "setup_diffusion_pipeline_profiler", lambda *_args, **_kwargs: None)

    pipe = LTX2Pipeline(od_config=SimpleNamespace(model=str(tmp_path), enable_diffusion_pipeline_profiler=False))

    assert pipe.model_version == "2.3"
    assert pipe.component_profile is LTX23_COMPONENT_PROFILE
    assert pipe.pipeline_recipe is LTX23_ONE_STAGE_RECIPE
    assert pipe.preserve_sp_padded_audio_duration
    assert pipe.reports_stage_durations


def test_ltx_one_stage_entry_exposes_batch_image_and_distributed_decode_contracts():
    assert LTX2Pipeline.supports_request_batch
    assert LTX2Pipeline.support_image_input
    assert LTX2Pipeline.unified_text_image_entry
    assert LTX2Pipeline.distributed_video_decode
    assert not LTX2TwoStagePipeline.supports_request_batch
    assert LTX2TwoStagePipeline.support_image_input
    assert LTX2TwoStagePipeline.unified_text_image_entry


def test_ltx_dmd2_entries_retain_their_task_boundaries():
    assert not LTX2T2VDMD2Pipeline.support_image_input
    assert LTX2I2VDMD2Pipeline.support_image_input
    assert not LTX2I2VDMD2Pipeline.unified_text_image_entry


def test_ltx_request_batch_decode_splits_video_and_audio_per_request():
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    requests = [
        OmniDiffusionRequest(
            prompt=f"prompt-{index}",
            sampling_params=OmniDiffusionSamplingParams(),
            request_id=f"request-{index}",
        )
        for index in range(2)
    ]
    request_batch = DiffusionRequestBatch(requests)
    video = torch.tensor([[1.0], [2.0]])
    audio = torch.tensor([[3.0], [4.0]])
    pipe = object.__new__(LTX2Pipeline)
    torch.nn.Module.__init__(pipe)
    object.__setattr__(pipe, "_decode_output", lambda **_kwargs: DiffusionOutput(output=(video, audio)))
    forward_ctx = SimpleNamespace(
        req=request_batch,
        request_inputs=SimpleNamespace(
            output_type="np",
            generator=None,
            decode_timestep=0.0,
            decode_noise_scale=None,
        ),
        prompt_context=SimpleNamespace(connector_prompt_embeds=torch.empty(2, 1)),
        device=torch.device("cpu"),
        batch_size=2,
        num_videos_per_prompt=1,
    )

    outputs = pipe.decode_phase(
        LTXPhaseResult(
            forward_context=forward_ctx,
            video=video,
            audio=audio,
        )
    )

    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0].output[0], video[:1])
    torch.testing.assert_close(outputs[0].output[1], audio[:1])
    torch.testing.assert_close(outputs[1].output[0], video[1:])
    torch.testing.assert_close(outputs[1].output[1], audio[1:])


def test_ltx_public_entries_share_runtime_forward_template():
    assert "_forward_request" not in LTX2Pipeline.__dict__
    assert "forward" not in LTX2Pipeline.__dict__
    assert LTX2Pipeline._forward_request is LTXRuntime._forward_request
    assert LTX2Pipeline._run_recipe is LTXRuntime._run_recipe


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"timesteps": [1]}, "timesteps"),
        ({"guidance_rescale": 0.5}, "guidance_rescale"),
        ({"noise_scale": 1.0}, "noise_scale"),
        ({"return_dict": False}, "return_dict"),
        ({"attention_kwargs": {}}, "attention_kwargs"),
    ],
)
def test_ltx_public_forward_rejects_unsupported_generic_options(kwargs, field_name):
    pipe = object.__new__(LTX2Pipeline)

    with pytest.raises(ValueError, match=field_name):
        pipe.forward(SimpleNamespace(), **kwargs)


def test_ltx2_canonical_forward_requires_keyword_arguments():
    pipe = object.__new__(LTX2Pipeline)

    with pytest.raises(TypeError):
        pipe.forward(SimpleNamespace(), "prompt")


def test_ltx_versions_share_request_prompt_and_step_templates():
    shared_methods = (
        "_get_gemma_prompt_embeds",
        "encode_prompt",
        "check_inputs",
        "_resolve_request_inputs",
        "_prepare_prompt_context",
        "_denoise_step",
    )
    for method_name in shared_methods:
        base_method = getattr(LTXRuntime, method_name)
        assert getattr(LTX2Pipeline, method_name) is base_method


def test_ltx_versions_select_guidance_without_overriding_control_flow():
    assert LTX2Pipeline.guidance_executor is LTX_GUIDANCE_EXECUTOR
    assert LTX2Pipeline._predict_noise_for_step is LTXRuntime._predict_noise_for_step
    assert LTX2Pipeline.combine_cfg_noise is LTXRuntime.combine_cfg_noise


def test_ltx_official_recipes_build_all_guidance_passes():
    ltx2_plan = LTXGuidancePlan.build(LTX2_ONE_STAGE_RECIPE.request_guidance)
    ltx23_plan = LTXGuidancePlan.build(LTX23_ONE_STAGE_RECIPE.request_guidance)

    assert ltx2_plan.names == ltx23_plan.names == ("cond", "uncond", "ptb", "mod")
    assert LTX2_ONE_STAGE_RECIPE.request_guidance.video.stg_blocks == (29,)
    assert LTX23_ONE_STAGE_RECIPE.request_guidance.video.stg_blocks == (28,)
    assert LTX23_ONE_STAGE_RECIPE.request_guidance.video.cfg_scale == 3.0
    assert LTX23_ONE_STAGE_RECIPE.request_guidance.audio.cfg_scale == 7.0
    assert LTX2_ONE_STAGE_RECIPE.phases[0].use_official_sigma_schedule
    assert LTX23_ONE_STAGE_RECIPE.phases[0].use_official_sigma_schedule
    assert not LTX_POSITIVE_ONLY_RECIPE.phases[0].use_official_sigma_schedule


def test_ltx_guidance_combines_official_x0_deltas_in_fp32():
    cond = torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16)
    uncond = torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16)
    perturbed = torch.tensor([[0.5, 2.0]], dtype=torch.bfloat16)
    isolated = torch.tensor([[0.25, 2.5]], dtype=torch.bfloat16)
    guidance = LTXModalityGuidance(
        cfg_scale=3.0,
        stg_scale=0.5,
        modality_scale=2.0,
        stg_blocks=(0,),
    )

    actual = combine_guided_x0(
        cond=cond,
        uncond_text=uncond,
        uncond_perturbed=perturbed,
        uncond_modality=isolated,
        guidance=guidance,
    )
    expected = (
        cond.float()
        + 2.0 * (cond.float() - uncond.float())
        + 0.5 * (cond.float() - perturbed.float())
        + (cond.float() - isolated.float())
    ).to(cond.dtype)

    torch.testing.assert_close(actual, expected)


def test_ltx_positive_only_guidance_preserves_official_x0_roundtrip():
    sample = torch.full((1, 1), 1e8, dtype=torch.float32)
    velocity = torch.ones((1, 1), dtype=torch.bfloat16)
    sigma = torch.tensor(0.5)

    actual = LTX_GUIDANCE_EXECUTOR._guide_modality(
        sample,
        {"cond": velocity},
        sigma,
        LTXModalityGuidance(),
    )
    expected = velocity_from_x0(sample, x0_from_velocity(sample, velocity, sigma), sigma)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_ltx_velocity_from_x0_materializes_official_scalar_sigma():
    item_calls = []

    class TrackingTensor(torch.Tensor):
        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.tensor(0.725), False)

        def item(self):
            item_calls.append(True)
            return super().item()

    sample = torch.tensor([[1.0]], dtype=torch.bfloat16)
    x0 = torch.tensor([[0.5]], dtype=torch.bfloat16)

    velocity_from_x0(sample, x0, TrackingTensor())

    assert item_calls == [True]


def test_ltx_ancestral_positive_only_guidance_preserves_raw_velocity():
    sample = torch.full((1, 1), 1e8, dtype=torch.float32)
    velocity = torch.ones((1, 1), dtype=torch.bfloat16)

    actual = LTX_GUIDANCE_EXECUTOR._guide_modality(
        sample,
        {"cond": velocity},
        torch.tensor(0.5),
        LTXModalityGuidance(),
        preserve_positive_velocity=True,
    )

    assert actual is velocity


@pytest.mark.parametrize("model_version", ["2.5", "2.3"])
def test_ltx_t2v_denoise_state_uses_official_bfloat16_dtype(model_version):
    expected_dtype = torch.bfloat16
    captured = {}

    def prepare_latents(**kwargs):
        captured["video_dtype"] = kwargs["dtype"]
        return torch.empty(1, 1, 1, dtype=kwargs["dtype"])

    def prepare_audio_latents(*_args, **kwargs):
        captured["audio_dtype"] = kwargs["dtype"]
        return torch.empty(1, 1, 1, dtype=kwargs["dtype"]), 1, 1

    pipeline = SimpleNamespace(
        model_version=model_version,
        transformer=SimpleNamespace(config=SimpleNamespace(in_channels=128)),
        prepare_latents=prepare_latents,
        prepare_audio_latents=prepare_audio_latents,
        audio_sampling_rate=48_000,
        audio_hop_length=160,
        audio_vae_temporal_compression_ratio=4,
        audio_vae_mel_compression_ratio=4,
        audio_vae=SimpleNamespace(config=SimpleNamespace(mel_bins=64, latent_channels=8)),
        _resolve_audio_latent_length=lambda length, _latents: length,
    )
    request = SimpleNamespace(
        num_videos_per_prompt=1,
        height=64,
        width=64,
        num_frames=1,
        frame_rate=24.0,
        generator=None,
        latents=None,
        audio_latents=None,
        audio_latents_normalized=False,
    )
    prompt = SimpleNamespace(
        batch_size=1,
        positive_connector_prompt_embeds=torch.empty(1, dtype=torch.bfloat16),
        positive_connector_audio_prompt_embeds=torch.empty(1, dtype=torch.bfloat16),
    )

    LTXRuntime._prepare_video_latents_stage(
        pipeline,
        request,
        prompt,
        device=torch.device("cpu"),
        noise_scale=1.0,
    )
    LTXRuntime._prepare_audio_latents_stage(
        pipeline,
        request,
        prompt,
        device=torch.device("cpu"),
        noise_scale=1.0,
    )

    assert captured == {"video_dtype": expected_dtype, "audio_dtype": expected_dtype}


def test_ltx_guidance_skips_stg_without_perturbed_blocks():
    cond = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    guidance = LTXModalityGuidance(stg_scale=1.0, stg_blocks=())

    plan = LTXGuidancePlan.build(LTXGuidanceSpec(video=guidance))
    actual = combine_guided_x0(
        cond=cond,
        uncond_text=0.0,
        uncond_perturbed=0.0,
        uncond_modality=0.0,
        guidance=guidance,
    )

    assert plan.names == ("cond",)
    torch.testing.assert_close(actual, cond)


def test_ltx_guidance_uses_official_close_to_disabled_semantics():
    guidance = LTXModalityGuidance(
        cfg_scale=1.0 + 1e-10,
        stg_scale=0.0,
        modality_scale=1.0 + 1e-10,
        stg_blocks=(0,),
    )

    assert LTXGuidancePlan.build(LTXGuidanceSpec(video=guidance)).names == ("cond",)


def test_ltx_guidance_rescale_is_invariant_to_request_batching():
    cond = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 30.0], [50.0, 70.0]],
        ]
    )
    uncond = torch.stack((torch.zeros_like(cond[0]), cond[1]))
    guidance = LTXModalityGuidance(cfg_scale=2.0, rescale_scale=1.0)

    actual = combine_guided_x0(
        cond=cond,
        uncond_text=uncond,
        uncond_perturbed=0.0,
        uncond_modality=0.0,
        guidance=guidance,
    )
    individually = torch.cat(
        [
            combine_guided_x0(
                cond=cond[index : index + 1],
                uncond_text=uncond[index : index + 1],
                uncond_perturbed=0.0,
                uncond_modality=0.0,
                guidance=guidance,
            )
            for index in range(cond.shape[0])
        ]
    )

    # Fusing independent requests must not change either request's output.
    torch.testing.assert_close(actual, individually)


def test_ltx_guidance_rescale_handles_zero_variance_prediction():
    cond = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    guidance = LTXModalityGuidance(cfg_scale=2.0, rescale_scale=1.0)

    actual = combine_guided_x0(
        cond=cond,
        uncond_text=2 * cond,
        uncond_perturbed=0.0,
        uncond_modality=0.0,
        guidance=guidance,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.zeros_like(cond))


def test_ltx_audio_guidance_rescale_ignores_sp_padding_tokens():
    logical_cond = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])
    logical_uncond = torch.tensor([[[0.5, -1.0], [2.0, 1.0]]])
    guidance = LTXModalityGuidance(cfg_scale=3.0, rescale_scale=1.0)
    expected = combine_guided_x0(
        cond=logical_cond,
        uncond_text=logical_uncond,
        uncond_perturbed=0.0,
        uncond_modality=0.0,
        guidance=guidance,
    )
    padded_cond = torch.cat([logical_cond, torch.tensor([[[1000.0, -1000.0]]])], dim=1)
    padded_uncond = torch.cat([logical_uncond, torch.tensor([[[-500.0, 500.0]]])], dim=1)

    actual = combine_guided_x0(
        cond=padded_cond,
        uncond_text=padded_uncond,
        uncond_perturbed=0.0,
        uncond_modality=0.0,
        guidance=guidance,
        rescale_token_count=2,
    )

    torch.testing.assert_close(actual[:, :2], expected)


def test_ltx_guidance_builds_per_sample_stg_and_modality_masks():
    plan = LTXGuidancePlan.build(LTX23_ONE_STAGE_RECIPE.request_guidance)
    perturbations = build_perturbation_kwargs(plan, batch_size=2, reference=torch.ones(8, 1, 1))

    assert perturbations["video_self_attention_blocks"] == (28,)
    assert perturbations["audio_self_attention_blocks"] == (28,)
    assert perturbations["video_self_attention_mask"].flatten().tolist() == [1, 1, 1, 1, 0, 0, 1, 1]
    assert perturbations["a2v_cross_attention_mask"].flatten().tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
    torch.testing.assert_close(
        perturbations["a2v_cross_attention_mask"],
        perturbations["v2a_cross_attention_mask"],
    )


def test_ltx_default_sigma_schedule_matches_official_shift_and_stretch():
    scheduler = SimpleNamespace(
        config={
            "base_image_seq_len": 1024,
            "max_image_seq_len": 4096,
            "base_shift": 0.95,
            "max_shift": 2.05,
            "shift_terminal": 0.1,
        }
    )

    sigmas = _official_ltx_sigmas(scheduler, steps=4, device=torch.device("cpu"))

    torch.testing.assert_close(
        sigmas,
        torch.tensor([1.0, 0.8671, 0.6316, 0.1, 0.0]),
        rtol=1e-4,
        atol=1e-4,
    )


def test_ltx_sigma_schedule_does_not_stretch_when_terminal_config_is_missing():
    scheduler = SimpleNamespace(
        config={
            "base_image_seq_len": 1024,
            "max_image_seq_len": 4096,
            "base_shift": 0.95,
            "max_shift": 2.05,
        }
    )

    sigmas = _official_ltx_sigmas(scheduler, steps=4, device=torch.device("cpu"))

    torch.testing.assert_close(
        sigmas,
        torch.tensor([1.0, 0.9589, 0.8859, 0.7214, 0.0]),
        rtol=1e-4,
        atol=1e-4,
    )


def test_ltx_sigma_schedule_does_not_stretch_when_terminal_is_disabled():
    scheduler = SimpleNamespace(
        config={
            "base_image_seq_len": 1024,
            "max_image_seq_len": 4096,
            "base_shift": 0.95,
            "max_shift": 2.05,
            "shift_terminal": None,
        }
    )

    sigmas = _official_ltx_sigmas(scheduler, steps=4, device=torch.device("cpu"))

    torch.testing.assert_close(
        sigmas,
        torch.tensor([1.0, 0.9589, 0.8859, 0.7214, 0.0]),
        rtol=1e-4,
        atol=1e-4,
    )


def test_ltx_component_loader_installs_omni_connector_attention(monkeypatch):
    installed = []
    kernels = []

    class OmniAttention:
        def __init__(self, **kwargs):
            kernels.append(kwargs)

    class Attention:
        heads = 32
        head_dim = 128
        inner_kv_dim = 4096

        def set_processor(self, processor):
            installed.append(processor)

    monkeypatch.setattr(ltx2_components, "OmniAttention", OmniAttention)
    connectors = SimpleNamespace(
        video_connector=SimpleNamespace(
            learnable_registers=object(),
            transformer_blocks=[SimpleNamespace(attn1=Attention())],
        ),
        audio_connector=SimpleNamespace(
            learnable_registers=object(),
            transformer_blocks=[SimpleNamespace(attn1=Attention())],
        ),
    )

    _install_connector_attention(connectors)

    assert len(installed) == 2
    assert all(processor.__class__.__name__ == "_LTXConnectorAttnProcessor" for processor in installed)
    assert all(kernel["role"] == "ltx2.connector" for kernel in kernels)
    assert all(kernel["role_category"] == "self" for kernel in kernels)
    assert all(kernel["skip_sequence_parallel"] for kernel in kernels)
    assert all(kernel["disable_kv_quant"] for kernel in kernels)


def test_ltx_connector_attention_dispatches_through_omni_kernel():
    calls = []

    class Backend:
        @staticmethod
        def get_name():
            return "FLASH_ATTN"

    class OmniAttention:
        attn_backend = Backend

        def __call__(self, query, key, value, metadata):
            calls.append((query, key, value, metadata))
            return query

    class UpcastingNorm(torch.nn.Module):
        def forward(self, tensor):
            return tensor.float()

    attention = SimpleNamespace(
        heads=2,
        head_dim=4,
        inner_kv_dim=8,
        norm_q=UpcastingNorm(),
        norm_k=UpcastingNorm(),
        to_q=torch.nn.Identity(),
        to_k=torch.nn.Identity(),
        to_v=torch.nn.Identity(),
        to_out=(torch.nn.Identity(), torch.nn.Identity()),
        to_gate_logits=None,
        rope_type="split",
        omni_attention=OmniAttention(),
    )
    hidden_states = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    additive_mask = torch.zeros(2, 1, 3, 3)
    rotary = (
        torch.randn(2, 2, 3, 2, dtype=torch.float32),
        torch.randn(2, 2, 3, 2, dtype=torch.float32),
    )

    output = ltx2_components._LTXConnectorAttnProcessor()(
        attention,
        hidden_states,
        attention_mask=additive_mask,
        query_rotary_emb=rotary,
    )

    query, key, value, metadata = calls[0]
    assert query.shape == key.shape == value.shape == (2, 3, 2, 4)
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert metadata.attn_mask.shape == (2, 3)
    assert metadata.attn_mask.dtype == torch.bool
    assert metadata.attn_mask.all()
    expected_rotary = tuple(component.to(torch.bfloat16) for component in rotary)
    expected = ltx2_components.apply_split_rotary_emb(hidden_states, expected_rotary, head_dim=4)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_ltx2_distilled_pipeline_shares_recipe_runtime():
    assert issubclass(LTX2DistilledPipeline, LTXI2VConditioningMixin)
    assert issubclass(LTX2DistilledPipeline, LTX2Pipeline)
    assert LTX2DistilledPipeline._run_recipe is LTXRuntime._run_recipe
    assert "_forward_request" not in LTX2DistilledPipeline.__dict__


def test_ltx_variants_share_denoise_loop_and_i2v_conditioning():
    assert "_denoise_loop" not in LTX2Pipeline.__dict__
    assert issubclass(LTX2Pipeline, LTXI2VConditioningMixin)
    assert LTX2Pipeline.prepare_latents is LTXI2VConditioningMixin.prepare_latents
    assert LTX2Pipeline.prepare_audio_latents is LTXRuntime.prepare_audio_latents
    assert LTX2Pipeline.decode_phase is LTXRuntime.decode_phase


def test_ltx_seeded_latents_match_official_packed_rng_layout():
    pipeline = SimpleNamespace(
        model_version="2.5",
        vae_spatial_compression_ratio=2,
        vae_temporal_compression_ratio=2,
        transformer_spatial_patch_size=1,
        transformer_temporal_patch_size=1,
        audio_vae_mel_compression_ratio=2,
        od_config=SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1)),
    )
    device = torch.device("cpu")
    actual_generator = torch.Generator(device=device).manual_seed(123)
    expected_generator = torch.Generator(device=device).manual_seed(123)

    actual_video = ltx2_latents.prepare_video_latents(
        pipeline,
        batch_size=1,
        num_channels_latents=2,
        height=4,
        width=6,
        num_frames=3,
        dtype=torch.float32,
        device=device,
        generator=actual_generator,
    )
    actual_audio, original_length, padded_length = ltx2_latents.prepare_audio_latents(
        pipeline,
        batch_size=1,
        num_channels_latents=2,
        audio_latent_length=3,
        num_mel_bins=4,
        dtype=torch.float32,
        device=device,
        generator=actual_generator,
    )

    expected_video = torch.randn((1, 12, 2), generator=expected_generator)
    expected_audio = torch.randn((1, 3, 4), generator=expected_generator)

    torch.testing.assert_close(actual_video, expected_video, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_audio, expected_audio, rtol=0.0, atol=0.0)
    assert (original_length, padded_length) == (3, 3)


def test_denoise_executor_owns_progress_and_interrupt():
    updates = []

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def update(self):
            updates.append(True)

    pipeline = SimpleNamespace(
        interrupt=False,
        progress_bar=lambda total: Progress(),
    )
    seen: list[tuple[int, float]] = []
    timesteps = torch.tensor([3.0, 2.0, 1.0])
    initial_state = LTXAVState(video=torch.tensor(0.0), audio=torch.tensor(10.0))

    def step(index, timestep, state):
        seen.append((index, timestep.item()))
        pipeline.interrupt = True
        return LTXAVState(video=state.video + 1, audio=state.audio + 1)

    state = LTXDenoiseExecutor.run(pipeline, initial_state, timesteps, step)

    assert seen == [(0, 3.0)]
    assert updates == [True]
    torch.testing.assert_close(state.video, torch.tensor(1.0))
    torch.testing.assert_close(state.audio, torch.tensor(11.0))


def test_ltx25_distilled_two_stage_executes_custom_phase_schedules():
    request_inputs = LTXRequestInputs(
        prompt="prompt",
        negative_prompt="",
        height=64,
        width=64,
        num_frames=1,
        frame_rate=24.0,
        num_inference_steps=8,
        guidance=LTXGuidanceSpec.positive_only(),
        num_videos_per_prompt=1,
        generator=None,
        latents=None,
        audio_latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        decode_timestep=0.25,
        decode_noise_scale=0.5,
        output_type="np",
        max_sequence_length=16,
    )
    prompt_context = object()
    phase_calls = []

    def resolve_request_inputs(req, **kwargs):
        return request_inputs

    def run_phase(req, inputs, *, prompt_context=None, **kwargs):
        phase_recipe = kwargs["phase_recipe"]
        assert kwargs["image"] is source_image
        phase_calls.append((phase_recipe, inputs, prompt_context))
        if len(phase_calls) == 1:
            assert prompt_context is None
            assert inputs.decode_timestep == 0.25
            assert inputs.decode_noise_scale == 0.5
            context = prompt_context_sentinel
            assert phase_recipe.name == "generate_lowres"
            assert (inputs.height, inputs.width) == (32, 32)
            assert inputs.num_inference_steps == 2
            assert kwargs["sigmas"] == stage_1_sigmas
            assert kwargs["noise_scale"] == 1.0
            video = torch.ones(1, 128, 1, 1, 1)
            audio = torch.full((1, 8, 1, 2), 2.0)
            audio_for_next_phase = torch.full((1, 8, 1, 2), 5.0)
        else:
            assert prompt_context is prompt_context_sentinel
            assert phase_recipe.name == "refine"
            assert (inputs.height, inputs.width) == (64, 64)
            assert inputs.latents.shape == (1, 128, 1, 2, 2)
            torch.testing.assert_close(inputs.audio_latents, torch.full((1, 8, 1, 2), 5.0))
            assert inputs.audio_latents_normalized
            assert inputs.guidance_scale == 1.0
            assert inputs.num_inference_steps == 2
            assert kwargs["sigmas"] == stage_2_sigmas
            assert kwargs["noise_scale"] == stage_2_sigmas[0]
            assert inputs.decode_timestep == 0.0
            assert inputs.decode_noise_scale is None
            context = prompt_context
            video = torch.full((1, 128, 1, 2, 2), 3.0)
            audio = torch.full((1, 8, 1, 2), 4.0)
            audio_for_next_phase = None
        return LTXPhaseResult(
            forward_context=SimpleNamespace(prompt_context=context),
            video=video,
            audio=audio,
            audio_for_next_phase=audio_for_next_phase,
        )

    def decode_phase(phase):
        return DiffusionOutput(output=(phase.video, phase.audio))

    class FakeUpsampler(torch.nn.Module):
        dtype = torch.float32

        def forward(self, latents):
            return latents.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

    prompt_context_sentinel = prompt_context
    stage_1_sigmas = [1.0, 0.5, 0.0]
    stage_2_sigmas = [0.8, 0.2, 0.0]
    pipeline = object.__new__(LTX2TwoStagePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.pipeline_recipe = LTX25_DISTILLED_TWO_STAGE_RECIPE
    pipeline.device = torch.device("cpu")
    pipeline.vae_spatial_compression_ratio = 32
    pipeline.vae_temporal_compression_ratio = 8
    pipeline.latent_upsampler = FakeUpsampler()
    object.__setattr__(pipeline, "_resolve_request_inputs", resolve_request_inputs)
    object.__setattr__(pipeline, "run_phase", run_phase)
    object.__setattr__(pipeline, "decode_phase", decode_phase)

    source_image = object()
    output = pipeline.forward(
        SimpleNamespace(sampling_params_list=[], prompts=[]),
        image=source_image,
        stage_1_sigmas=stage_1_sigmas,
        stage_2_sigmas=stage_2_sigmas,
    )

    assert len(phase_calls) == 2
    assert phase_calls[1][2] is prompt_context_sentinel
    torch.testing.assert_close(output.output[0], torch.full((1, 128, 1, 2, 2), 3.0))
    torch.testing.assert_close(output.output[1], torch.full((1, 8, 1, 2), 4.0))


def test_ltx2_distilled_stage2_reapplies_final_resolution_i2v_conditioning():
    from vllm_omni.diffusion.models.ltx2 import ltx2_latents

    pipeline = object.__new__(LTX2TwoStagePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae_spatial_compression_ratio = 32
    pipeline.vae_temporal_compression_ratio = 8
    pipeline.transformer_spatial_patch_size = 1
    pipeline.transformer_temporal_patch_size = 1
    pipeline.vae = SimpleNamespace(
        latents_mean=torch.zeros(2),
        latents_std=torch.ones(2),
        config=SimpleNamespace(scaling_factor=1.0),
    )
    object.__setattr__(
        pipeline,
        "_encode_i2v_image_latents",
        lambda image, **_kwargs: torch.full((image.shape[0], 2, 1, 2, 2), 7.0),
    )

    packed, conditioning_mask = pipeline.prepare_latents(
        image=torch.zeros(1, 3, 64, 64),
        batch_size=1,
        num_channels_latents=2,
        height=64,
        width=64,
        num_frames=9,
        noise_scale=0.0,
        dtype=torch.float32,
        device=torch.device("cpu"),
        latents=torch.ones(1, 2, 2, 2, 2),
    )
    unpacked = ltx2_latents.unpack_latents(packed, 2, 2, 2)

    torch.testing.assert_close(unpacked[:, :, :1], torch.full((1, 2, 1, 2, 2), 7.0))
    torch.testing.assert_close(unpacked[:, :, 1:], torch.ones(1, 2, 1, 2, 2))
    assert conditioning_mask.shape == (1, 8)
    assert conditioning_mask[:, :4].all()
    assert not conditioning_mask[:, 4:].any()


def test_ltx2_distilled_two_stage_recipe_is_fixed_positive_only():
    stage1, stage2 = LTX2_DISTILLED_TWO_STAGE_RECIPE.phases

    assert stage1.spatial_downscale == 2
    assert stage1.guidance == stage2.guidance == LTXGuidanceSpec.positive_only()
    assert LTXGuidancePlan.build(stage1.guidance).names == ("cond",)
    assert LTXGuidancePlan.build(stage2.guidance).names == ("cond",)
    assert not stage1.allow_guidance_override
    assert not stage2.allow_guidance_override
    assert stage1.noise_scale == 1.0
    assert stage1.sigmas == (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
    assert stage2.sigmas == (0.909375, 0.725, 0.421875, 0.0)
    assert not LTX2_DISTILLED_TWO_STAGE_RECIPE.allow_request_sigmas
    assert not LTX2_DISTILLED_TWO_STAGE_RECIPE.allow_request_latents
    assert not LTX2_DISTILLED_TWO_STAGE_RECIPE.allow_negative_prompt
    assert LTX2_DISTILLED_TWO_STAGE_RECIPE.fixed_num_inference_steps
    assert (LTX2_DISTILLED_TWO_STAGE_RECIPE.height, LTX2_DISTILLED_TWO_STAGE_RECIPE.width) == (1024, 1536)


@pytest.mark.parametrize(
    ("request_inputs", "error"),
    [
        (lambda request: replace(request, height=32), "divisible by 64"),
        (lambda request: replace(request, num_frames=8), "8 \\* k \\+ 1"),
        (lambda request: replace(request, num_inference_steps=7), "uses 8 fixed Stage 1"),
        (lambda request: replace(request, latents=torch.empty(1)), "request-provided"),
        (lambda request: replace(request, audio_latents=torch.empty(1)), "request-provided"),
        (lambda request: replace(request, negative_prompt="negative"), "negative conditioning"),
        (lambda request: replace(request, negative_prompt_embeds=torch.empty(1)), "negative conditioning"),
        (lambda request: replace(request, guidance=LTX2_ONE_STAGE_RECIPE.request_guidance), "positive-only"),
    ],
)
def test_ltx_two_stage_recipe_rejects_unsupported_request_values(request_inputs, error):
    request = LTXRequestInputs(
        prompt="prompt",
        negative_prompt="",
        height=64,
        width=64,
        num_frames=1,
        frame_rate=24.0,
        num_inference_steps=8,
        guidance=LTXGuidanceSpec.positive_only(),
        num_videos_per_prompt=1,
        generator=None,
        latents=None,
        audio_latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        decode_timestep=0.0,
        decode_noise_scale=None,
        output_type="np",
        max_sequence_length=16,
    )

    with pytest.raises(ValueError, match=error):
        validate_pipeline_request(
            request_inputs(request),
            pipeline_recipe=LTX2_DISTILLED_TWO_STAGE_RECIPE,
            vae_spatial_compression_ratio=32,
            vae_temporal_compression_ratio=8,
            pipeline_name="LTX2TwoStagePipeline",
        )


def test_ltx_two_stage_recipe_rejects_sigmas():
    request = LTXRequestInputs(
        prompt="prompt",
        negative_prompt="",
        height=64,
        width=64,
        num_frames=1,
        frame_rate=24.0,
        num_inference_steps=8,
        guidance=LTXGuidanceSpec.positive_only(),
        num_videos_per_prompt=1,
        generator=None,
        latents=None,
        audio_latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        decode_timestep=0.0,
        decode_noise_scale=None,
        output_type="np",
        max_sequence_length=16,
    )

    with pytest.raises(ValueError, match="fixed phase sigma schedules"):
        validate_pipeline_request(
            request,
            pipeline_recipe=LTX2_DISTILLED_TWO_STAGE_RECIPE,
            vae_spatial_compression_ratio=32,
            vae_temporal_compression_ratio=8,
            pipeline_name="LTX2TwoStagePipeline",
            request_sigmas=[1.0, 0.0],
        )


@pytest.mark.parametrize(
    ("direct_kwargs", "sampling_kwargs", "error"),
    [
        ({"sigmas": [1.0, 0.0]}, {}, "fixed phase sigma schedules"),
        ({}, {"sigmas": [1.0, 0.0]}, "fixed phase sigma schedules"),
        ({"stage_1_sigmas": [1.0, 0.0]}, {}, "per-stage sigma schedules"),
        ({"latents": torch.empty(1)}, {}, "request-provided video or audio latents"),
        ({}, {"latents": torch.empty(1)}, "request-provided video or audio latents"),
        ({"audio_latents": torch.empty(1)}, {}, "request-provided video or audio latents"),
        (
            {},
            {"extra_args": {"audio_latents": torch.empty(1)}},
            "request-provided video or audio latents",
        ),
        ({"negative_prompt": "negative"}, {}, "negative conditioning"),
        ({"negative_prompt_embeds": torch.empty(1)}, {}, "negative conditioning"),
        ({}, {"num_inference_steps": 7}, "uses 8 fixed Stage 1"),
        ({"guidance_scale": 4.0}, {}, "fixed positive-only guidance"),
        ({}, {"guidance_scale": 4.0}, "fixed positive-only guidance"),
    ],
    ids=(
        "direct-sigmas",
        "request-sigmas",
        "direct-stage-sigmas",
        "direct-video-latents",
        "request-video-latents",
        "direct-audio-latents",
        "request-audio-latents",
        "direct-negative-prompt",
        "direct-negative-embeds",
        "request-num-steps",
        "direct-guidance",
        "request-guidance",
    ),
)
def test_ltx_distilled_forward_rejects_fixed_recipe_overrides(direct_kwargs, sampling_kwargs, error):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling_values = {
        "height": 64,
        "width": 64,
        "num_frames": 1,
        "num_inference_steps": 8,
        **sampling_kwargs,
    }
    sampling = OmniDiffusionSamplingParams(**sampling_values)
    req = DiffusionRequestBatch(
        [
            OmniDiffusionRequest(
                prompt="prompt",
                sampling_params=sampling,
                request_id="ltx-distilled-fixed-inputs",
            )
        ]
    )
    pipeline = _make_ltx_request_pipe(LTX2TwoStagePipeline)

    with pytest.raises(ValueError, match=error):
        pipeline.forward(req, **direct_kwargs)


def test_ltx_distilled_dummy_run_uses_fixed_recipe_values():
    from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    req = DiffusionRequestBatch(
        [
            OmniDiffusionRequest(
                prompt="dummy run",
                sampling_params=OmniDiffusionSamplingParams(
                    height=512,
                    width=512,
                    num_frames=LTX2TwoStagePipeline.dummy_run_num_frames,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                ),
                request_id=DUMMY_DIFFUSION_REQUEST_ID,
            )
        ]
    )
    pipeline = _make_ltx_request_pipe(LTX2TwoStagePipeline)
    seen = {}

    def fake_run_recipe(_req, request_inputs, **_kwargs):
        seen["request_inputs"] = request_inputs
        return "dummy-output"

    object.__setattr__(pipeline, "_run_recipe", fake_run_recipe)

    assert pipeline.forward(req) == "dummy-output"
    assert seen["request_inputs"].num_frames == 1
    assert seen["request_inputs"].num_inference_steps == 8
    assert seen["request_inputs"].guidance == LTXGuidanceSpec.positive_only()


def test_ltx_custom_sigmas_bypass_scheduler_shifting():
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True,
        base_shift=0.95,
        max_shift=2.05,
        shift_terminal=0.1,
    )
    pipeline = SimpleNamespace(scheduler=scheduler)
    sigmas = [1.0, 0.75, 0.25, 0.0]

    audio_scheduler, _, timesteps = prepare_scheduler_stage(
        pipeline,
        SimpleNamespace(num_inference_steps=3),
        device=torch.device("cpu"),
        sigmas=sigmas,
        timesteps=None,
        latent_num_frames=1,
        latent_height=1,
        latent_width=1,
        use_official_sigma_schedule=False,
    )

    expected = torch.tensor(sigmas, dtype=torch.float32)
    torch.testing.assert_close(pipeline.scheduler.sigmas, expected)
    torch.testing.assert_close(audio_scheduler.sigmas, expected)
    torch.testing.assert_close(timesteps, expected[:-1] * scheduler.config.num_train_timesteps)


class TestLTXRequestParsing:
    def test_request_resolves_official_and_explicit_negative_prompts_per_sample(self):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt=prompt,
                    sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
                    request_id=f"ltx-negative-prompt-{index}",
                )
                for index, prompt in enumerate(
                    (
                        "use the official default",
                        {"prompt": "disable the default", "negative_prompt": ""},
                        {"prompt": "custom negative", "negative_prompt": "custom"},
                    )
                )
            ]
        )

        resolved = _resolve_request_inputs_for_test(_make_ltx_request_pipe(LTX2Pipeline), req)

        assert resolved.negative_prompt == [LTX_DEFAULT_NEGATIVE_PROMPT, "", "custom"]

    def test_forward_negative_prompt_fallback_is_not_discarded(self):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt="prompt",
                    sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
                    request_id="ltx-forward-negative-prompt",
                )
            ]
        )

        resolved = _resolve_request_inputs_for_test(
            _make_ltx_request_pipe(LTX2Pipeline),
            req,
            negative_prompt="forward fallback",
        )

        assert resolved.negative_prompt == ["forward fallback"]

    def test_unified_t2v_i2v_entry_resolves_request_inputs(self):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        prompt_embeds = torch.tensor([[1.0, 2.0]])
        negative_prompt_embeds = torch.tensor([[3.0, 4.0]])
        prompt_attention_mask = torch.tensor([True, False])
        negative_attention_mask = torch.tensor([False, True])
        video_latents = torch.ones(1, 2, 3)
        audio_latents = torch.zeros(1, 2, 3)
        generator = torch.Generator().manual_seed(123)

        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt={
                        "prompt": "shared prompt",
                        "negative_prompt": "shared negative",
                        "additional_information": {
                            "prompt_embeds": prompt_embeds,
                            "negative_prompt_embeds": negative_prompt_embeds,
                            "attention_mask": prompt_attention_mask,
                            "negative_attention_mask": negative_attention_mask,
                        },
                    },
                    sampling_params=OmniDiffusionSamplingParams(
                        height=384,
                        width=512,
                        num_frames=25,
                        frame_rate=12.5,
                        num_inference_steps=1,
                        num_outputs_per_prompt=2,
                        guidance_scale=6.0,
                        generator=generator,
                        latents=video_latents,
                        extra_args={"audio_latents": audio_latents},
                        decode_timestep=[0.1],
                        decode_noise_scale=[0.2],
                        output_type="latent",
                        max_sequence_length=17,
                    ),
                    request_id="ltx23-shared-request-inputs",
                )
            ]
        )

        resolved = _resolve_request_inputs_for_test(_make_ltx_request_pipe(LTX2Pipeline), req)

        assert resolved.prompt is None
        assert resolved.negative_prompt is None
        assert resolved.height == 384
        assert resolved.width == 512
        assert resolved.num_frames == 25
        assert resolved.frame_rate == 12.5
        assert resolved.num_inference_steps == 2
        assert resolved.guidance_scale == 6.0
        assert resolved.num_videos_per_prompt == 2
        assert isinstance(resolved.generator, list)
        assert [gen.initial_seed() for gen in resolved.generator] == [123, 123]
        assert resolved.decode_timestep == [0.1]
        assert resolved.decode_noise_scale == [0.2]
        assert resolved.output_type == "latent"
        assert resolved.max_sequence_length == 17
        torch.testing.assert_close(resolved.latents, video_latents)
        torch.testing.assert_close(resolved.audio_latents, audio_latents)
        torch.testing.assert_close(resolved.prompt_embeds, torch.stack([prompt_embeds]))
        torch.testing.assert_close(resolved.negative_prompt_embeds, torch.stack([negative_prompt_embeds]))
        torch.testing.assert_close(resolved.prompt_attention_mask, torch.stack([prompt_attention_mask]))
        torch.testing.assert_close(resolved.negative_prompt_attention_mask, torch.stack([negative_attention_mask]))

    def test_nonbatch_distilled_request_uses_precomputed_positive_embeddings(self):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        prompt_embeds = torch.tensor([[1.0, 2.0]])
        prompt_attention_mask = torch.tensor([True, False])
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt={
                        "prompt": "must not be forwarded with embeddings",
                        "additional_information": {
                            "prompt_embeds": prompt_embeds,
                            "attention_mask": prompt_attention_mask,
                        },
                    },
                    sampling_params=OmniDiffusionSamplingParams(num_inference_steps=8),
                    request_id="ltx-distilled-prompt-embeds",
                )
            ]
        )

        resolved = _resolve_request_inputs_for_test(
            _make_ltx_request_pipe(LTX2TwoStagePipeline),
            req,
            guidance_scale=None,
        )

        assert resolved.prompt is None
        assert resolved.negative_prompt == [""]
        torch.testing.assert_close(resolved.prompt_embeds, torch.stack([prompt_embeds]))
        torch.testing.assert_close(resolved.prompt_attention_mask, torch.stack([prompt_attention_mask]))

    def test_request_resolves_per_modality_guidance_and_common_cfg_override(self):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt={"prompt": "prompt", "negative_prompt": "negative"},
                    sampling_params=OmniDiffusionSamplingParams(
                        num_inference_steps=2,
                        extra_args={
                            "video_cfg_scale": 4.0,
                            "audio_cfg_guidance_scale": 6.0,
                            "video_stg_guidance_scale": 0.5,
                            "audio_stg_scale": 0.25,
                            "a2v_guidance_scale": 2.0,
                            "v2a_guidance_scale": 4.0,
                            "video_rescale_scale": 0.2,
                            "audio_rescale_scale": 0.3,
                            "video_stg_blocks": [3, 4],
                            "audio_stg_blocks": 5,
                        },
                    ),
                    request_id="ltx-guidance-overrides",
                )
            ]
        )
        pipe = _make_ltx_request_pipe(LTX2Pipeline)

        resolved = _resolve_request_inputs_for_test(pipe, req, guidance_scale=None)

        assert resolved.guidance.video == LTXModalityGuidance(4.0, 0.5, 2.0, 0.2, (3, 4))
        assert resolved.guidance.audio == LTXModalityGuidance(6.0, 0.25, 4.0, 0.3, (5,))

        common = _resolve_request_inputs_for_test(pipe, req, guidance_scale=2.0)
        assert common.guidance.video.cfg_scale == common.guidance.audio.cfg_scale == 2.0
        # LTX exposes independent video/audio rescale controls, not the
        # model-agnostic sampling field.
        assert common.guidance.video.rescale_scale == 0.2
        assert common.guidance.audio.rescale_scale == 0.3

    def test_request_resolves_custom_sigmas_from_extra_args(self):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        sigmas = [1.0, 0.5, 0.0]
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt="prompt",
                    sampling_params=OmniDiffusionSamplingParams(extra_args={"sigmas": sigmas}),
                    request_id="ltx-custom-sigmas",
                )
            ]
        )

        assert _make_ltx_request_pipe(LTX2Pipeline)._resolve_request_sigmas(req, None) == sigmas

    def test_request_resolves_per_stage_sigmas_from_extra_args(self):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        stage_1_sigmas = [1.0, 0.4, 0.0]
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt="prompt",
                    sampling_params=OmniDiffusionSamplingParams(extra_args={"stage_1_sigmas": stage_1_sigmas}),
                    request_id="ltx-custom-phase-sigmas",
                )
            ]
        )

        resolved = _make_ltx_request_pipe(LTX2TwoStagePipeline)._resolve_request_phase_sigmas(
            req, [1.0, 0.0], [0.8, 0.0]
        )

        assert resolved == (stage_1_sigmas, [0.8, 0.0])

    @pytest.mark.parametrize("value", [-1, 52, 1.5, True])
    def test_request_rejects_invalid_image_crf(self, value):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt="prompt",
                    sampling_params=OmniDiffusionSamplingParams(extra_args={"image_crf": value}),
                    request_id="ltx-invalid-image-crf",
                )
            ]
        )

        with pytest.raises(ValueError, match="image_crf"):
            _resolve_request_inputs_for_test(_make_ltx_request_pipe(LTX2Pipeline), req)

    def test_request_rejects_unsupported_generic_scheduler_and_rescale_controls(self):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        cases = (
            (OmniDiffusionSamplingParams(timesteps=torch.tensor([1.0])), "timesteps"),
            (OmniDiffusionSamplingParams(guidance_rescale=0.5), "guidance_rescale"),
            (OmniDiffusionSamplingParams(extra_args={"flow_shift": 3.0}), "flow_shift"),
        )
        pipe = _make_ltx_request_pipe(LTX2Pipeline)

        for index, (sampling, field_name) in enumerate(cases):
            req = DiffusionRequestBatch(
                [
                    OmniDiffusionRequest(
                        prompt="prompt",
                        sampling_params=sampling,
                        request_id=f"ltx-unsupported-{index}",
                    )
                ]
            )
            with pytest.raises(ValueError, match=field_name):
                _resolve_request_inputs_for_test(pipe, req)

    def test_request_batch_rejects_mixed_guidance_specs(self):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt="prompt-0",
                    sampling_params=OmniDiffusionSamplingParams(
                        num_inference_steps=2,
                        extra_args={"video_cfg_scale": scale},
                    ),
                    request_id=f"ltx-mixed-guidance-{index}",
                )
                for index, scale in enumerate((3.0, 4.0))
            ]
        )

        with pytest.raises(ValueError, match="identical video/audio guidance"):
            _resolve_request_inputs_for_test(
                _make_ltx_request_pipe(LTX2Pipeline),
                req,
                guidance_scale=None,
            )

    def test_request_input_resolution_rejects_mixed_precomputed_prompt_fields(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = _make_ltx_request_pipe(LTX2Pipeline)
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt={
                        "prompt": "with embeds",
                        "additional_information": {"prompt_embeds": torch.tensor([[1.0]])},
                    },
                    sampling_params=OmniDiffusionSamplingParams(
                        height=384,
                        width=512,
                        num_frames=25,
                        num_inference_steps=2,
                    ),
                    request_id="ltx23-mixed-precomputed-fields-0",
                ),
                OmniDiffusionRequest(
                    prompt={"prompt": "without embeds"},
                    sampling_params=OmniDiffusionSamplingParams(
                        height=384,
                        width=512,
                        num_frames=25,
                        num_inference_steps=2,
                    ),
                    request_id="ltx23-mixed-precomputed-fields-1",
                ),
            ]
        )

        with pytest.raises(ValueError, match="mix of provided and missing prompt_embeds"):
            _resolve_request_inputs_for_test(pipe, req)


class TestLTXForwardStages:
    def test_gemma_tokenization_matches_official_bos_and_left_padding(self):
        pipe = _make_ltx_request_pipe(LTX2Pipeline)

        class FakeTokenizer:
            bos_token_id = 2
            eos_token_id = 1
            eos_token = "<eos>"
            pad_token_id = 0
            pad_token = "<pad>"
            padding_side = "right"

            def __init__(self):
                self.calls = []

            def __call__(self, text, **kwargs):
                self.calls.append((text, kwargs))
                ids = [7, 8, 9, self.eos_token_id] if text == "long" else [6, self.eos_token_id]
                return SimpleNamespace(input_ids=ids)

            def pad(self, encoded, *, padding, max_length, return_tensors, return_attention_mask):
                assert padding == "max_length"
                assert return_tensors == "pt"
                assert return_attention_mask is True
                rows = []
                masks = []
                for ids in encoded["input_ids"]:
                    pad = max_length - len(ids)
                    rows.append([self.pad_token_id] * pad + ids)
                    masks.append([0] * pad + [1] * len(ids))
                return SimpleNamespace(input_ids=torch.tensor(rows), attention_mask=torch.tensor(masks))

        class FakeTextEncoder(torch.nn.Module):
            dtype = torch.float32

            def __init__(self):
                super().__init__()
                self.seen_input_ids = None
                self.seen_attention_mask = None

            def forward(self, *, input_ids, attention_mask, output_hidden_states):
                assert output_hidden_states is True
                self.seen_input_ids = input_ids.clone()
                self.seen_attention_mask = attention_mask.clone()
                return SimpleNamespace(hidden_states=(input_ids.unsqueeze(-1).float(),))

        tokenizer = FakeTokenizer()
        text_encoder = FakeTextEncoder()
        object.__setattr__(pipe, "tokenizer", tokenizer)
        object.__setattr__(pipe, "text_encoder", text_encoder)

        pipe._get_gemma_prompt_embeds(
            [" long ", "short"],
            max_sequence_length=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        torch.testing.assert_close(text_encoder.seen_input_ids, torch.tensor([[2, 7, 8, 9], [0, 2, 6, 1]]))
        torch.testing.assert_close(text_encoder.seen_attention_mask, torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]]))
        assert tokenizer.padding_side == "left"
        assert [call[0] for call in tokenizer.calls] == ["long", "short"]

    def test_encode_prompt_batches_positive_and_negative_gemma_inputs(self):
        pipe = _make_ltx_request_pipe(LTX2Pipeline)
        calls = []

        def fake_get_gemma_prompt_embeds(**kwargs):
            calls.append(kwargs)
            count = len(kwargs["prompt"])
            embeds = torch.arange(count, dtype=torch.float32).view(count, 1, 1)
            mask = torch.arange(count).view(count, 1)
            return embeds, mask

        object.__setattr__(pipe, "_get_gemma_prompt_embeds", fake_get_gemma_prompt_embeds)

        prompt_embeds, prompt_mask, negative_embeds, negative_mask = pipe.encode_prompt(
            prompt=["positive-0", "positive-1"],
            negative_prompt=["negative-0", "negative-1"],
            do_classifier_free_guidance=True,
            num_videos_per_prompt=2,
            device=torch.device("cpu"),
        )

        assert len(calls) == 1
        assert calls[0]["prompt"] == ["positive-0", "positive-1", "negative-0", "negative-1"]
        assert calls[0]["num_videos_per_prompt"] == 1
        torch.testing.assert_close(prompt_embeds.flatten(), torch.tensor([0.0, 0.0, 1.0, 1.0]))
        torch.testing.assert_close(negative_embeds.flatten(), torch.tensor([2.0, 2.0, 3.0, 3.0]))
        torch.testing.assert_close(prompt_mask.flatten(), torch.tensor([0, 0, 1, 1]))
        torch.testing.assert_close(negative_mask.flatten(), torch.tensor([2, 2, 3, 3]))

    def test_encode_prompt_repeats_precomputed_embeds_for_num_outputs(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = _make_ltx_request_pipe(LTX2Pipeline)
        prompt_embeds = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
        negative_prompt_embeds = torch.tensor([[[-1.0], [-2.0]], [[-3.0], [-4.0]]])
        prompt_attention_mask = torch.tensor([[True, False], [False, True]])
        negative_prompt_attention_mask = torch.tensor([[False, False], [True, True]])

        (
            repeated_prompt_embeds,
            repeated_prompt_attention_mask,
            repeated_negative_prompt_embeds,
            repeated_negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt=None,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(
            repeated_prompt_embeds,
            torch.tensor([[[1.0], [2.0]], [[1.0], [2.0]], [[3.0], [4.0]], [[3.0], [4.0]]]),
        )
        torch.testing.assert_close(
            repeated_negative_prompt_embeds,
            torch.tensor([[[-1.0], [-2.0]], [[-1.0], [-2.0]], [[-3.0], [-4.0]], [[-3.0], [-4.0]]]),
        )
        torch.testing.assert_close(
            repeated_prompt_attention_mask,
            torch.tensor([[True, False], [True, False], [False, True], [False, True]]),
        )
        torch.testing.assert_close(
            repeated_negative_prompt_attention_mask,
            torch.tensor([[False, False], [False, False], [True, True], [True, True]]),
        )

    def test_t2v_forward_delegates_to_shared_recipe_runtime(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = _make_ltx_request_pipe(LTX2Pipeline)
        request_sigmas = [1.0, 0.4]
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt="stage refactor prompt",
                    sampling_params=OmniDiffusionSamplingParams(
                        height=384,
                        width=512,
                        num_frames=25,
                        frame_rate=16,
                        num_inference_steps=3,
                        guidance_scale=4.5,
                        output_type="latent",
                        sigmas=request_sigmas,
                    ),
                    request_id="ltx23-t2v-forward-stage-delegation",
                )
            ]
        )
        fallback_sigmas = [1.0, 0.5]
        seen = {}

        def fake_run_recipe(req_arg, request_inputs, **kwargs):
            seen["req"] = req_arg
            seen["request_inputs"] = request_inputs
            seen["kwargs"] = kwargs
            return ["delegated"]

        object.__setattr__(pipe, "_run_recipe", fake_run_recipe)

        output = pipe.forward(req, sigmas=fallback_sigmas)

        assert output == ["delegated"]
        assert seen["req"] is req
        assert seen["request_inputs"].height == 384
        assert seen["request_inputs"].width == 512
        assert seen["request_inputs"].num_frames == 25
        assert seen["request_inputs"].frame_rate == 16.0
        assert seen["request_inputs"].guidance_scale == 4.5
        assert seen["request_inputs"].output_type == "latent"
        assert seen["kwargs"] == {
            "request_sigmas": request_sigmas,
            "request_phase_sigmas": None,
            "image": None,
        }

    def test_t2v_forward_prefers_request_sigmas(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        request_sigmas = [1.0, 0.25]
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt="custom sigma schedule",
                    sampling_params=OmniDiffusionSamplingParams(
                        num_inference_steps=2,
                        sigmas=request_sigmas,
                    ),
                    request_id="ltx23-request-sigmas",
                )
            ]
        )
        pipe = _make_ltx_request_pipe(LTX2Pipeline)
        seen = {}

        def fake_run_recipe(req_arg, request_inputs, **kwargs):
            del req_arg, request_inputs
            seen.update(kwargs)
            return ["delegated"]

        object.__setattr__(pipe, "_run_recipe", fake_run_recipe)

        output = pipe.forward(req, sigmas=[1.0, 0.5])

        assert output == ["delegated"]
        assert seen["request_sigmas"] == request_sigmas


class TestPipelineComponents:
    def test_ltx_pipeline_declares_offload_components(self):
        """LTX2Pipeline exposes all one-stage modules to offload discovery."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.transformer = torch.nn.Linear(1, 1)
        pipe.text_encoder = torch.nn.Linear(1, 1)
        pipe.connectors = torch.nn.Linear(1, 1)
        pipe.vae = torch.nn.Linear(1, 1)
        pipe.audio_vae = torch.nn.Linear(1, 1)
        pipe.vocoder = torch.nn.Linear(1, 1)

        modules = ModuleDiscovery.discover(pipe)

        assert LTX2Pipeline._dit_modules == ["transformer"]
        assert LTX2Pipeline._encoder_modules == ["text_encoder", "connectors"]
        assert LTX2Pipeline._vae_modules == ["vae", "audio_vae"]
        assert LTX2Pipeline._resident_modules == ["vocoder"]
        assert modules.dit_names == ["transformer"]
        assert modules.encoder_names == ["text_encoder", "connectors"]
        assert modules.resident_names == ["vocoder"]
        assert len(modules.vaes) == 2

    def test_distilled_profile_keeps_upsampler_in_managed_resident_components(self):
        assert LTX2_DISTILLED_COMPONENT_PROFILE.resident_modules == ("vocoder", "latent_upsampler")


class TestLTX23DecodeConditioning:
    def test_decode_conditioning_expands_per_prompt_values_to_effective_batch(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_runtime import _expand_per_prompt_decode_value

        assert _expand_per_prompt_decode_value(
            [0.1, 0.2],
            prompt_batch_size=2,
            effective_batch_size=4,
            field_name="decode_timestep",
        ) == [0.1, 0.1, 0.2, 0.2]
        assert _expand_per_prompt_decode_value(
            [0.3],
            prompt_batch_size=2,
            effective_batch_size=4,
            field_name="decode_timestep",
        ) == [0.3, 0.3, 0.3, 0.3]
        assert _expand_per_prompt_decode_value(
            [0.1, 0.2, 0.3, 0.4],
            prompt_batch_size=2,
            effective_batch_size=4,
            field_name="decode_timestep",
        ) == [0.1, 0.2, 0.3, 0.4]

    def test_decode_conditioning_rejects_ambiguous_lengths(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_runtime import _expand_per_prompt_decode_value

        with pytest.raises(ValueError, match="decode_timestep"):
            _expand_per_prompt_decode_value(
                [0.1, 0.2, 0.3],
                prompt_batch_size=2,
                effective_batch_size=4,
                field_name="decode_timestep",
            )


class TestRegistryIntegration:
    """Verify the unified one-stage and existing two-stage entries."""

    def test_pipeline_module_paths(self):
        """Registry entries must point to the correct modules."""
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        assert _DIFFUSION_MODELS["LTX2Pipeline"] == ("ltx2", "pipeline_ltx2", "LTX2Pipeline")
        assert _DIFFUSION_MODELS["LTX2FullPipeline"] == (
            "ltx2",
            "pipeline_ltx2",
            "LTX2FullPipeline",
        )
        assert _DIFFUSION_MODELS["LTX2TwoStagePipeline"] == (
            "ltx2",
            "pipeline_ltx2_two_stage",
            "LTX2TwoStagePipeline",
        )
        assert _DIFFUSION_MODELS["LTX2DistilledPipeline"] == (
            "ltx2",
            "pipeline_ltx2_two_stage",
            "LTX2DistilledPipeline",
        )
        removed_entries = {
            "LTX2ImageToVideoPipeline",
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
            "LTX2TwoStagesPipeline",
            "LTX2ImageToVideoTwoStagesPipeline",
        }
        assert removed_entries.isdisjoint(_DIFFUSION_MODELS)

    def test_post_process_funcs_registered(self):
        """Pipeline variants must map to get_ltx2_post_process_func."""
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        expected = [
            "LTX2Pipeline",
            "LTX2FullPipeline",
            "LTX2TwoStagePipeline",
            "LTX2DistilledPipeline",
            "LTX2T2VDMD2Pipeline",
            "LTX2I2VDMD2Pipeline",
        ]
        for name in expected:
            assert name in _DIFFUSION_POST_PROCESS_FUNCS, f"{name} not in _DIFFUSION_POST_PROCESS_FUNCS"
            assert _DIFFUSION_POST_PROCESS_FUNCS[name] == "get_ltx2_post_process_func"

    @pytest.mark.parametrize(
        "model_class_name",
        [
            "LTX2Pipeline",
            "LTX2FullPipeline",
            "LTX2TwoStagePipeline",
            "LTX2DistilledPipeline",
            "LTX2T2VDMD2Pipeline",
            "LTX2I2VDMD2Pipeline",
        ],
    )
    def test_post_process_func_resolves_from_every_entry_module(self, model_class_name):
        from vllm_omni.diffusion.registry import get_diffusion_post_process_func

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as config_file:
                json.dump({"output_sampling_rate": 48000}, config_file)

            post_process = get_diffusion_post_process_func(
                SimpleNamespace(model_class_name=model_class_name, model=tmpdir)
            )

        result = post_process((torch.zeros(1), torch.zeros(1)))
        assert result["audio_sample_rate"] == 48000

    def test_cache_dit_for_ltx2_does_not_have_custom_enablers_registered(self):
        """Pipeline variants are *not* registered in CUSTOM_DIT_ENABLERS."""
        from vllm_omni.diffusion.cache.cachedit import CUSTOM_DIT_ENABLERS

        # NOTE: We used to have custom enablers for this model, but refactored to handle
        # it more generically. Now we only need to ensure it has git cache adapter config.
        expected = ["LTX2Pipeline"]
        for name in expected:
            assert name not in CUSTOM_DIT_ENABLERS, f"{name} not in CUSTOM_DIT_ENABLERS"


class TestVocoderSampleRateDetection:
    """Test _detect_vocoder_output_sample_rate logic."""

    def test_detects_48khz_from_config(self):
        """Should detect output_sampling_rate=48000 from vocoder/config.json."""
        from vllm_omni.diffusion.models.ltx2.ltx2_components import _detect_vocoder_output_sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"output_sampling_rate": 48000, "input_sampling_rate": 16000}, f)

            result = _detect_vocoder_output_sample_rate(tmpdir)
            assert result == 48000

    def test_returns_none_for_no_output_sr(self):
        """Should return None if vocoder config has no output_sampling_rate."""
        from vllm_omni.diffusion.models.ltx2.ltx2_components import _detect_vocoder_output_sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"sampling_rate": 16000}, f)

            result = _detect_vocoder_output_sample_rate(tmpdir)
            assert result is None

    def test_returns_none_for_missing_directory(self):
        """Should return None if vocoder directory doesn't exist."""
        from vllm_omni.diffusion.models.ltx2.ltx2_components import _detect_vocoder_output_sample_rate

        result = _detect_vocoder_output_sample_rate("/nonexistent/path")
        assert result is None


class TestPostProcessFunction:
    """Test the post-process function factory."""

    def test_post_process_includes_audio_sample_rate(self):
        """Post-process func should include audio_sample_rate when detected."""
        import torch

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import get_ltx2_post_process_func

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"output_sampling_rate": 48000}, f)

            # Create a minimal od_config mock
            class MockConfig:
                model = tmpdir

            func = get_ltx2_post_process_func(MockConfig())

            video = torch.zeros(1, 3, 4, 64, 64)
            audio = torch.zeros(1, 1, 48000)
            result = func((video, audio))

            assert isinstance(result, dict)
            assert "video" in result
            assert "audio" in result
            assert result["audio_sample_rate"] == 48000

    def test_post_process_without_vocoder_config(self):
        """Post-process func should work without vocoder config (no audio_sample_rate key)."""
        import torch

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import get_ltx2_post_process_func

        class MockConfig:
            model = "/nonexistent/path"

        func = get_ltx2_post_process_func(MockConfig())

        video = torch.zeros(1, 3, 4, 64, 64)
        audio = torch.zeros(1, 1, 16000)
        result = func((video, audio))

        assert isinstance(result, dict)
        assert "video" in result
        assert "audio" in result
        assert "audio_sample_rate" not in result
