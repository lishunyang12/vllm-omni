# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contract tests for MiniMax H3 Super Acceleration."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.ltx2.ltx2_components import (
    LTX25_TWO_STAGE_COMPONENT_PROFILE,
    resolve_ltx_checkpoint_kind,
    resolve_ltx_component_profile,
)
from vllm_omni.diffusion.models.ltx2.ltx2_latents import pack_audio_latents, pack_latents
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import (
    LTX25_H3_REFINER_RECIPE,
    LTX_STAGE_2_DISTILLED_SIGMAS,
    resolve_ltx_pipeline_recipe,
)
from vllm_omni.diffusion.models.ltx2.pipeline_ltx25_h3_refiner import (
    H3_REFINER_AUDIO_SAMPLE_RATE,
    LTX25H3RefinerPipeline,
    _as_audio_tensor,
    _as_video_tensor,
    _configure_h3_refiner_vae,
    _encode_h3_refiner_media,
    _resolve_refiner_frame_count,
    get_ltx25_h3_refiner_post_process_func,
)
from vllm_omni.diffusion.models.ltx2.taehv import (
    TAEHV_CHECKPOINT_SHA256,
    TAEHV_CHECKPOINT_URL,
    LTXWideTAEHVDecoder,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_super import (
    MiniMaxH3SuperDraftPipeline,
    _minimax_h3_super_post_process,
    _prepare_h3_super_handoff,
)
from vllm_omni.diffusion.models.minimax_h3.taeh3 import (
    TAEH3_CHECKPOINT_SHA256,
    TAEH3_CHECKPOINT_URL,
    TAEH3Decoder,
)
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS, _DIFFUSION_POST_PROCESS_FUNCS
from vllm_omni.model_executor.stage_input_processors.minimax_h3_super import h3_to_ltx25_refiner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_h3_refiner_recipe_enters_only_fixed_three_step_phase():
    recipe = resolve_ltx_pipeline_recipe("h3_refiner", "2.5")

    assert recipe is LTX25_H3_REFINER_RECIPE
    assert recipe.num_inference_steps == 3
    assert recipe.phases[0].sigmas == LTX_STAGE_2_DISTILLED_SIGMAS
    assert recipe.phases[0].adapter_slot == "ltx_distilled"
    assert recipe.phases[0].adapter_scale == 0.8
    assert not recipe.phases[0].guidance.do_cfg
    assert not recipe.phases[0].guidance.do_stg
    assert not recipe.phases[0].guidance.do_modality_guidance
    assert resolve_ltx_component_profile("h3_refiner", "2.5") is LTX25_TWO_STAGE_COMPONENT_PROFILE
    assert resolve_ltx_checkpoint_kind("h3_refiner") == "regular"


@pytest.mark.parametrize(("draft_frames", "refiner_frames"), [(124, 121), (243, 241)])
def test_h3_refiner_trims_released_draft_shapes(draft_frames, refiner_frames):
    video = np.zeros((draft_frames, 8, 12, 3), dtype=np.float32)

    assert _as_video_tensor(video).shape == (1, 3, draft_frames, 8, 12)
    assert _resolve_refiner_frame_count(draft_frames) == refiner_frames


def test_h3_refiner_conforms_mono_audio_to_stereo_and_exact_duration():
    audio = _as_audio_tensor(torch.zeros(100))

    assert audio.shape == (1, 2, 100)
    conformed = LTX25H3RefinerPipeline._conform_source_audio(audio, frame_count=121)
    assert conformed.shape == (1, 2, int(121 / 24 * H3_REFINER_AUDIO_SAMPLE_RATE))


def test_h3_refiner_uses_released_full_temporal_vae_tile():
    vae = SimpleNamespace(use_framewise_decoding=False)

    _configure_h3_refiner_vae(vae)

    assert vae.tile_sample_min_num_frames == 128
    assert vae.tile_sample_stride_num_frames == 104
    assert vae.tile_sample_min_height == 768
    assert vae.tile_sample_stride_height == 704
    assert vae.tile_sample_min_width == 768
    assert vae.tile_sample_stride_width == 704
    assert vae.use_framewise_decoding


def test_h3_refiner_returns_encoder_ready_video_and_reuses_source_audio():
    class FakeVAE:
        dtype = torch.float32

    class FakeTAEHV:
        def decode_video(self, latents):
            assert latents.shape == (1, 128, 1, 2, 2)
            return torch.ones(1, 1, 3, 2, 2)

    source_audio = torch.randn(1, 2, 32)
    pipeline = SimpleNamespace(
        _h3_source_audio=source_audio,
        _make_output=lambda output: DiffusionOutput(output=output),
        vae=FakeVAE(),
        taehv_decoder=FakeTAEHV(),
        distributed_video_decode=False,
        encode_output_video=False,
    )
    latents = torch.zeros(1, 128, 1, 2, 2)

    output = LTX25H3RefinerPipeline._decode_output(
        pipeline,
        latents=latents,
        audio_latents=torch.ones(1, 8, 2),
        output_type="pt",
        connector_prompt_embeds=torch.zeros(1),
        generator=None,
        device=torch.device("cpu"),
        decode_timestep=0.0,
        decode_noise_scale=None,
        prompt_batch_size=1,
    )

    assert output.output[0].shape == (1, 1, 2, 2, 3)
    assert output.output[0].dtype == torch.uint8
    assert torch.all(output.output[0] == 255)
    assert output.output[1] is source_audio


def test_h3_refiner_keeps_normalized_video_latents_for_taehv():
    normalized_video = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 2, 2)
    raw_audio = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    pipeline = SimpleNamespace(
        transformer_spatial_patch_size=1,
        transformer_temporal_patch_size=1,
        audio_vae=SimpleNamespace(
            latents_mean=torch.tensor(10.0),
            latents_std=torch.tensor(2.0),
        ),
    )
    context = SimpleNamespace(
        latent_num_frames=1,
        latent_height=2,
        latent_width=2,
        original_audio_num_frames=2,
        latent_mel_bins=2,
    )

    video, audio = LTX25H3RefinerPipeline._unpack_and_denormalize_stage(
        pipeline,
        context,
        pack_latents(normalized_video),
        pack_audio_latents(raw_audio),
    )

    torch.testing.assert_close(video, normalized_video)
    torch.testing.assert_close(audio, raw_audio * 2.0 + 10.0)


def test_h3_refiner_encodes_media_before_worker_ipc(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_mux(video, audio, **kwargs):
        captured.update(video=video, audio=audio, kwargs=kwargs)
        return b"encoded-mp4"

    monkeypatch.setattr("vllm_omni.diffusion.utils.media_utils.mux_video_audio_bytes", fake_mux)
    video = torch.zeros(1, 2, 4, 6, 3, dtype=torch.uint8)
    audio = torch.zeros(1, 2, 32)

    encoded = _encode_h3_refiner_media(video, audio)

    assert encoded == b"encoded-mp4"
    assert captured["video"].shape == (2, 4, 6, 3)
    assert captured["video"].dtype == np.uint8
    assert captured["audio"].shape == (2, 32)
    assert captured["kwargs"]["fps"] == 24.0
    assert captured["kwargs"]["audio_sample_rate"] == 32_000


def test_h3_refiner_postprocess_exposes_preencoded_mp4():
    postprocess = get_ltx25_h3_refiner_post_process_func(None)

    output = postprocess({"video_mp4": b"encoded-mp4"})

    assert output["payload"] == {"video": b"encoded-mp4"}
    assert output["metadata"]["video"] == {"fps": 24.0, "media_type": "video/mp4"}
    assert output["metadata"]["audio"]["sample_rate"] == 32_000


def test_ltx_wide_taehv_matches_pinned_layout_and_timeline(monkeypatch):
    import vllm_omni.diffusion.models.ltx2.taehv as taehv_module

    decoder = LTXWideTAEHVDecoder()

    def fake_decoder(_decoder, value):
        return torch.zeros(value.shape[0], value.shape[1] * 8, 48, 1, 1)

    monkeypatch.setattr(taehv_module, "_apply_decoder_sequential", fake_decoder)
    video = decoder.decode_video(torch.zeros(1, 128, 16, 1, 1))

    assert decoder.decoder[1].weight.shape == (1024, 128, 3, 3)
    assert len(TAEHV_CHECKPOINT_SHA256) == 64
    assert video.shape == (1, 121, 3, 4, 4)


@pytest.mark.parametrize(("value", "expected"), [(None, 5.0), (5, 5.0), (10.0, 10.0)])
def test_h3_super_duration_contract(value, expected):
    extra = {} if value is None else {"duration": value}
    assert MiniMaxH3SuperDraftPipeline._resolve_super_duration(extra) == expected


def test_h3_super_duration_rejects_unpublished_shape():
    with pytest.raises(ValueError, match="5 or 10"):
        MiniMaxH3SuperDraftPipeline._resolve_super_duration({"duration": 8})


def test_h3_super_forward_pins_official_v01_sampling(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3_super as super_module

    pipeline = object.__new__(MiniMaxH3SuperDraftPipeline)
    pipeline._turbo_v01_lora_adapter_ids = {1}
    sampling = SimpleNamespace(
        lora_request=SimpleNamespace(lora_int_id=1),
        lora_scale=0.0625,
        extra_args={"duration": 5.0, "flow_shift": 6.0},
        height=None,
        width=None,
        fps=None,
        frame_rate=None,
        num_inference_steps=None,
    )
    request = SimpleNamespace(sampling_params=sampling)
    sentinel = object()
    monkeypatch.setattr(super_module.MiniMaxH3Pipeline, "forward", lambda _self, _request: sentinel)

    assert MiniMaxH3SuperDraftPipeline.forward(pipeline, request) is sentinel
    assert sampling.lora_scale == 0.0625
    assert sampling.extra_args == {
        "duration": 5.0,
        "task": "fl2va",
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }
    assert (sampling.height, sampling.width, sampling.fps, sampling.num_inference_steps) == (512, 896, 24, 5)


@pytest.mark.parametrize(
    ("duration", "expected_frames", "expected_latent_frames"),
    [(5.0, 124, 37), (10.0, 243, 72)],
)
def test_h3_super_preserves_h3_latent_timeline(
    duration,
    expected_frames,
    expected_latent_frames,
):
    pipeline = object.__new__(MiniMaxH3SuperDraftPipeline)
    sampling = SimpleNamespace(
        fps=24,
        extra_args={"duration": duration},
        num_frames=1,
        height=512,
        width=896,
    )

    _, _, frames, latent_frames, _ = pipeline._resolve_shape(
        "fl2va",
        sampling,
        Image.new("RGB", (896, 512)),
    )

    assert frames == expected_frames
    assert latent_frames == expected_latent_frames


@pytest.mark.parametrize(
    ("latent_frames", "expected_frames"),
    [(37, 124), (72, 243)],
)
def test_h3_super_taeh3_pads_decoded_timeline(
    monkeypatch,
    latent_frames,
    expected_frames,
):
    import vllm_omni.diffusion.models.minimax_h3.taeh3 as taeh3_module

    def fake_decoder(_decoder, value):
        batch = value.shape[0]
        return torch.zeros(batch, latent_frames * 4, 12, 1, 1)

    monkeypatch.setattr(taeh3_module, "_apply_decoder_parallel", fake_decoder)
    decoder = TAEH3Decoder()
    latent = torch.zeros(1, 24, latent_frames, 1, 1)

    video = decoder.decode_video(latent)

    assert video.shape == (1, 3, expected_frames, 2, 2)


def test_h3_super_taeh3_decoder_matches_pinned_checkpoint_layout():
    decoder = TAEH3Decoder()

    assert decoder.decoder[1].weight.shape == (256, 24, 3, 3)
    assert decoder.decoder[22].weight.shape == (12, 64, 3, 3)
    assert len(TAEH3_CHECKPOINT_SHA256) == 64


def test_h3_super_builds_released_compact_handoff():
    video = torch.linspace(0, 1, 124 * 3 * 2 * 3).reshape(1, 3, 124, 2, 3)
    audio = torch.linspace(-2, 2, 170_000).reshape(1, 1, -1)

    prepared_video, prepared_audio = _prepare_h3_super_handoff(
        video,
        audio,
        target_height=4,
        target_width=6,
    )

    assert prepared_video.shape == (1, 3, 121, 4, 6)
    assert prepared_video.dtype == torch.bfloat16
    assert prepared_video.is_contiguous()
    assert float(prepared_video.amin()) >= -1
    assert float(prepared_video.amax()) <= 1
    assert prepared_audio.shape == (1, 2, int(121 / 24 * 32_000))
    assert prepared_audio.dtype == torch.float32
    assert prepared_audio.is_contiguous()
    torch.testing.assert_close(prepared_audio[:, 0], prepared_audio[:, 1])


def test_h3_super_postprocess_preserves_compact_bfloat16_video():
    video = torch.zeros(1, 3, 1, 2, 2, dtype=torch.bfloat16)
    audio = torch.zeros(1, 2, 4)

    result = _minimax_h3_super_post_process((video, audio))

    assert result["video"].dtype == torch.bfloat16
    assert result["video"].shape == video.shape
    assert result["audio"].dtype == torch.float32


def test_h3_super_decode_uses_taeh3_and_official_audio_vae(monkeypatch):
    import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_super as super_module

    expected_video = torch.ones(1, 3, 124, 8, 12)
    expected_audio = torch.ones(1, 2, 100)
    expected_handoff = (torch.ones(1, 3, 1, 2, 2, dtype=torch.bfloat16), expected_audio)
    video_latent = torch.zeros(1, 24, 37, 1, 1)
    audio_latent = torch.zeros(1, 2, 50)
    captured: dict[str, Any] = {}

    def fake_prepare(video, audio):
        captured["video"] = video
        captured["audio"] = audio
        return expected_handoff

    monkeypatch.setattr(super_module, "_prepare_h3_super_handoff", fake_prepare)

    class FakeTAEH3:
        def decode_video(self, value):
            assert value is video_latent
            return expected_video

    class FakeAudioVAE:
        def decode_latent(self, value):
            assert value is audio_latent
            return expected_audio

    class Pipeline:
        taeh3_decoder = FakeTAEH3()
        audio_vae = FakeAudioVAE()

        @staticmethod
        def _component_on_device(component):
            class Context:
                def __enter__(self):
                    return component

                def __exit__(self, *args):
                    return False

            return Context()

    video, audio = MiniMaxH3SuperDraftPipeline.decode(
        Pipeline(),
        video_latent,
        audio_latent,
        height=6,
        width=10,
    )

    assert (video, audio) == expected_handoff
    assert captured["video"].shape == (1, 3, 124, 6, 10)
    assert captured["audio"] is expected_audio


def test_h3_refiner_skips_preprocessing_for_compact_handoff(monkeypatch):
    video = torch.full((1, 3, 1, 384, 672), -1.0, dtype=torch.bfloat16)
    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        vae=SimpleNamespace(dtype=torch.bfloat16),
    )

    def fail_interpolate(*args, **kwargs):
        raise AssertionError("compact handoff must not be resized twice")

    monkeypatch.setattr(torch.nn.functional, "interpolate", fail_interpolate)
    prepared, first_frame = LTX25H3RefinerPipeline._prepare_h3_video(
        pipeline,
        video,
        frame_count=1,
    )

    assert prepared.data_ptr() == video.data_ptr()
    assert prepared.dtype == torch.bfloat16
    assert prepared.shape == video.shape
    assert np.asarray(first_frame).max() == 0


def test_h3_to_ltx_bridge_preserves_raw_video_audio_and_source_frame():
    video = np.zeros((124, 8, 12, 3), dtype=np.float32)
    audio = np.zeros((1, 2, 100), dtype=np.float32)
    first_frame = object()
    source = SimpleNamespace(
        images=[video],
        multimodal_output={"audio": audio, "fps": 24, "audio_sample_rate": 32_000},
    )

    result = h3_to_ltx25_refiner(
        [source],
        {"prompt": "fixed prompt", "multi_modal_data": {"image": first_frame}},
    )

    assert result["prompt"] == "fixed prompt"
    additional = result["additional_information"]
    assert additional["h3_video"] is video
    assert additional["h3_audio"] is audio
    assert additional["h3_first_frame"] is first_frame


def test_h3_super_models_and_postprocessors_are_registered():
    assert _DIFFUSION_MODELS["MiniMaxH3SuperDraftPipeline"] == (
        "minimax_h3",
        "pipeline_minimax_h3_super",
        "MiniMaxH3SuperDraftPipeline",
    )
    assert _DIFFUSION_MODELS["LTX25H3RefinerPipeline"] == (
        "ltx2",
        "pipeline_ltx25_h3_refiner",
        "LTX25H3RefinerPipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["MiniMaxH3SuperDraftPipeline"] == ("get_minimax_h3_super_post_process_func")
    assert _DIFFUSION_POST_PROCESS_FUNCS["LTX25H3RefinerPipeline"] == ("get_ltx25_h3_refiner_post_process_func")


def test_h3_super_deploy_merges_two_independent_diffusion_stages():
    pipeline = OMNI_PIPELINES["minimax_h3_super"]
    deploy_path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "minimax_h3_super.yaml"
    stages = merge_pipeline_deploy(pipeline, load_deploy_config(deploy_path))

    assert len(stages) == 2
    assert stages[0].yaml_runtime["devices"] == "0"
    assert stages[1].yaml_runtime["devices"] == "1"
    assert stages[0].yaml_engine_args["model"] == "MiniMaxAI/MiniMax-H3"
    assert stages[0].yaml_engine_args["model_class_name"] == "MiniMaxH3SuperDraftPipeline"
    assert "diffusion_attention_config" not in stages[0].yaml_engine_args
    assert stages[0].yaml_engine_args["additional_config"]["taeh3_checkpoint"] == TAEH3_CHECKPOINT_URL
    assert stages[0].yaml_extras["default_sampling_params"]["height"] == 512
    assert stages[0].yaml_extras["default_sampling_params"]["num_inference_steps"] == 5
    assert stages[0].yaml_extras["default_sampling_params"]["seed"] == 50803
    assert stages[0].yaml_extras["default_sampling_params"]["lora_request"]["lora_name"] == "lx2v_4s_v01_544p"
    assert stages[0].yaml_extras["default_sampling_params"]["lora_scale"] == 0.0625
    assert stages[0].yaml_extras["default_sampling_params"]["extra_args"]["flow_shift"] == 12.0
    assert stages[1].yaml_engine_args["model"] == "Lightricks/LTX-2.5-Diffusers"
    assert stages[1].yaml_engine_args["model_class_name"] == "LTX25H3RefinerPipeline"
    assert "diffusion_attention_config" not in stages[1].yaml_engine_args
    assert stages[1].yaml_engine_args["enforce_eager"] is False
    assert stages[1].yaml_engine_args["diffusion_compile_dynamic"] is False
    assert stages[1].yaml_engine_args["additional_config"]["taehv_checkpoint"] == TAEHV_CHECKPOINT_URL
    assert stages[1].yaml_engine_args["additional_config"]["encode_output_video"] is True
    assert stages[1].yaml_extras["default_sampling_params"]["height"] == 768
    assert stages[1].yaml_extras["default_sampling_params"]["num_inference_steps"] == 3
    assert stages[1].yaml_extras["default_sampling_params"]["seed"] == 50803
