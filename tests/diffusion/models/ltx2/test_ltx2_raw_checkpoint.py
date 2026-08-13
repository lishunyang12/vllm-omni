# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.models.ltx2.ltx2_components import (
    _detect_vocoder_output_sample_rate,
    detect_ltx_model_version,
    get_ltx2_post_process_func,
    resolve_ltx_checkpoint_variant,
)
from vllm_omni.diffusion.models.ltx2.ltx2_raw_checkpoint import (
    convert_ltx2_audio_vae_config,
    convert_ltx2_connectors_config,
    convert_ltx2_diffusion_video_vae_config,
    convert_ltx2_transformer_config,
    convert_ltx2_upsampler_config,
    convert_ltx2_video_vae_config,
    convert_ltx2_vocoder_config,
    infer_ltx2_checkpoint_variant,
    is_ltx2_raw_checkpoint_layout,
    load_ltx2_embedded_gemma_assets,
    map_ltx2_audio_file_weight,
    map_ltx2_text_encoder_file_weight,
    map_ltx2_transformer_file_weight,
    map_ltx2_upsampler_weight,
    map_ltx2_video_vae_weight,
    parse_json_metadata,
    read_safetensors_metadata,
    resolve_ltx2_raw_checkpoint_layout,
)
from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model, is_ltx25_raw_checkpoint

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_diffusers_loader_cold_import_is_cycle_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _write_checkpoint(path: Path, metadata=None, tensors=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors or {"dummy": torch.zeros(1)}, path, metadata=metadata or {})


def _make_layout(root: Path):
    for variant in ("dev", "distilled"):
        _write_checkpoint(root / f"diffusion_models/ltx-2.5-22b-{variant}-transformer-bf16.safetensors")
    _write_checkpoint(root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    _write_checkpoint(root / "vae/ltx-2.5-audio-vae-bf16.safetensors")
    _write_checkpoint(root / "vae/ltx-2.5-video-vae-bf16.safetensors")
    _write_checkpoint(root / "vae/ltx-2.5-video-vae-conv-bf16.safetensors")
    _write_checkpoint(root / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")


def test_official_raw_checkpoint_defaults_to_distilled_two_stage(tmp_path):
    _make_layout(tmp_path)

    assert is_ltx25_raw_checkpoint(str(tmp_path))
    assert is_diffusion_model(str(tmp_path))
    assert resolve_model_class_name(str(tmp_path)) == "LTX2TwoStagePipeline"
    assert detect_ltx_model_version(str(tmp_path)) == "2.5"
    assert resolve_ltx_checkpoint_variant("two_stage", "2.5", None) == "distilled"
    assert resolve_ltx_checkpoint_variant("one_stage", "2.5", None) == "full"

    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()
    assert config.model_class_name == "LTX2TwoStagePipeline"
    assert config.task_type is None


def test_layout_selection_is_explicit_with_complete_snapshot(tmp_path):
    _make_layout(tmp_path)

    distilled = resolve_ltx2_raw_checkpoint_layout(
        tmp_path,
        checkpoint_variant="distilled",
        video_decoder_type="conv",
        require_latent_upsampler=True,
    )
    full = resolve_ltx2_raw_checkpoint_layout(
        tmp_path,
        checkpoint_variant="full",
        video_decoder_type="diffusion",
    )

    assert distilled.transformer.name == "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    assert distilled.video_vae.name == "ltx-2.5-video-vae-conv-bf16.safetensors"
    assert distilled.checkpoint_variant == "distilled"
    assert full.transformer.name == "ltx-2.5-22b-dev-transformer-bf16.safetensors"
    assert full.video_vae.name == "ltx-2.5-video-vae-bf16.safetensors"
    assert is_ltx2_raw_checkpoint_layout(tmp_path, checkpoint_variant="full", video_decoder_type="diffusion")


def test_layout_requires_selected_file_and_two_stage_upsampler(tmp_path):
    _make_layout(tmp_path)
    (tmp_path / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors").unlink()

    assert not is_ltx2_raw_checkpoint_layout(tmp_path, checkpoint_variant="full", video_decoder_type="conv")
    (tmp_path / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors").unlink()
    with pytest.raises(ValueError, match="two-stage execution requires"):
        resolve_ltx2_raw_checkpoint_layout(
            tmp_path,
            checkpoint_variant="distilled",
            video_decoder_type="conv",
            require_latent_upsampler=True,
        )


def test_variant_inference_is_validation_only_and_ignores_scheduler():
    scheduler_only = {"config": json.dumps({"scheduler": {"sampler": "LinearQuadratic"}})}

    assert infer_ltx2_checkpoint_variant(scheduler_only) == "unknown"
    assert infer_ltx2_checkpoint_variant(scheduler_only, filename="ltx-distilled.safetensors") == "distilled"
    assert infer_ltx2_checkpoint_variant({}, filename="ltx-dev.safetensors") == "full"
    assert infer_ltx2_checkpoint_variant({"checkpoint_variant": "full"}, filename="distilled.safetensors") == "full"


def test_metadata_and_embedded_assets_read_only_headers_and_asset_tensors(tmp_path):
    path = tmp_path / "packed-gemma.safetensors"
    _write_checkpoint(
        path,
        metadata={"gemma_config": json.dumps({"model_type": "gemma4_unified"})},
        tensors={
            "model.layers.0.weight": torch.ones(2),
            "tokenizer_json": torch.tensor(list(b'{"version":"1.0"}'), dtype=torch.uint8),
            "hf_asset__chat_template.jinja": torch.tensor(list(b"{{ message }}"), dtype=torch.int8),
        },
    )

    metadata = read_safetensors_metadata(path)
    assets = load_ltx2_embedded_gemma_assets(path)

    assert parse_json_metadata(metadata, "gemma_config")["model_type"] == "gemma4_unified"
    assert assets == {"tokenizer.json": b'{"version":"1.0"}', "chat_template.jinja": b"{{ message }}"}


@pytest.mark.parametrize(
    ("raw_key", "mapped_key"),
    [
        ("model.diffusion_model.patchify_proj.weight", "transformer.proj_in.weight"),
        ("model.diffusion_model.audio_patchify_proj.bias", "transformer.audio_proj_in.bias"),
        (
            "model.diffusion_model.av_ca_video_scale_shift_adaln_single.weight",
            "transformer.av_cross_attn_video_scale_shift.weight",
        ),
        (
            "model.diffusion_model.transformer_blocks.0.attn1.q_norm.weight",
            "transformer.transformer_blocks.0.attn1.norm_q.weight",
        ),
        (
            "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn.k_norm.weight",
            "connectors.video_connector.transformer_blocks.0.attn.norm_k.weight",
        ),
    ],
)
def test_transformer_file_key_mapping(raw_key, mapped_key):
    assert map_ltx2_transformer_file_weight(raw_key) == mapped_key


@pytest.mark.parametrize(
    ("raw_key", "mapped_key"),
    [
        ("text_embedding_projection.video_aggregate_embed.weight", "connectors.video_text_proj_in.weight"),
        (
            "model.layers.0.self_attn.q_proj.weight",
            "text_encoder.model.language_model.layers.0.self_attn.q_proj.weight",
        ),
        ("vision_model.patch_embedding.weight", "text_encoder.model.vision_embedder.patch_embedding.weight"),
        ("multi_modal_projector.proj.weight", "text_encoder.model.embed_vision.proj.weight"),
        ("audio_projector.proj.weight", "text_encoder.model.embed_audio.proj.weight"),
        ("tokenizer_json", None),
    ],
)
def test_packed_text_encoder_key_mapping(raw_key, mapped_key):
    assert map_ltx2_text_encoder_file_weight(raw_key) == mapped_key


def test_audio_and_vocoder_key_mapping():
    assert map_ltx2_audio_file_weight("audio_vae.per_channel_statistics.mean-of-means") == "audio_vae.latents_mean"
    assert (
        map_ltx2_audio_file_weight("vocoder.vocoder.ups.0.resblocks.0.conv_pre.weight")
        == "vocoder.vocoder.upsamplers.0.resnets.0.conv_in.weight"
    )
    assert (
        map_ltx2_audio_file_weight("vocoder.bwe_generator.downsample.lowpass.filter")
        == "vocoder.bwe_generator.downsample.filter"
    )


@pytest.mark.parametrize(
    ("raw_key", "mapped_key"),
    [
        ("encoder.down_blocks.0.res_blocks.0.conv1.weight", "vae.encoder.down_blocks.0.resnets.0.conv1.weight"),
        ("encoder.down_blocks.7.conv.conv.weight", "vae.encoder.down_blocks.3.downsamplers.0.conv.conv.weight"),
        ("encoder.down_blocks.8.res_blocks.0.weight", "vae.encoder.mid_block.resnets.0.weight"),
        ("decoder.up_blocks.0.res_blocks.0.weight", "vae.decoder.mid_block.resnets.0.weight"),
        ("decoder.up_blocks.7.conv.conv.weight", "vae.decoder.up_blocks.3.upsamplers.0.conv.conv.weight"),
        ("decoder.up_blocks.8.res_blocks.0.weight", "vae.decoder.up_blocks.3.resnets.0.weight"),
        ("per_channel_statistics.std-of-means", "vae.latents_std"),
        ("per_channel_statistics.channel", None),
    ],
)
def test_conv_video_vae_key_mapping(raw_key, mapped_key):
    assert map_ltx2_video_vae_weight(raw_key) == mapped_key


def _raw_transformer_config():
    return {
        "num_layers": 48,
        "positional_embedding_max_pos": [20, 2048, 2048],
        "audio_positional_embedding_max_pos": [20],
        "qk_norm": "rms_norm",
        "frequencies_precision": "float64",
        "connector_num_attention_heads": 32,
        "connector_attention_head_dim": 128,
        "connector_num_layers": 8,
        "connector_num_learnable_registers": 128,
        "audio_connector_num_attention_heads": 32,
        "audio_connector_attention_head_dim": 64,
        "connector_positional_embedding_max_pos": [4096],
        "cross_attention_dim": 4096,
        "audio_cross_attention_dim": 2048,
    }


def _raw_video_vae_config():
    residuals = [4, 6, 4, 2, 2]
    encoder_compressions = ["compress_space_res", "compress_time_res", "compress_all_res", "compress_all_res"]
    decoder_compressions = ["compress_space", "compress_time", "compress_all", "compress_all"]

    def interleave(names, multipliers):
        result = []
        for layers, name, multiplier in zip(residuals, names, multipliers, strict=False):
            result.extend((("res_x", {"num_layers": layers}), (name, {"multiplier": multiplier})))
        result.append(("res_x", {"num_layers": residuals[-1]}))
        return result

    return {
        "_class_name": "CausalVideoAutoencoder",
        "causal_decoder": False,
        "decoder_base_channels": 128,
        "decoder_blocks": interleave(decoder_compressions, [2, 2, 1, 2]),
        "encoder_base_channels": 128,
        "encoder_blocks": interleave(encoder_compressions, [2, 2, 2, 1]),
        "in_channels": 3,
        "latent_channels": 128,
        "out_channels": 3,
        "patch_size": 4,
        "spatial_padding_mode": "zeros",
    }


def test_transformer_connector_video_and_diffvae_config_conversion():
    raw_transformer = _raw_transformer_config()
    transformer = convert_ltx2_transformer_config(raw_transformer)
    connectors = convert_ltx2_connectors_config(
        raw_transformer, {"text_config": {"hidden_size": 3072, "num_hidden_layers": 35}}
    )
    video_vae = convert_ltx2_video_vae_config(_raw_video_vae_config())
    diffvae = convert_ltx2_diffusion_video_vae_config(
        {"_class_name": "DiffusionVideoAutoencoder", "latent_channels": 128}
    )

    assert transformer["qk_norm"] == "rms_norm_across_heads"
    assert transformer["base_width"] == 2048
    assert connectors["caption_channels"] == 3072
    assert connectors["text_proj_in_factor"] == 36
    assert connectors["causal_temporal_positioning"] is False
    assert video_vae["block_out_channels"] == (256, 512, 1024, 1024)
    assert video_vae["decoder_block_out_channels"] == (256, 512, 512, 1024)
    assert diffvae["decoder_stage_depths"] == (4, 6, 4, 2, 8)


def test_audio_vocoder_and_upsampler_config_conversion():
    audio = {
        "model": {
            "params": {
                "ddconfig": {
                    "attn_resolutions": [],
                    "causality_axis": "height",
                    "ch": 128,
                    "ch_mult": [1, 2, 4],
                    "double_z": True,
                    "dropout": 0.0,
                    "in_channels": 2,
                    "mel_bins": 64,
                    "mid_block_add_attention": False,
                    "norm_type": "pixel",
                    "num_res_blocks": 2,
                    "out_ch": 2,
                    "resolution": 256,
                    "z_channels": 8,
                },
                "sampling_rate": 16000,
            }
        },
        "preprocessing": {"stft": {"causal": True, "hop_length": 160}},
    }
    generator = {
        "activation": "snakebeta",
        "resblock_dilation_sizes": [[1, 3, 5]],
        "resblock_kernel_sizes": [3],
        "stereo": True,
        "upsample_initial_channel": 1536,
        "upsample_kernel_sizes": [11, 4],
        "upsample_rates": [5, 2],
        "use_bias_at_final": False,
        "use_tanh_at_final": False,
    }
    bwe = {
        **generator,
        "hop_length": 80,
        "input_sampling_rate": 16000,
        "n_fft": 512,
        "num_mels": 64,
        "output_sampling_rate": 48000,
        "upsample_initial_channel": 512,
        "win_size": 512,
    }

    audio_config = convert_ltx2_audio_vae_config(audio)
    vocoder_config = convert_ltx2_vocoder_config({"vocoder": generator, "bwe": bwe})
    upsampler_config = convert_ltx2_upsampler_config(
        {"_class_name": "LatentUpsampler", "spatial_scale": 2.0, "rational_resampler": False}
    )

    assert audio_config["mel_hop_length"] == 160
    assert vocoder_config["output_sampling_rate"] == 48000
    assert upsampler_config == {"rational_spatial_scale": 2.0, "use_rational_resampler": False}
    assert map_ltx2_upsampler_weight("res_blocks.0.weight") == "latent_upsampler.res_blocks.0.weight"


def test_raw_materialization_pins_revision_and_exact_selected_files(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_raw_checkpoint

    snapshot = tmp_path / "snapshot"
    _make_layout(snapshot)
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(ltx2_raw_checkpoint, "snapshot_download", fake_snapshot_download)
    layout = ltx2_raw_checkpoint.materialize_ltx2_raw_checkpoint(
        "Lightricks/LTX-2.5",
        checkpoint_variant="distilled",
        video_decoder_type="conv",
        require_latent_upsampler=True,
        revision="pinned-revision",
        cache_dir=tmp_path / "cache",
    )

    assert layout.root == snapshot
    assert captured["revision"] == "pinned-revision"
    assert set(captured["allow_patterns"]) == {
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    }


def test_raw_weight_sources_and_adapter_are_strict(tmp_path):
    from types import SimpleNamespace

    from vllm_omni.diffusion.model_loader.checkpoint_adapters.ltx2 import LTX2RawCheckpointAdapter
    from vllm_omni.diffusion.models.ltx2.ltx2_components import _build_ltx2_raw_weight_sources

    _make_layout(tmp_path)
    layout = resolve_ltx2_raw_checkpoint_layout(
        tmp_path,
        checkpoint_variant="distilled",
        video_decoder_type="conv",
        require_latent_upsampler=True,
    )
    sources = _build_ltx2_raw_weight_sources(layout, "pinned-revision", include_latent_upsampler=True)

    assert len(sources) == 5
    assert all(source.model_or_path == str(tmp_path) for source in sources)
    assert all(source.revision == "pinned-revision" for source in sources)
    assert all(source.prefix == "" and source.fall_back_to_pt is False for source in sources)

    one_stage_sources = _build_ltx2_raw_weight_sources(layout, "pinned-revision", include_latent_upsampler=False)
    assert len(one_stage_sources) == 4
    assert all(source.subfolder != "latent_upscale_models" for source in one_stage_sources)

    text_source = next(source for source in sources if source.subfolder == "text_encoders")
    model = SimpleNamespace(_ltx2_raw_checkpoint=True)
    assert LTX2RawCheckpointAdapter.is_compatible(model, text_source, None, True)
    adapter = LTX2RawCheckpointAdapter(model, text_source)
    tensor = torch.ones(1)
    assert list(
        adapter.adapt(
            [
                ("tokenizer_json", torch.tensor([1], dtype=torch.uint8)),
                ("model.norm.weight", tensor),
            ]
        )
    ) == [("text_encoder.model.language_model.norm.weight", tensor)]
    with pytest.raises(ValueError, match="Unmapped required tensor"):
        list(adapter.adapt([("unexpected.required.weight", tensor)]))

    compat_model = torch.nn.Module()
    compat_model._ltx2_raw_checkpoint = True
    compat_model.text_encoder = torch.nn.Module()
    compat_model.text_encoder.model = torch.nn.Module()
    compat_model.text_encoder.model.embed_vision = torch.nn.Module()
    compat_model.text_encoder.model.embed_vision.patch_dense = torch.nn.Linear(1, 1)
    compat_model.text_encoder.model.embed_vision.multimodal_embedder = torch.nn.Module()
    compat_model.text_encoder.model.embed_vision.multimodal_embedder.embedding_projection = torch.nn.Linear(
        1, 1, bias=False
    )
    compat_adapter = LTX2RawCheckpointAdapter(compat_model, text_source)
    mapped = list(
        compat_adapter.adapt(
            [
                ("vision_model.patch_dense.weight", tensor),
                ("multi_modal_projector.embedding_projection.weight", tensor),
            ]
        )
    )
    assert [name for name, _ in mapped] == [
        "text_encoder.model.embed_vision.patch_dense.weight",
        "text_encoder.model.embed_vision.multimodal_embedder.embedding_projection.weight",
    ]


def test_raw_component_wiring_uses_embedded_configs_and_convvae(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.ltx2 import ltx2_components

    _make_layout(tmp_path)
    layout = resolve_ltx2_raw_checkpoint_layout(
        tmp_path,
        checkpoint_variant="distilled",
        video_decoder_type="conv",
        require_latent_upsampler=True,
    )
    metadata = SimpleNamespace(
        model_version="2.5.0",
        transformer={"patch_size": 1},
        connectors={"component": "connectors"},
        scheduler={"num_train_timesteps": 1000},
        gemma={"model_type": "fake_gemma"},
        audio_vae={"component": "audio_vae"},
        vocoder={"component": "vocoder"},
        video_vae={"component": "conv_vae"},
        latent_upsampler={"component": "upsampler"},
    )
    calls = {}

    class FakeConfig:
        @classmethod
        def from_dict(cls, config):
            calls["gemma_config"] = config
            return config

    class FakeFactory:
        @classmethod
        def from_config(cls, config, **kwargs):
            calls.setdefault("components", []).append((config, kwargs))
            return SimpleNamespace(config=SimpleNamespace(**config))

    class FakeVAE(FakeFactory):
        @classmethod
        def from_config(cls, config, **kwargs):
            result = super().from_config(config, **kwargs)
            result.init_distributed = lambda: calls.setdefault("vae_distributed", True)
            return result

    monkeypatch.setattr(ltx2_components, "materialize_ltx2_raw_checkpoint", lambda *args, **kwargs: layout)
    monkeypatch.setattr(ltx2_components, "inspect_ltx2_raw_checkpoint", lambda _layout: metadata)
    monkeypatch.setattr(ltx2_components, "load_ltx2_embedded_gemma_assets", lambda _path: {"tokenizer.json": b"{}"})
    monkeypatch.setattr(ltx2_components, "_build_ltx2_raw_tokenizer", lambda _assets: "tokenizer")
    monkeypatch.setattr(ltx2_components, "CONFIG_MAPPING", {"fake_gemma": FakeConfig})
    monkeypatch.setattr(ltx2_components, "LTX2TextConnectors", FakeFactory)
    monkeypatch.setattr(ltx2_components, "DistributedAutoencoderKLLTX2Video", FakeVAE)
    monkeypatch.setattr(ltx2_components, "AutoencoderKLLTX2Audio", FakeFactory)
    monkeypatch.setattr(ltx2_components, "LTX2VocoderWithBWE", FakeFactory)
    monkeypatch.setattr(ltx2_components, "LTX2LatentUpsamplerModel", FakeFactory)
    monkeypatch.setattr(ltx2_components, "_install_connector_attention", lambda _component: None)
    monkeypatch.setattr(ltx2_components, "_place_aux_components", lambda _pipeline: None)
    monkeypatch.setattr(
        ltx2_components,
        "create_transformer_from_config",
        lambda config, quant_config=None: SimpleNamespace(config=SimpleNamespace(**config)),
    )

    class FakeTextEncoder:
        @classmethod
        def from_config(cls, config, **kwargs):
            calls["text_encoder"] = (config, kwargs)
            return SimpleNamespace(config=SimpleNamespace())

    pipe = SimpleNamespace(
        component_profile=SimpleNamespace(
            text_encoder_cls=FakeTextEncoder,
            resident_modules=("vocoder", "latent_upsampler"),
            scheduler_use_dynamic_shifting=False,
        ),
        checkpoint_variant="distilled",
        od_config=SimpleNamespace(),
    )
    od_config = SimpleNamespace(
        model=str(tmp_path),
        revision="pinned-revision",
        quantization_config=None,
    )
    pipe.od_config = od_config
    ltx2_components._initialize_raw_pipeline_components(pipe, od_config, torch.bfloat16)

    assert pipe._ltx2_raw_checkpoint is True
    assert pipe.tokenizer == "tokenizer"
    assert calls["text_encoder"][1] == {"dtype": torch.bfloat16}
    assert calls["vae_distributed"] is True
    assert any(config == {"component": "conv_vae"} for config, _kwargs in calls["components"])
    assert len(pipe.weights_sources) == 5


def test_raw_diffvae_is_explicitly_rejected_before_loading():
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.ltx2 import ltx2_components

    pipe = SimpleNamespace(component_profile=SimpleNamespace(text_encoder_cls=object, resident_modules=()))
    config = SimpleNamespace(ltx2_video_decoder_type="diffusion")
    with pytest.raises(ValueError, match="DiffVAE tensor mapping is not implemented"):
        ltx2_components._initialize_raw_pipeline_components(pipe, config, torch.bfloat16)


def test_detect_ltx_version_propagates_revision(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.ltx2 import ltx2_components

    index = tmp_path / "model_index.json"
    index.write_text(json.dumps({"text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"]}))
    captured = {}

    def fake_download(*, repo_id, filename, revision):
        captured.update(repo_id=repo_id, filename=filename, revision=revision)
        return str(index)

    monkeypatch.setattr(ltx2_components, "is_ltx25_raw_checkpoint", lambda model, revision=None: False)
    monkeypatch.setattr(ltx2_components, "hf_hub_download", fake_download)

    assert ltx2_components.detect_ltx_model_version("org/model", revision="pinned-revision") == "2.5"
    assert captured == {"repo_id": "org/model", "filename": "model_index.json", "revision": "pinned-revision"}


def test_vocoder_sample_rate_uses_official_raw_constant_without_download(monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_components

    captured = {}

    def fake_is_raw(model, revision=None):
        captured.update(model=model, revision=revision)
        return True

    monkeypatch.setattr(ltx2_components, "is_ltx25_raw_checkpoint", fake_is_raw)
    monkeypatch.setattr(
        ltx2_components,
        "hf_hub_download",
        lambda *args, **kwargs: pytest.fail("raw sample-rate detection must not download weights"),
    )

    assert _detect_vocoder_output_sample_rate("Lightricks/LTX-2.5", revision="pinned-revision") == 48_000
    assert captured == {"model": "Lightricks/LTX-2.5", "revision": "pinned-revision"}


def test_converted_vocoder_sample_rate_download_pins_revision(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.ltx2 import ltx2_components

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_sampling_rate": 44_100}))
    captured = {}

    monkeypatch.setattr(ltx2_components, "is_ltx25_raw_checkpoint", lambda model, revision=None: False)

    def fake_download(repo_id, filename, revision):
        captured.update(repo_id=repo_id, filename=filename, revision=revision)
        return str(config_path)

    monkeypatch.setattr(ltx2_components, "hf_hub_download", fake_download)

    assert _detect_vocoder_output_sample_rate("org/converted", revision="pinned-revision") == 44_100
    assert captured == {
        "repo_id": "org/converted",
        "filename": "vocoder/config.json",
        "revision": "pinned-revision",
    }


def test_post_process_sample_rate_propagates_revision(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.ltx2 import ltx2_components

    captured = {}

    def fake_detect(model, revision=None):
        captured.update(model=model, revision=revision)
        return 48_000

    monkeypatch.setattr(ltx2_components, "_detect_vocoder_output_sample_rate", fake_detect)
    post_process = get_ltx2_post_process_func(SimpleNamespace(model="org/model", revision="pinned-revision"))
    result = post_process((torch.zeros(1), torch.zeros(1)))

    assert result["audio_sample_rate"] == 48_000
    assert captured == {"model": "org/model", "revision": "pinned-revision"}
