# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E accuracy guards against pinned official Lightricks LTX references.

The LTX-2/2.3 comparison runs both runtimes through PyTorch SDPA and uses
``max_batch_size=4`` in the official reference to match Omni's fused guidance
batch. Video and audio guidance use the official non-HQ one-stage defaults;
only the generation shape and step count are reduced for CI runtime.
LTX-2.5 runs the official split-artifact pipelines with connector weights from
the same Diffusers checkpoint under test. The distilled case covers the fixed
two-stage schedule; the Full/SFT case covers raw dev weights and an explicit
shared one-stage schedule.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from tests.e2e.accuracy.helpers import reset_artifact_dir
from tests.helpers.mark import hardware_test

OFFICIAL_REPOSITORY = "https://github.com/Lightricks/LTX-2.git"
OFFICIAL_REVISION = "9377758131b1ffde4b7f766804590a6617bf2ab9"
LTX25_OFFICIAL_REVISION = "7954dcb0d986bdc36ef272564a9789ade07fcc65"
LTX25_OMNI_MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
LTX25_OMNI_MODEL_REVISION = "a6de4b5354f078db24d9cf4778c14846788aea3d"
LTX25_OFFICIAL_MODEL_ID = "Lightricks/LTX-2.5"
LTX25_OFFICIAL_MODEL_REVISION = "8a4ff96f581e72bedc1b44367581c49d544a05f1"
LTX25_OFFICIAL_COMMON_FILES = {
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
}
LTX25_OFFICIAL_VARIANT_FILES = {
    "distilled": {
        "transformer": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "spatial_upsampler": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    },
    "full": {
        "transformer": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    },
}
# Version selected by the pinned official source's uv.lock. Keep it isolated
# from Omni's runtime and development dependencies.
OFFICIAL_OPENIMAGEIO_VERSION = "3.1.11.0"
PROMPT = (
    "A space shuttle launches vertically above a desert launch pad. Bright exhaust flames and a dense white "
    "plume billow beneath it while the camera remains fixed."
)
LTX25_PROMPT = "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside."
NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, "
    "background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting "
    "direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, "
    "incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, "
    "silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect "
    "dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural "
    "transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, "
    "stylized filters, or AI artifacts."
)

# Both runtimes use PyTorch SDPA with the current Torch dispatch defaults.
ATTENTION_BACKEND = "torch_sdpa"
VIDEO_SSIM_MEAN_THRESHOLD = 0.95
VIDEO_SSIM_MIN_THRESHOLD = 0.90
VIDEO_PSNR_MEAN_THRESHOLD = 30.0
AUDIO_RELATIVE_L2_THRESHOLD = 0.2
AUDIO_COSINE_THRESHOLD = 0.95

# Decoded-output gates for the pinned official/Omni two-stage SDPA comparison.
# The metrics artifact preserves the exact observed values.
LTX25_VIDEO_SSIM_MEAN_THRESHOLD = 0.995
LTX25_VIDEO_SSIM_MIN_THRESHOLD = 0.99
LTX25_VIDEO_PSNR_MEAN_THRESHOLD = 40.0
LTX25_AUDIO_RELATIVE_L2_THRESHOLD = 0.3
LTX25_AUDIO_COSINE_THRESHOLD = 0.95

# Full/SFT exercises CFG/STG and independently converted dev weights, so use
# the established one-stage decoded-output gates rather than the tighter
# distilled fixed-schedule gates.
LTX25_FULL_VIDEO_SSIM_MEAN_THRESHOLD = VIDEO_SSIM_MEAN_THRESHOLD
LTX25_FULL_VIDEO_SSIM_MIN_THRESHOLD = VIDEO_SSIM_MIN_THRESHOLD
LTX25_FULL_VIDEO_PSNR_MEAN_THRESHOLD = VIDEO_PSNR_MEAN_THRESHOLD
LTX25_FULL_AUDIO_RELATIVE_L2_THRESHOLD = 0.3
LTX25_FULL_AUDIO_COSINE_THRESHOLD = AUDIO_COSINE_THRESHOLD


def test_ltx_reference_runner_unwraps_flattened_pipeline_output() -> None:
    from vllm_omni.outputs import OmniRequestOutput

    from .run_ltx_reference import _unwrap_omni_output

    frame = object()
    audio = object()
    output = OmniRequestOutput(
        stage_id=0,
        images=[frame],
        _multimodal_output={"audio": audio, "audio_sample_rate": 48_000},
    )

    frames, actual_audio, sample_rate = _unwrap_omni_output(output)

    assert frames is frame
    assert actual_audio is audio
    assert sample_rate == 48_000


@dataclass(frozen=True)
class LTXAccuracyCase:
    name: str
    model_id: str
    model_revision: str
    model_env: str
    model_class_name: str
    checkpoint_repo: str
    checkpoint_filename: str
    checkpoint_revision: str
    checkpoint_env: str
    stg_block: int
    prompt: str = PROMPT
    image_repo: str | None = None
    image_filename: str | None = None
    image_revision: str | None = None


CASES = (
    LTXAccuracyCase(
        name="ltx2",
        model_id="Lightricks/LTX-2",
        model_revision="47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
        model_env="VLLM_TEST_LTX2_MODEL",
        model_class_name="LTX2Pipeline",
        checkpoint_repo="Lightricks/LTX-2",
        checkpoint_filename="ltx-2-19b-dev.safetensors",
        checkpoint_revision="47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
        checkpoint_env="VLLM_TEST_LTX2_OFFICIAL_CHECKPOINT",
        stg_block=29,
    ),
    LTXAccuracyCase(
        name="ltx2_3",
        model_id="diffusers/LTX-2.3-Diffusers",
        model_revision="8eee8edcf067e838b843f926ec4d4cc9b2be1aaf",
        model_env="VLLM_TEST_LTX23_MODEL",
        model_class_name="LTX2Pipeline",
        checkpoint_repo="Lightricks/LTX-2.3",
        checkpoint_filename="ltx-2.3-22b-dev.safetensors",
        checkpoint_revision="4229404625088d21c4f112eb640fb04a0900ee25",
        checkpoint_env="VLLM_TEST_LTX23_OFFICIAL_CHECKPOINT",
        stg_block=28,
    ),
    LTXAccuracyCase(
        name="ltx2_3_i2v",
        model_id="diffusers/LTX-2.3-Diffusers",
        model_revision="8eee8edcf067e838b843f926ec4d4cc9b2be1aaf",
        model_env="VLLM_TEST_LTX23_MODEL",
        model_class_name="LTX2Pipeline",
        checkpoint_repo="Lightricks/LTX-2.3",
        checkpoint_filename="ltx-2.3-22b-dev.safetensors",
        checkpoint_revision="4229404625088d21c4f112eb640fb04a0900ee25",
        checkpoint_env="VLLM_TEST_LTX23_OFFICIAL_CHECKPOINT",
        stg_block=28,
        image_repo="huggingface/documentation-images",
        image_filename="diffusers/svd/rocket.png",
        image_revision="645d8364f0c7a101180b364811b5a11a362e4010",
    ),
)


def _run(command: list[str], *, env: dict[str, str], timeout: int = 1800) -> None:
    start = time.perf_counter()
    subprocess.run(command, env=env, timeout=timeout, check=True)
    print(f"{' '.join(command[:3])} finished in {time.perf_counter() - start:.1f}s")


def _clone_pinned_source(root: Path, repository: str, revision: str) -> None:
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "remote", "add", "origin", repository], check=True)
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(3):
        try:
            subprocess.run(
                ["git", "-C", str(root), "fetch", "--depth", "1", "origin", revision],
                check=True,
            )
            last_error = None
            break
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    subprocess.run(["git", "-C", str(root), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def _official_source(artifact_root: Path) -> tuple[Path, str]:
    repository = os.environ.get("VLLM_TEST_LTX_OFFICIAL_REPOSITORY", OFFICIAL_REPOSITORY)
    revision = os.environ.get("VLLM_TEST_LTX_OFFICIAL_REVISION", OFFICIAL_REVISION)
    configured_root = os.environ.get("VLLM_TEST_LTX_OFFICIAL_ROOT")
    root = Path(configured_root) if configured_root else artifact_root / f"official-source-{revision[:12]}"
    actual_revision = _git_revision(root) if root.exists() else None
    if actual_revision != revision and configured_root:
        raise AssertionError(f"Official source revision mismatch: {actual_revision} != {revision}")
    if actual_revision != revision:
        if root.exists():
            shutil.rmtree(root)
        _clone_pinned_source(root, repository, revision)
        actual_revision = _git_revision(root)
    assert actual_revision == revision, f"Official source revision mismatch: {actual_revision} != {revision}"
    return root, revision


def _ltx25_official_source(artifact_root: Path) -> tuple[Path, str]:
    repository = os.environ.get("VLLM_TEST_LTX25_OFFICIAL_REPOSITORY", OFFICIAL_REPOSITORY)
    revision = os.environ.get("VLLM_TEST_LTX25_OFFICIAL_REVISION", LTX25_OFFICIAL_REVISION)
    configured_root = os.environ.get("VLLM_TEST_LTX25_OFFICIAL_ROOT")
    root = Path(configured_root) if configured_root else artifact_root / f"official-source-{revision[:12]}"
    actual_revision = _git_revision(root) if root.exists() else None
    if actual_revision != revision and configured_root:
        raise AssertionError(f"Official LTX-2.5 source revision mismatch: {actual_revision} != {revision}")
    if actual_revision != revision:
        if root.exists():
            shutil.rmtree(root)
        _clone_pinned_source(root, repository, revision)
        actual_revision = _git_revision(root)
    assert actual_revision == revision, f"Official LTX-2.5 source revision mismatch: {actual_revision} != {revision}"
    return root, revision


def _official_runner_prefix() -> list[str]:
    """Run the reference with its missing binary dependency isolated from CI."""
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to run the pinned official LTX reference"
    return [
        uv,
        "run",
        "--no-project",
        "--with",
        f"openimageio=={OFFICIAL_OPENIMAGEIO_VERSION}",
        "--python",
        sys.executable,
        "python",
    ]


def _resolve_model(case: LTXAccuracyCase) -> Path:
    configured_model = os.environ.get(case.model_env)
    if configured_model and Path(configured_model).exists():
        return Path(configured_model)
    model_id = configured_model or case.model_id
    revision = os.environ.get(f"{case.model_env}_REVISION")
    if revision is None and model_id == case.model_id:
        revision = case.model_revision
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=[
                "model_index.json",
                "audio_vae/*",
                "connectors/config.json",
                "connectors/diffusion_pytorch_model.safetensors.index.json",
                "connectors/diffusion_pytorch_model-*.safetensors",
                "processor/*",
                "scheduler/*",
                "text_encoder/config.json",
                "text_encoder/generation_config.json",
                "text_encoder/model*",
                "tokenizer/*",
                "transformer/*",
                "vae/*",
                "vocoder/*",
            ],
        )
    )


def _resolve_ltx25_omni_model(*, transformer_subfolder: str = "transformer") -> tuple[Path, str, str | None]:
    if transformer_subfolder not in {"transformer", "transformer_full"}:
        raise ValueError(f"Unsupported LTX-2.5 transformer subfolder: {transformer_subfolder!r}")
    configured_model = os.environ.get("VLLM_TEST_LTX25_MODEL")
    configured_revision = os.environ.get("VLLM_TEST_LTX25_MODEL_REVISION")
    if configured_model and Path(configured_model).exists():
        model = Path(configured_model).resolve()
        snapshot_revision = model.name if model.parent.name == "snapshots" else None
        assert (model / transformer_subfolder).is_dir(), f"LTX-2.5 component not found: {model / transformer_subfolder}"
        return model, configured_model, configured_revision or snapshot_revision
    model_id = configured_model or LTX25_OMNI_MODEL_ID
    revision = configured_revision
    if revision is None and model_id == LTX25_OMNI_MODEL_ID:
        revision = LTX25_OMNI_MODEL_REVISION
    model = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=[
                "model_index.json",
                "audio_vae/*",
                "connectors/config.json",
                "connectors/diffusion_pytorch_model.safetensors.index.json",
                "connectors/diffusion_pytorch_model-*.safetensors",
                "scheduler/*",
                "text_encoder/config.json",
                "text_encoder/generation_config.json",
                "text_encoder/model*",
                "tokenizer/*",
                f"{transformer_subfolder}/*",
                "vae/*",
                "vocoder/*",
            ],
        )
    ).resolve()
    snapshot_revision = model.name if model.parent.name == "snapshots" else None
    return model, model_id, snapshot_revision or revision


def _resolve_ltx25_official_artifacts(
    *, checkpoint_variant: str = "distilled"
) -> tuple[dict[str, Path], str, str | None]:
    try:
        variant_files = LTX25_OFFICIAL_VARIANT_FILES[checkpoint_variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported official LTX-2.5 checkpoint variant: {checkpoint_variant!r}") from exc
    files = {**LTX25_OFFICIAL_COMMON_FILES, **variant_files}
    configured_model = os.environ.get("VLLM_TEST_LTX25_OFFICIAL_MODEL")
    configured_revision = os.environ.get("VLLM_TEST_LTX25_OFFICIAL_MODEL_REVISION")
    if configured_model and Path(configured_model).exists():
        root = Path(configured_model).resolve()
        model_source = configured_model
        snapshot_revision = root.name if root.parent.name == "snapshots" else None
        resolved_revision = configured_revision or snapshot_revision
    else:
        model_source = configured_model or LTX25_OFFICIAL_MODEL_ID
        revision = configured_revision
        if revision is None and model_source == LTX25_OFFICIAL_MODEL_ID:
            revision = LTX25_OFFICIAL_MODEL_REVISION
        root = Path(
            snapshot_download(
                repo_id=model_source,
                revision=revision,
                allow_patterns=list(files.values()),
            )
        ).resolve()
        snapshot_revision = root.name if root.parent.name == "snapshots" else None
        resolved_revision = snapshot_revision or revision

    artifacts = {name: root / relative_path for name, relative_path in files.items()}
    missing = [f"{name}: {path}" for name, path in artifacts.items() if not path.is_file()]
    assert not missing, f"Official LTX-2.5 artifacts are missing: {', '.join(missing)}"
    return artifacts, model_source, resolved_revision


def _resolve_gemma_root(model: Path) -> Path:
    configured_root = os.environ.get("VLLM_TEST_LTX_GEMMA_ROOT")
    if configured_root:
        root = Path(configured_root)
        assert root.is_dir(), f"Gemma root not found: {root}"
        return root
    return model


def _resolve_checkpoint(case: LTXAccuracyCase, model: Path) -> Path:
    configured_checkpoint = os.environ.get(case.checkpoint_env)
    if configured_checkpoint:
        checkpoint = Path(configured_checkpoint)
        assert checkpoint.is_file(), f"Official checkpoint not found: {checkpoint}"
        return checkpoint
    model_checkpoint = model / case.checkpoint_filename
    if model_checkpoint.is_file():
        return model_checkpoint
    return Path(
        hf_hub_download(
            repo_id=case.checkpoint_repo,
            filename=case.checkpoint_filename,
            revision=case.checkpoint_revision,
        )
    )


def _resolve_image(case: LTXAccuracyCase) -> Path | None:
    if case.image_filename is None:
        return None
    if case.image_repo is None or case.image_revision is None:
        raise ValueError(f"Incomplete image source for LTX accuracy case {case.name!r}.")
    configured_image = os.environ.get("VLLM_TEST_LTX_I2V_IMAGE")
    if configured_image:
        image = Path(configured_image)
        assert image.is_file(), f"LTX I2V conditioning image not found: {image}"
        return image
    return Path(
        hf_hub_download(
            repo_id=case.image_repo,
            repo_type="dataset",
            filename=case.image_filename,
            revision=case.image_revision,
        )
    )


def _request(case: LTXAccuracyCase, image: Path | None) -> dict[str, object]:
    request: dict[str, object] = {
        "prompt": case.prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "width": 512,
        "height": 384,
        "num_frames": 25,
        "fps": 24,
        "num_inference_steps": 20,
        "seed": 42,
        "video_cfg_scale": 3.0,
        "audio_cfg_scale": 7.0,
        "video_stg_scale": 1.0,
        "audio_stg_scale": 1.0,
        "video_modality_scale": 3.0,
        "audio_modality_scale": 3.0,
        "video_rescale_scale": 0.7,
        "audio_rescale_scale": 0.7,
        "video_stg_blocks": [case.stg_block],
        "audio_stg_blocks": [case.stg_block],
    }
    if image is not None:
        request["image"] = str(image.resolve())
    return request


def _ltx25_request() -> dict[str, object]:
    return {
        "prompt": LTX25_PROMPT,
        # Keep the nightly parity test small enough to run the pinned official
        # pipeline and Omni sequentially on one H100. The public recipe covers
        # the release-resolution 1920x1088x121 configuration.
        "width": 1024,
        "height": 576,
        "num_frames": 25,
        "fps": 24,
        "num_inference_steps": 8,
        "seed": 42,
    }


def _ltx25_full_sigmas(num_inference_steps: int) -> list[float]:
    """Materialize the official LTX-2.5 one-stage schedule in FP32."""
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
    base_anchor = 1024
    max_anchor = 4096
    base_shift = 0.95
    max_shift = 2.05
    slope = (max_shift - base_shift) / (max_anchor - base_anchor)
    sigma_shift = max_anchor * slope + (base_shift - slope * base_anchor)
    exp_shift = math.exp(sigma_shift)
    sigmas = torch.where(sigmas != 0, exp_shift / (exp_shift + (1 / sigmas - 1)), 0)

    non_zero = sigmas != 0
    one_minus_sigmas = 1.0 - sigmas[non_zero]
    scale = one_minus_sigmas[-1] / (1.0 - 0.1)
    sigmas[non_zero] = 1.0 - one_minus_sigmas / scale
    return [float(sigma) for sigma in sigmas]


def _ltx25_full_request() -> dict[str, object]:
    num_inference_steps = 30
    return {
        "prompt": LTX25_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        # Full guidance expands to a four-way batch. Keep the decoded output
        # small enough for the pinned official and Omni runs to share one H100.
        "width": 512,
        "height": 384,
        "num_frames": 25,
        "fps": 24,
        "num_inference_steps": num_inference_steps,
        "seed": 42,
        "video_cfg_scale": 3.0,
        "audio_cfg_scale": 7.0,
        "video_stg_scale": 1.0,
        "audio_stg_scale": 1.0,
        "video_modality_scale": 3.0,
        "audio_modality_scale": 3.0,
        "video_rescale_scale": 0.7,
        "audio_rescale_scale": 0.7,
        "video_stg_blocks": [28],
        "audio_stg_blocks": [28],
        "sigmas": _ltx25_full_sigmas(num_inference_steps),
    }


def test_ltx25_full_request_pins_official_schedule() -> None:
    request = _ltx25_full_request()
    sigmas = request["sigmas"]

    assert isinstance(sigmas, list)
    assert isinstance(request["num_inference_steps"], int)
    assert len(sigmas) == request["num_inference_steps"] + 1
    assert math.isclose(sigmas[0], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert sigmas[-1] == 0.0
    assert all(sigmas[index] > sigmas[index + 1] for index in range(len(sigmas) - 1))


def _video_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    assert reference.shape == prediction.shape
    assert reference.ndim == 4 and reference.shape[-1] == 3
    ssim_scores: list[float] = []
    psnr_scores: list[float] = []
    for reference_frame, prediction_frame in zip(reference, prediction, strict=True):
        reference_tensor = torch.from_numpy(reference_frame).permute(2, 0, 1).unsqueeze(0)
        prediction_tensor = torch.from_numpy(prediction_frame).permute(2, 0, 1).unsqueeze(0)
        ssim_scores.append(float(StructuralSimilarityIndexMeasure(data_range=1.0)(prediction_tensor, reference_tensor)))
        psnr_scores.append(float(PeakSignalNoiseRatio(data_range=1.0)(prediction_tensor, reference_tensor)))
    difference = np.abs(reference.astype(np.float64) - prediction.astype(np.float64))
    return {
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_min": float(np.min(ssim_scores)),
        "psnr_mean_db": float(np.mean(psnr_scores)),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
    }


def _canonical_audio(audio: np.ndarray) -> np.ndarray:
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio[0]
    return audio.astype(np.float64)


def _audio_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | bool]:
    reference = _canonical_audio(reference)
    prediction = _canonical_audio(prediction)
    assert reference.shape == prediction.shape
    difference = reference - prediction
    reference_norm = max(float(np.linalg.norm(reference)), 1e-12)
    prediction_norm = max(float(np.linalg.norm(prediction)), 1e-12)
    return {
        "bitwise_equal": bool(np.array_equal(reference, prediction)),
        "max_abs": float(np.abs(difference).max()),
        "mean_abs": float(np.abs(difference).mean()),
        "relative_l2": float(np.linalg.norm(difference) / reference_norm),
        "cosine_similarity": float(np.vdot(reference.ravel(), prediction.ravel()) / (reference_norm * prediction_norm)),
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.full_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_ltx_one_stage_matches_official(case: LTXAccuracyCase, accuracy_artifact_root: Path) -> None:
    """Compare official and Omni raw AV outputs from the same E2E request."""
    output_root = reset_artifact_dir(accuracy_artifact_root / "ltx_official" / case.name)
    official_root, official_revision = _official_source(accuracy_artifact_root / "ltx_official")
    model = _resolve_model(case)
    gemma_root = _resolve_gemma_root(model)
    checkpoint = _resolve_checkpoint(case, model)
    image = _resolve_image(case)
    request_path = output_root / "request.json"
    request_path.write_text(json.dumps(_request(case, image), indent=2) + "\n")

    runner = Path(__file__).with_name("run_ltx_reference.py")
    runner_args = [
        str(runner),
        "--request",
        str(request_path),
    ]
    if os.environ.get("VLLM_TEST_LTX_ENABLE_LAYERWISE_OFFLOAD", "").lower() in {"1", "true", "yes", "on"}:
        runner_args.append("--enable-layerwise-offload")
    env = os.environ.copy()
    env["VLLM_TEST_LTX_OFFICIAL_REVISION"] = official_revision
    env["PYTHONUNBUFFERED"] = "1"
    repository_root = Path(__file__).resolve().parents[4]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repository_root) if not existing_pythonpath else f"{repository_root}{os.pathsep}{existing_pythonpath}"
    )

    official_output = output_root / "official"
    _run(
        _official_runner_prefix()
        + runner_args
        + [
            "--backend",
            "official",
            "--output-dir",
            str(official_output),
            "--official-root",
            str(official_root),
            "--checkpoint",
            str(checkpoint),
            "--gemma-root",
            str(gemma_root),
        ],
        env=env,
    )

    omni_output = output_root / "omni"
    _run(
        [sys.executable]
        + runner_args
        + [
            "--backend",
            "omni",
            "--output-dir",
            str(omni_output),
            "--model",
            str(model),
            "--model-class-name",
            case.model_class_name,
        ],
        env=env,
    )

    official_metadata = json.loads((official_output / "metadata.json").read_text())
    omni_metadata = json.loads((omni_output / "metadata.json").read_text())
    assert official_metadata["attention_backend"] == ATTENTION_BACKEND
    assert omni_metadata["attention_backend"] == ATTENTION_BACKEND
    assert official_metadata["audio_sample_rate"] == omni_metadata["audio_sample_rate"]
    video_metrics = _video_metrics(
        np.load(official_output / "video.npy"),
        np.load(omni_output / "video.npy"),
    )
    audio_metrics = _audio_metrics(
        np.load(official_output / "audio.npy"),
        np.load(omni_output / "audio.npy"),
    )
    result = {
        "case": case.name,
        "task": "i2v" if image is not None else "t2v",
        "attention_backend": ATTENTION_BACKEND,
        "official_revision": official_revision,
        "model_revision": case.model_revision,
        "checkpoint_revision": case.checkpoint_revision,
        "video": video_metrics,
        "audio": audio_metrics,
    }
    (output_root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    assert video_metrics["ssim_mean"] >= VIDEO_SSIM_MEAN_THRESHOLD
    assert video_metrics["ssim_min"] >= VIDEO_SSIM_MIN_THRESHOLD
    assert video_metrics["psnr_mean_db"] >= VIDEO_PSNR_MEAN_THRESHOLD
    assert audio_metrics["relative_l2"] <= AUDIO_RELATIVE_L2_THRESHOLD
    assert audio_metrics["cosine_similarity"] >= AUDIO_COSINE_THRESHOLD


@pytest.mark.full_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_ltx25_distilled_two_stage_matches_official(accuracy_artifact_root: Path) -> None:
    """Compare Omni with the official runtime using the same checkpoint connector weights."""
    artifact_parent = accuracy_artifact_root / "ltx_official"
    output_root = reset_artifact_dir(artifact_parent / "ltx2_5_distilled")
    official_root, official_revision = _ltx25_official_source(artifact_parent)
    official_artifacts, official_model_source, official_model_revision = _resolve_ltx25_official_artifacts()
    omni_model, omni_model_source, omni_model_revision = _resolve_ltx25_omni_model()
    request_path = output_root / "request.json"
    request_path.write_text(json.dumps(_ltx25_request(), indent=2) + "\n")

    runner = Path(__file__).with_name("run_ltx_reference.py")
    runner_args = [
        str(runner),
        "--request",
        str(request_path),
        "--enable-layerwise-offload",
    ]
    env = os.environ.copy()
    env["VLLM_TEST_LTX_OFFICIAL_REVISION"] = official_revision
    env["PYTHONUNBUFFERED"] = "1"
    repository_root = Path(__file__).resolve().parents[4]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repository_root) if not existing_pythonpath else f"{repository_root}{os.pathsep}{existing_pythonpath}"
    )

    official_output = output_root / "official"
    _run(
        _official_runner_prefix()
        + runner_args
        + [
            "--backend",
            "official",
            "--output-dir",
            str(official_output),
            "--official-root",
            str(official_root),
            "--transformer-path",
            str(official_artifacts["transformer"]),
            "--text-encoder-path",
            str(official_artifacts["text_encoder"]),
            "--video-vae-path",
            str(official_artifacts["video_vae"]),
            "--audio-vae-path",
            str(official_artifacts["audio_vae"]),
            "--spatial-upsampler-path",
            str(official_artifacts["spatial_upsampler"]),
            "--connector-model",
            str(omni_model),
        ],
        env=env,
    )

    omni_output = output_root / "omni"
    _run(
        [sys.executable]
        + runner_args
        + [
            "--backend",
            "omni",
            "--output-dir",
            str(omni_output),
            "--model",
            str(omni_model),
            "--model-class-name",
            "LTX2TwoStagePipeline",
        ],
        env=env,
    )

    official_metadata = json.loads((official_output / "metadata.json").read_text())
    omni_metadata = json.loads((omni_output / "metadata.json").read_text())
    assert official_metadata["official_revision"] == official_revision
    assert official_metadata["attention_backend"] == ATTENTION_BACKEND
    assert official_metadata["connector_model"] == str(omni_model)
    assert omni_metadata["attention_backend"] == ATTENTION_BACKEND
    assert official_metadata["audio_sample_rate"] == omni_metadata["audio_sample_rate"]
    video_metrics = _video_metrics(
        np.load(official_output / "video.npy"),
        np.load(omni_output / "video.npy"),
    )
    audio_metrics = _audio_metrics(
        np.load(official_output / "audio.npy"),
        np.load(omni_output / "audio.npy"),
    )
    result = {
        "case": "ltx2_5_distilled",
        "task": "t2v",
        "attention_backend": ATTENTION_BACKEND,
        "official_revision": official_revision,
        "official_model": official_model_source,
        "official_model_revision": official_model_revision,
        "official_connector_model": omni_model_source,
        "official_connector_revision": omni_model_revision,
        "omni_model": omni_model_source,
        "omni_model_revision": omni_model_revision,
        "resolved_omni_model_path": str(omni_model),
        "video": video_metrics,
        "audio": audio_metrics,
    }
    (output_root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    assert video_metrics["ssim_mean"] >= LTX25_VIDEO_SSIM_MEAN_THRESHOLD
    assert video_metrics["ssim_min"] >= LTX25_VIDEO_SSIM_MIN_THRESHOLD
    assert video_metrics["psnr_mean_db"] >= LTX25_VIDEO_PSNR_MEAN_THRESHOLD
    assert audio_metrics["cosine_similarity"] >= LTX25_AUDIO_COSINE_THRESHOLD
    assert audio_metrics["relative_l2"] <= LTX25_AUDIO_RELATIVE_L2_THRESHOLD


@pytest.mark.full_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_ltx25_full_one_stage_matches_official(accuracy_artifact_root: Path) -> None:
    """Compare raw official Full/SFT weights with Omni's converted Full pipeline."""
    artifact_parent = accuracy_artifact_root / "ltx_official"
    output_root = reset_artifact_dir(artifact_parent / "ltx2_5_full")
    official_root, official_revision = _ltx25_official_source(artifact_parent)
    official_artifacts, official_model_source, official_model_revision = _resolve_ltx25_official_artifacts(
        checkpoint_variant="full"
    )
    omni_model, omni_model_source, omni_model_revision = _resolve_ltx25_omni_model(
        transformer_subfolder="transformer_full"
    )
    request_path = output_root / "request.json"
    request_path.write_text(json.dumps(_ltx25_full_request(), indent=2) + "\n")

    runner = Path(__file__).with_name("run_ltx_reference.py")
    runner_args = [
        str(runner),
        "--request",
        str(request_path),
        "--enable-layerwise-offload",
    ]
    env = os.environ.copy()
    env["VLLM_TEST_LTX_OFFICIAL_REVISION"] = official_revision
    env["PYTHONUNBUFFERED"] = "1"
    repository_root = Path(__file__).resolve().parents[4]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repository_root) if not existing_pythonpath else f"{repository_root}{os.pathsep}{existing_pythonpath}"
    )

    official_output = output_root / "official"
    _run(
        _official_runner_prefix()
        + runner_args
        + [
            "--backend",
            "official",
            "--output-dir",
            str(official_output),
            "--official-root",
            str(official_root),
            "--official-pipeline",
            "full_one_stage",
            "--transformer-path",
            str(official_artifacts["transformer"]),
            "--text-encoder-path",
            str(official_artifacts["text_encoder"]),
            "--video-vae-path",
            str(official_artifacts["video_vae"]),
            "--audio-vae-path",
            str(official_artifacts["audio_vae"]),
            "--connector-model",
            str(omni_model),
        ],
        env=env,
    )

    omni_output = output_root / "omni"
    _run(
        [sys.executable]
        + runner_args
        + [
            "--backend",
            "omni",
            "--output-dir",
            str(omni_output),
            "--model",
            str(omni_model),
            "--model-class-name",
            "LTX2Pipeline",
        ],
        env=env,
    )

    official_metadata = json.loads((official_output / "metadata.json").read_text())
    omni_metadata = json.loads((omni_output / "metadata.json").read_text())
    assert official_metadata["official_revision"] == official_revision
    assert official_metadata["pipeline"] == "TI2VidOneStagePipeline"
    assert official_metadata["attention_backend"] == ATTENTION_BACKEND
    assert official_metadata["connector_model"] == str(omni_model)
    assert omni_metadata["model_class_name"] == "LTX2Pipeline"
    assert omni_metadata["attention_backend"] == ATTENTION_BACKEND
    assert official_metadata["seed"] == omni_metadata["seed"]
    assert official_metadata["sigmas"] == omni_metadata["sigmas"]
    assert official_metadata["audio_sample_rate"] == omni_metadata["audio_sample_rate"]
    video_metrics = _video_metrics(
        np.load(official_output / "video.npy"),
        np.load(omni_output / "video.npy"),
    )
    audio_metrics = _audio_metrics(
        np.load(official_output / "audio.npy"),
        np.load(omni_output / "audio.npy"),
    )
    result = {
        "case": "ltx2_5_full",
        "task": "t2v",
        "attention_backend": ATTENTION_BACKEND,
        "official_revision": official_revision,
        "official_model": official_model_source,
        "official_model_revision": official_model_revision,
        "official_connector_model": omni_model_source,
        "official_connector_revision": omni_model_revision,
        "omni_model": omni_model_source,
        "omni_model_revision": omni_model_revision,
        "resolved_omni_model_path": str(omni_model),
        "video": video_metrics,
        "audio": audio_metrics,
    }
    (output_root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    assert video_metrics["ssim_mean"] >= LTX25_FULL_VIDEO_SSIM_MEAN_THRESHOLD
    assert video_metrics["ssim_min"] >= LTX25_FULL_VIDEO_SSIM_MIN_THRESHOLD
    assert video_metrics["psnr_mean_db"] >= LTX25_FULL_VIDEO_PSNR_MEAN_THRESHOLD
    assert audio_metrics["relative_l2"] <= LTX25_FULL_AUDIO_RELATIVE_L2_THRESHOLD
    assert audio_metrics["cosine_similarity"] >= LTX25_FULL_AUDIO_COSINE_THRESHOLD
