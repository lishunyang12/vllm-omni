#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize Wan2.2 to a ModelOpt FP8 Hugging Face checkpoint.

Calibrates the DiT transformer(s) using a small video prompt set and exports a
diffusers-style directory whose transformer(s) carry ModelOpt FP8 metadata.
The exported checkpoint is consumable by vllm-omni's ModelOpt FP8 adapter
(see vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt_fp8.py).

Layers kept full precision match the #2728 / #2795 pattern: condition embedder
(time/text/image), patch embedding, modulation (scale_shift_table), final
norm + proj_out, and sequence-parallel helpers. All attention + FFN linears
are quantized — static calibration handles the numerics that online FP8
couldn't (see #2920 ablation).

Supported targets:
- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (single-transformer, 80GB BF16 fits one GPU)
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (MoE, two transformers, needs 2+ GPUs BF16)
- `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (MoE, two transformers, needs 2+ GPUs BF16)

For VACE variants (Wan-AI/Wan2.X-VACE-*), use the dedicated script
`quantize_wan2_2_vace_modelopt_fp8.py` instead

For MoE A14B variants the diffusers pipeline routes between `transformer` (high
noise, t >= boundary_timestep) and `transformer_2` (low noise) automatically
based on `boundary_ratio` from `model_index.json`. A single calibration run
collects amax statistics for both via timestep-conditioned forward passes.

For I2V variants diffusers' WanImageToVideoPipeline takes a required `image`
kwarg, so calibration must pair every prompt with a reference image — pass
`--is-i2v` together with `--reference-images <dir-or-file>`.

Example(TI2V-5B):
    python examples/quantization/quantize_wan2_2_modelopt_fp8.py \
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
        --output ./wan22-ti2v-modelopt-fp8 \
        --overwrite
Example(T2V-A14B):
    python examples/quantization/quantize_wan2_2_modelopt_fp8.py \
            --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
            --output ./wan22-t2v-modelopt-fp8 \
            --calib-boundary-ratio 0.5 \
            --overwrite
Example(I2V-A14B):
    python examples/quantization/quantize_wan2_2_modelopt_fp8.py \
            --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
            --output ./wan22-i2v-modelopt-fp8 \
            --is-i2v --reference-images /path/to/ref_images/ \
            --calib-boundary-ratio 0.5 \
            --overwrite
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline

DEFAULT_PROMPTS = [
    "A dog running across a field of golden wheat.",
    "An astronaut riding a horse across the surface of Mars, red dust swirling, cinematic wide shot.",
    "A hummingbird hovering in front of a vibrant red flower, slow motion, macro shot.",
    "A crackling campfire at night under a starry sky, sparks rising into the dark.",
    "An underwater shot of a coral reef with tropical fish swimming by, sun rays piercing the water.",
    "A close-up of a blooming rose covered in morning dew, soft natural light.",
    "A peaceful mountain village at dawn, mist rolling over the rooftops, cinematic establishing shot.",
    "A skateboarder doing a kickflip in an urban plaza, slow motion, golden hour lighting.",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Input Wan2.2 diffusers directory or HF id.")
    p.add_argument("--output", required=True, help="Output directory for the ModelOpt FP8 checkpoint.")
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--height", type=int, default=704, help="Calibration video height (Wan2.2 TI2V-5B native: 704).")
    p.add_argument("--width", type=int, default=1280, help="Calibration video width (Wan2.2 TI2V-5B native: 1280).")
    p.add_argument(
        "--num-frames",
        type=int,
        default=49,
        help="Frames per calibration sample. 49 matches the typical short benchmark; "
        "use 17 to reduce memory pressure during calibration.",
    )
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument(
        "--calib-steps",
        type=int,
        default=10,
        help="Denoising steps per calibration prompt (10 is enough for amax statistics).",
    )
    p.add_argument(
        "--calib-size",
        type=int,
        default=8,
        help="How many prompts to use for calibration. It is now decoupled with "
        "number of DEFAULT_PROMPTS, i.e. type any size you like",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Custom calibration prompt. Repeat to provide multiple.",
    )
    p.add_argument(
        "--quantize-mha",
        action="store_true",
        help="Enable FP8 attention K/V/softmax quantizers. Off by default — Wan2.2's long attention "
        "sequences amplified FP8 drift in the online ablation (see #2920).",
    )
    p.add_argument(
        "--weight-block-size",
        type=str,
        default=None,
        help="Per-block weight quantization as 'M,N'. Only '128,128' is accepted because upstream "
        "vLLM's ModelOptFp8PbWoLinearMethod hardcodes that block shape. Default: per-tensor. "
        "Block-wise saves checkpoints with FP8_PB_WO routing (per-block static weights + per-token-"
        "group dynamic activations); per-tensor uses static FP8 with calibrated activation scales.",
    )
    p.add_argument(
        "--calib-boundary-ratio",
        type=float,
        default=None,
        help="Pass-1-only boundary_ratio override for Wan2.2 MoE calibration. Only takes "
        "effect when the loaded pipeline has transformer_2. Lowering it (e.g. 0.5) shifts "
        "more denoising steps onto `transformer` so its quantizers see a richer amax "
        "sample WITHOUT bumping --calib-steps. Pass 2 always restores the model's "
        "production boundary_ratio (A14B = 0.875) to keep transformer_2's amax in "
        "production distribution. If unset, both passes use the production value (default).",
    )
    p.add_argument(
        "--is-i2v",
        action="store_true",
        help="Set when quantizing a Wan2.2 I2V model (e.g. Wan2.2-I2V-A14B-Diffusers). "
        "diffusers' WanImageToVideoPipeline takes a required `image` kwarg, so calibration "
        "must pair every prompt with a reference image — pass --reference-images.",
    )
    p.add_argument(
        "--reference-images",
        type=str,
        default=None,
        help="Requires --is-i2v. Directory of jpg/jpeg/png/webp files (or a single image). "
        "Every calibration sample is paired with a cycled ref image since image_embedder "
        "is required, not optional, in I2V pipelines. Warning: one image per sample",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return p


def _parse_block_size(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    parts = [int(x) for x in spec.split(",") if x.strip()]
    if len(parts) != 2:
        raise SystemExit(f"--weight-block-size must be 'M,N' (2 ints), got {spec!r}")
    return parts


def _require_modelopt() -> Any:
    try:
        import modelopt.torch.quantization as mtq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "NVIDIA ModelOpt is not installed. Install with:\n"
            "  pip install 'nvidia-modelopt[all]'\n"
            f"Original error: {exc}"
        ) from exc
    return mtq


def _ensure_paths(args: argparse.Namespace) -> tuple[str, Path]:
    model_path = args.model
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {output_dir}\nPass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    return model_path, output_dir


def _select_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _load_reference_images(spec: str | None) -> list[Any]:
    """Load PIL.Image list from a directory or a single file path."""
    if spec is None:
        return []
    from PIL import Image

    p = Path(spec).expanduser()
    if not p.exists():
        raise SystemExit(f"--reference-images path not found: {p}")
    if p.is_file():
        return [Image.open(p).convert("RGB")]
    image_paths = sorted(
        f for f in p.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if not image_paths:
        raise SystemExit(f"No image files (jpg/jpeg/png/webp) found in {p}")
    return [Image.open(f).convert("RGB") for f in image_paths]


def _build_calib_samples(
    args: argparse.Namespace,
    is_i2v: bool,
    ref_images: list[Any],
) -> list[tuple[str, Any]]:
    """Build calibration (prompt, reference_image_or_None) pairs.

    - Non-I2V (T2V/TI2V/A14B-T2V): every sample is (prompt, None).
    - I2V: every sample paired with a cycled ref image (image kwarg is required
      by diffusers' WanImageToVideoPipeline). Prompt pool is DEFAULT_PROMPTS
      since the image dominates the visual signal — text mainly drives motion.
    """
    if args.calib_size <= 0:
        raise SystemExit("--calib-size must be positive.")

    prompts = args.prompt or DEFAULT_PROMPTS
    if is_i2v:
        # ref_images is guaranteed non-empty by main()'s validation (--is-i2v
        # requires --reference-images).
        return [(prompt, ref_images[i % len(ref_images)]) for i, prompt in enumerate(prompts)]
    return [(prompt, None) for prompt in prompts]


# Layers to KEEP at full precision. Wan2.2's module naming:
# - condition_embedder: time_embedder, time_proj, text_embedder, image_embedder (I2V)
# - patch_embedding: Conv3dLayer (already not Linear, belt-and-suspenders skip)
# - scale_shift_table: nn.Parameter modulation (not Linear, but pattern guard)
# - norm_out: AdaLayerNorm final
# - proj_out: final nn.Linear
# - timestep_proj_prepare / output_scale_shift_prepare: SP helpers
def _filter_func_wan22(name: str) -> bool:
    pattern = re.compile(
        r"(proj_out.*|"
        r".*(condition_embedder|patch_embedding|"
        r"norm_out|scale_shift_table|"
        r"timestep_proj_prepare|output_scale_shift_prepare).*)"
    )
    return pattern.match(name) is not None


def _mha_filter_func(name: str) -> bool:
    pattern = re.compile(
        r".*(q_bmm_quantizer|k_bmm_quantizer|v_bmm_quantizer|softmax_quantizer|bmm2_output_quantizer).*"
    )
    return pattern.match(name) is not None


def _disable_known_problematic_quantizers(mtq: Any, backbone: torch.nn.Module, *, quantize_mha: bool) -> None:
    if not hasattr(mtq, "disable_quantizer"):
        return
    mtq.disable_quantizer(backbone, _filter_func_wan22)
    if not quantize_mha:
        mtq.disable_quantizer(backbone, _mha_filter_func)


def _move_tensor(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (tuple, list)):
        moved = [_move_tensor(v, device) for v in value]
        return type(value)(moved)
    return value


def _make_input_device_hook(target_device: torch.device):
    """Pre-hook that moves all tensor args/kwargs onto the module's device."""

    def pre_hook(_module, args, kwargs):
        new_args = tuple(_move_tensor(a, target_device) for a in args)
        new_kwargs = {k: _move_tensor(v, target_device) for k, v in kwargs.items()}
        return new_args, new_kwargs

    return pre_hook


def _make_output_device_hook(primary_device: torch.device):
    """Post-hook that moves outputs back to the pipeline's primary device."""

    def post_hook(_module, _args, output):
        return _move_tensor(output, primary_device)

    return post_hook


def _load_pipeline(model_path: str, dtype: torch.dtype) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    transformer_2 = getattr(pipe, "transformer_2", None)
    if transformer_2 is not None and torch.cuda.device_count() >= 2:
        # diffusers' WanPipeline routes between the two by boundary_timestep but does
        # NOT transfer activations across devices, so this case bridge transformer_2 with
        # forward hooks: pre-hook moves inputs cuda:0 -> cuda:1, post-hook moves
        # outputs back cuda:1 -> cuda:0. The pipeline then sees a uniform cuda:0
        # state and scheduler.step works without modification.
        primary = torch.device("cuda:0")
        secondary = torch.device("cuda:1")
        pipe.transformer.to(primary)
        transformer_2.to(secondary)
        for component_name in ("text_encoder", "vae", "image_encoder"):
            component = getattr(pipe, component_name, None)
            if component is not None:
                component.to(primary)
        transformer_2.register_forward_pre_hook(_make_input_device_hook(secondary), with_kwargs=True)
        transformer_2.register_forward_hook(_make_output_device_hook(primary))
        print(f"  device map:      transformer={primary}, transformer_2={secondary} (cross-device hooks installed)")
    else:
        pipe.to("cuda")
    return pipe


def _build_forward_loop(
    pipe: DiffusionPipeline,
    args: argparse.Namespace,
    samples: list[tuple[str, Any]],
):
    """Build a forward_loop that drives `pipe` over the calibration samples.

    Samples carrying a reference image are forwarded with `image=PIL.Image`
    (the kwarg expected by diffusers' WanImageToVideoPipeline). Samples with
    ref=None call pipe(prompt=...) — the standard T2V path.
    """
    generator = torch.Generator(device="cuda")

    # Try setting guidance on the pipeline's guider if present (newer diffusers APIs).
    guider = getattr(pipe, "guider", None)
    if guider is not None and hasattr(guider, "guidance_scale"):
        try:
            guider.guidance_scale = args.guidance_scale
        except Exception:
            pass

    base_kwargs = dict(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.calib_steps,
        output_type="latent",
    )

    def forward_loop(*_unused_args, **_unused_kwargs) -> None:
        with torch.inference_mode():
            for idx, (prompt, ref_image) in enumerate(samples):
                generator.manual_seed(args.seed + idx)
                kwargs = dict(base_kwargs)
                if ref_image is not None:
                    kwargs["image"] = ref_image
                # Try with guidance_scale first; fall back without on TypeError
                # for pipelines that take CFG via guider config only.
                try:
                    pipe(prompt=prompt, generator=generator, guidance_scale=args.guidance_scale, **kwargs)
                except TypeError as exc:
                    if "guidance_scale" not in str(exc):
                        raise
                    pipe(prompt=prompt, generator=generator, **kwargs)

    return forward_loop


def _summarize_export(output_dir: Path, subfolder: str = "transformer") -> None:
    cfg_path = output_dir / subfolder / "config.json"
    if not cfg_path.exists():
        print(f"[warn] {cfg_path} missing.", file=sys.stderr)
        return
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        print(f"[warn] No quantization_config in {subfolder}/config.json.", file=sys.stderr)
        return
    print(f"Export summary ({subfolder}):")
    print(f"  quant_method: {qc.get('quant_method')}")
    print(f"  quant_algo:   {qc.get('quant_algo')}")
    producer = qc.get("producer")
    if isinstance(producer, dict):
        print(f"  producer:     {producer.get('name')} {producer.get('version')}")
    print(f"  config path:  {cfg_path}")


def _force_export_quantized_weights(backbone: torch.nn.Module, dtype: torch.dtype) -> int:
    """Convert in-memory weights of quantized modules to actual FP8 storage.

    `export_hf_checkpoint` skips this step for unknown model types (Wan2.2 isn't
    in ModelOpt's recognized-model registry), so we must call the per-weight
    export helper ourselves. Same workaround as the HunyuanVideo-1.5 / HunyuanImage-3
    calibration helpers.
    """
    from modelopt.torch.export.quant_utils import (
        QUANTIZATION_NONE,
        get_quantization_format,
        quantizer_attr_names,
        weight_attr_names,
    )
    from modelopt.torch.export.unified_export_hf import _export_quantized_weight

    exported = 0
    for name, module in backbone.named_modules():
        try:
            quantization_format = get_quantization_format(module)
        except Exception as exc:
            print(f"[warn] Could not inspect quantization format for {name}: {exc}", file=sys.stderr)
            continue
        if quantization_format == QUANTIZATION_NONE:
            continue
        for weight_name in weight_attr_names(module):
            quantizer_attrs = quantizer_attr_names(weight_name)
            weight_quantizer = getattr(module, quantizer_attrs.weight_quantizer, None)
            if weight_quantizer is None or not getattr(weight_quantizer, "is_enabled", False):
                continue
            _export_quantized_weight(module, dtype, weight_name)
            exported += 1
    return exported


def _wan22_quant_config_block(weight_block_size: list[int] | None = None) -> dict:
    """Mirror ModelOpt FP8 metadata expected by vllm-omni's adapter (#2913).

    For per-block weight quantization,upstream's FP8_PB_WO hardcodes _WEIGHT_BLOCK_SIZE = (128, 128), so any other
    block shape produces a checkpoint vLLM cannot serve.
    """
    if weight_block_size is not None and tuple(weight_block_size) != (128, 128):
        raise ValueError(
            f"--weight-block-size {tuple(weight_block_size)} not supported: upstream vLLM's "
            "ModelOptFp8PbWoLinearMethod hardcodes (128, 128). Pass '128,128' or omit the flag."
        )

    weights_cfg: dict = {"dynamic": False, "num_bits": 8, "type": "float"}
    if weight_block_size is not None:
        weights_cfg["strategy"] = "block"
        weights_cfg["block_structure"] = f"{weight_block_size[0]}x{weight_block_size[1]}"
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                "weights": weights_cfg,
                "targets": ["Linear"],
            }
        },
        "ignore": [
            "condition_embedder*",
            "norm_out*",
            "output_scale_shift_prepare*",
            "patch_embedding*",
            "proj_out*",
            "scale_shift_table*",
            "timestep_proj_prepare*",
        ],
        "producer": {"name": "modelopt"},
        "quant_algo": "FP8_PB_WO" if weight_block_size is not None else "FP8",
        "quant_method": "modelopt",
    }


def _patch_quant_config(
    output_dir: Path,
    subfolder: str = "transformer",
    weight_block_size: list[int] | None = None,
) -> None:
    """Inject quant_algo: FP8 + config_groups into <subfolder>/config.json so
    vllm-omni's adapter (#2913) recognises the checkpoint as ModelOpt FP8.

    For Wan2.2 MoE (T2V/I2V-A14B), call once per transformer subfolder
    (`transformer` and `transformer_2`).
    """
    cfg_path = output_dir / subfolder / "config.json"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    new_qc = _wan22_quant_config_block(weight_block_size=weight_block_size)
    existing = cfg.get("quantization_config")
    if isinstance(existing, dict):
        producer = existing.get("producer")
        if isinstance(producer, dict):
            new_qc["producer"] = producer

    cfg["quantization_config"] = new_qc
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _save_pipeline_with_fp8_transformers(
    pipe: DiffusionPipeline,
    model_path: str,
    output_dir: Path,
    max_shard_size: str = "5GB",
) -> None:
    """Copy source dir verbatim minus transformer/(_2), then save quantized transformer(s).

    For Wan2.2 MoE (T2V/I2V-A14B), `pipe.transformer_2` is also saved into the
    `transformer_2/` subfolder. Single-transformer variants (TI2V-5B) skip it.
    """
    from modelopt.torch.export.diffusers_utils import hide_quantizers_from_state_dict

    src = Path(model_path)
    if not src.exists():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(model_path))

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(src, output_dir, ignore=shutil.ignore_patterns("transformer", "transformer_2"))

    backbones: list[tuple[str, torch.nn.Module]] = [("transformer", pipe.transformer)]
    transformer_2 = getattr(pipe, "transformer_2", None)
    if transformer_2 is not None:
        backbones.append(("transformer_2", transformer_2))

    for subfolder, backbone in backbones:
        out = output_dir / subfolder
        with hide_quantizers_from_state_dict(backbone):
            backbone.save_pretrained(
                str(out),
                safe_serialization=True,
                max_shard_size=max_shard_size,
            )


def _calibrate(
    backbone: torch.nn.Module,
    label: str,
    *,
    mtq: Any,
    quant_config: dict,
    forward_loop,
    quantize_mha: bool,
) -> torch.nn.Module:
    """Wrap one transformer backbone with quantizers and run calibration.

    Returns the (possibly replaced) backbone module so the caller can rebind
    `pipe.transformer` / `pipe.transformer_2` to the wrapped instance. The
    backbone's weights remain in their original dtype here — call
    `_force_export` afterwards to commit FP8 storage.
    """
    print(f"\nCalibrating {label}...")
    quantized = mtq.quantize(backbone, quant_config, forward_loop)
    if quantized is not None:
        backbone = quantized
    _disable_known_problematic_quantizers(mtq, backbone, quantize_mha=quantize_mha)
    return backbone


def _force_export(backbone: torch.nn.Module, label: str, dtype: torch.dtype) -> None:
    """Convert calibrated weights to actual FP8 storage."""
    print(f"\nForcing FP8 weight serialization for {label} (Wan2.2 isn't in ModelOpt's")
    print("recognized-model registry, so we call the per-weight export helper ourselves)...")
    exported = _force_export_quantized_weights(backbone, dtype)
    print(f"  -> {exported} weights converted to FP8 in {label}")
    if exported == 0:
        raise SystemExit(
            f"No quantized weights were exported in {label}. Calibration may have skipped every "
            "layer (check the disable_quantizer regex) or `mtq.quantize` did not actually wrap "
            "any weight quantizers."
        )


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ModelOpt FP8 quantization.")

    mtq = _require_modelopt()
    model_path, output_dir = _ensure_paths(args)
    dtype = _select_dtype(args.dtype)
    weight_block_size = _parse_block_size(args.weight_block_size)

    if args.reference_images is not None and not args.is_i2v:
        raise SystemExit("--reference-images requires --is-i2v.")
    if args.is_i2v and args.reference_images is None:
        raise SystemExit(
            "--is-i2v requires --reference-images: diffusers' WanImageToVideoPipeline "
            "takes a required `image` kwarg, so calibration must pair every prompt with "
            "a reference image."
        )
    ref_images = _load_reference_images(args.reference_images) if args.is_i2v else []
    samples = _build_calib_samples(args, args.is_i2v, ref_images)
    sample_label = f"I2V={len(samples)}" if args.is_i2v else f"T2V={len(samples)}"

    print("Quantization plan:")
    print(f"  input:           {args.model}")
    print(f"  output:          {output_dir}")
    print(f"  dtype:           {dtype}")
    print(f"  height/width:    {args.height}x{args.width}")
    print(f"  num_frames:      {args.num_frames}")
    print(f"  calib_size:      {len(samples)} ({sample_label})")
    print(f"  calib_steps:     {args.calib_steps}")
    print(f"  quantize_mha:    {args.quantize_mha}")
    print(f"  is_i2v:          {args.is_i2v}")
    if args.is_i2v:
        print(f"  reference imgs:  {len(ref_images)}")
    print(
        f"  weight strategy: {'block-wise ' + str(weight_block_size) if weight_block_size else 'per-tensor (default)'}"
    )

    pipe = _load_pipeline(model_path, dtype)
    is_dual = getattr(pipe, "transformer_2", None) is not None
    if is_dual:
        print("  detected MoE A14B variant (transformer + transformer_2)")

    # Capture the model's production boundary_ratio (from model_index.json) so
    # we can restore it before pass 2. --calib-boundary-ratio only overrides
    # pass 1 to give `transformer` more amax samples; pass 2 must run at the
    # production boundary so `transformer_2` calibrates on the same noise
    # distribution it will see at inference time.
    production_boundary = pipe.config.get("boundary_ratio") if is_dual else None

    quant_config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    if weight_block_size is not None:
        quant_config["quant_cfg"]["*weight_quantizer"] = {
            "num_bits": (4, 3),
            "block_sizes": {-1: weight_block_size[1], -2: weight_block_size[0]},
        }
        print(
            f"  -> overriding weight quantizer with block_sizes={weight_block_size} "
            f"({weight_block_size[0]}x{weight_block_size[1]} tiles)"
        )

    forward_loop = _build_forward_loop(pipe, args, samples)

    # Single-transformer (TI2V-5B) does one pass; MoE A14B variants do two.
    # The diffusers Wan22 pipeline routes between transformer (high noise) and
    # transformer_2 (low noise) by boundary_timestep, so each forward_loop run
    # exercises the backbone currently being calibrated. mtq.quantize wraps
    # quantizers and then drives the forward_loop to collect amax statistics.
    #
    # Calibration must complete for BOTH backbones BEFORE any force_export call:
    # Before _force_export, transformer's weights must still be BF16 at that point.
    if is_dual and args.calib_boundary_ratio is not None:
        pipe.register_to_config(boundary_ratio=args.calib_boundary_ratio)
        print(
            f"\n  pass 1 boundary_ratio: {args.calib_boundary_ratio} "
            f"(override of production {production_boundary} for transformer sample boost)"
        )

    pipe.transformer = _calibrate(
        pipe.transformer,
        "transformer",
        mtq=mtq,
        quant_config=quant_config,
        forward_loop=forward_loop,
        quantize_mha=args.quantize_mha,
    )
    if is_dual:
        if args.calib_boundary_ratio is not None:
            pipe.register_to_config(boundary_ratio=production_boundary)
            print(
                f"\n  pass 2 boundary_ratio: {production_boundary} "
                "(restored to production for transformer_2 in-distribution calibration)"
            )
        pipe.transformer_2 = _calibrate(
            pipe.transformer_2,
            "transformer_2",
            mtq=mtq,
            quant_config=quant_config,
            forward_loop=forward_loop,
            quantize_mha=args.quantize_mha,
        )

    _force_export(pipe.transformer, "transformer", dtype)
    if is_dual:
        _force_export(pipe.transformer_2, "transformer_2", dtype)

    print("\nSaving pipeline with FP8 transformer(s)...")
    _save_pipeline_with_fp8_transformers(pipe, model_path, output_dir)
    _patch_quant_config(output_dir, subfolder="transformer", weight_block_size=weight_block_size)
    if is_dual:
        _patch_quant_config(output_dir, subfolder="transformer_2", weight_block_size=weight_block_size)
    print(f"Saved to: {output_dir}")
    _summarize_export(output_dir, subfolder="transformer")
    if is_dual:
        _summarize_export(output_dir, subfolder="transformer_2")

    print("\nNext: validate the checkpoint with vllm-omni:")
    if args.is_i2v:
        print(
            "  python examples/offline_inference/image_to_video/image_to_video.py \\\n"
            f"    --model {output_dir} \\\n"
            "    --quantization fp8 \\\n"
            "    --prompt 'A subject from the reference image moves through the scene.' \\\n"
            "    --image <path/to/your/reference.jpg> \\\n"
            f"    --height {args.height} --width {args.width} --num-frames {args.num_frames} \\\n"
            "    --num-inference-steps 30 --guidance-scale 5.0 --seed 42 \\\n"
            "    --output outputs/wan22_i2v_modelopt_fp8.mp4"
        )
    else:
        print(
            "  python examples/offline_inference/text_to_video/text_to_video.py \\\n"
            f"    --model {output_dir} \\\n"
            "    --quantization fp8 \\\n"
            "    --prompt 'A dog running across a field of golden wheat.' \\\n"
            f"    --height {args.height} --width {args.width} --num-frames {args.num_frames} \\\n"
            "    --num-inference-steps 30 --guidance-scale 5.0 --seed 42 \\\n"
            "    --output outputs/wan22_modelopt_fp8.mp4"
        )
    print(
        "\n  (--quantization fp8 is auto-upgraded to ModelOpt FP8 at runtime because the "
        "checkpoint's config.json has modelopt metadata.)"
    )


if __name__ == "__main__":
    main()
