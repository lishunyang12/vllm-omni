#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize Wan2.2 to a ModelOpt NVFP4 W4A8 Hugging Face checkpoint.

Uses NVFP4 weights (4-bit FP4 with 16-element group-wise scales) + FP8
activations (E4M3).  Pure W4A4 (FP4 activations) was rejected after the
activation amax variance check showed ~20x ratio across denoising steps in
cross-attention layers (see check_activation_variance.py); a static FP4
activation scale cannot cover that range without severe clipping or
precision loss.

Supports both Wan2.2 variants:
  - Wan-AI/Wan2.2-T2V-A14B-Diffusers   (dual-DiT: transformer + transformer_2)
  - Wan-AI/Wan2.2-TI2V-5B-Diffusers    (single DiT + image conditioning)

IMPORTANT - SERVING REQUIRES BLACKWELL:
    NVFP4 GEMM kernels (CutlassNvFp4ScaledMMLinearKernel) require sm_100 /
    Blackwell hardware (B200, RTX Pro 6000, RTX 5090).  Calibration runs
    fine on H100 - this script just writes packed FP4 weights + FP8 scales
    to disk - but you cannot validate or serve the resulting checkpoint on
    H100.  Publish to HF Hub and test on Blackwell when available.

Example:
    python examples/quantization/quantize_wan2_2_modelopt_nvfp4.py \\
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \\
        --output ./wan22-t2v-a14b-modelopt-nvfp4 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline

# 8 diverse calibration prompts spanning motion, scene, and composition axes.
DEFAULT_PROMPTS = [
    "A person running through a crowded city street at dusk, motion blur in the background.",
    "A candle flame burning steadily on a wooden table, soft warm light.",
    "Ocean waves crashing against rocky cliffs, sea spray flying upward.",
    "A drone flying over a dense forest canopy at sunrise, slow horizontal pan.",
    "A close-up of a hand writing equations on a chalkboard, white chalk marks appearing.",
    "A woman smiling and turning her head slowly toward the camera, shallow depth of field.",
    "Colorful paint dropped into clear water, spreading and swirling in slow motion.",
    "A busy outdoor marketplace with vendors and shoppers moving between stalls.",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Input Wan2.2 diffusers directory or HF id.")
    p.add_argument("--output", required=True, help="Output directory for the ModelOpt NVFP4 checkpoint.")
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--calib-steps", type=int, default=40)
    p.add_argument("--calib-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument(
        "--quantize-mha",
        action="store_true",
        help="Enable attention K/V/softmax BMM quantizers. Off by default (extra recipes, marginal win).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def _require_modelopt() -> Any:
    try:
        import modelopt.torch.quantization as mtq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "NVIDIA ModelOpt is not installed. Install with:\n"
            "  pip install 'nvidia-modelopt[all]'\n"
            f"Original error: {exc}"
        ) from exc
    if not hasattr(mtq, "NVFP4_DEFAULT_CFG"):
        raise SystemExit(
            "Your modelopt version doesn't expose NVFP4_DEFAULT_CFG. "
            "Upgrade with: pip install -U 'nvidia-modelopt[all]'"
        )
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


def _build_prompts(args: argparse.Namespace) -> list[str]:
    prompts = args.prompt or DEFAULT_PROMPTS
    if args.calib_size <= 0:
        raise SystemExit("--calib-size must be positive.")
    if len(prompts) < args.calib_size:
        repeats = (args.calib_size + len(prompts) - 1) // len(prompts)
        prompts = (prompts * repeats)[: args.calib_size]
    return prompts[: args.calib_size]


def _build_w4a8_quant_config(mtq: Any) -> dict:
    """W4A8 = NVFP4 weights (4-bit FP4) + FP8 activations (E4M3).

    Prefer mtq.W4A8_NVFP4_FP8_CFG when available; otherwise patch
    NVFP4_DEFAULT_CFG by upgrading every input quantizer to FP8 (8-bit float
    with 7 mantissa-equivalent bits, mapping to E4M3).
    """
    if hasattr(mtq, "W4A8_NVFP4_FP8_CFG"):
        return copy.deepcopy(mtq.W4A8_NVFP4_FP8_CFG)
    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    quant_cfg = cfg.get("quant_cfg", {})
    for key, val in list(quant_cfg.items()):
        if isinstance(val, dict) and "input_quantizer" in key:
            val["num_bits"] = (8, 7)
            val.pop("block_sizes", None)
    return cfg


def _filter_func_wan22(name: str) -> bool:
    """Match quantizer names that should be skipped (kept in full precision).

    Includes the standard Wan2.2 disables (input/output projections, embedders,
    norms, scale_shift_table) plus the most extreme cross-attention outputs
    flagged by check_activation_variance.py with amax ratios > 10x.  Those
    layers cannot be safely quantized even with FP8 activations because their
    weight magnitudes also vary enough to produce noticeable error.
    """
    base = re.compile(
        r"(proj_out.*|"
        r".*(condition_embedder|patch_embedding|"
        r"norm_out|scale_shift_table|"
        r"timestep_proj_prepare|output_scale_shift_prepare).*)"
    )
    if base.match(name) is not None:
        return True
    # Extreme-variance cross-attention layers (attn2 to_out / to_k) at the
    # deep blocks flagged by check_activation_variance.py with ratio > 10x.
    extreme_attn2 = re.compile(r".*\.blocks\.(19|30|31|34|35|36|37|38|39)\.attn2\.to_(out\.0|k)\..*")
    return extreme_attn2.match(name) is not None


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


def _is_image_conditioned(pipe: DiffusionPipeline) -> bool:
    """True if the pipeline's __call__ accepts an `image` argument (TI2V/I2V)."""
    try:
        sig = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return False
    return "image" in sig.parameters


def _make_dummy_image(height: int, width: int):
    import numpy as np
    from PIL import Image

    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


def _list_transformers(pipe: DiffusionPipeline) -> list[tuple[str, torch.nn.Module]]:
    """Return [(attr_name, module)] for every DiT on the pipeline.

    Wan2.2 A14B has both `transformer` (low-noise) and `transformer_2`
    (high-noise).  TI2V-5B has only `transformer`.
    """
    out = []
    for attr in ("transformer", "transformer_2"):
        mod = getattr(pipe, attr, None)
        if mod is not None:
            out.append((attr, mod))
    return out


def _load_pipeline(model_path: str, dtype: torch.dtype) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    return pipe


def _build_forward_loop(pipe: DiffusionPipeline, args: argparse.Namespace, prompts: list[str]):
    generator = torch.Generator(device="cuda")

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
    if _is_image_conditioned(pipe):
        base_kwargs["image"] = _make_dummy_image(args.height, args.width)

    def forward_loop(*_unused_args, **_unused_kwargs) -> None:
        with torch.inference_mode():
            for idx, prompt in enumerate(prompts):
                generator.manual_seed(args.seed + idx)
                try:
                    pipe(prompt=prompt, generator=generator, guidance_scale=args.guidance_scale, **base_kwargs)
                except TypeError as exc:
                    if "guidance_scale" not in str(exc):
                        raise
                    pipe(prompt=prompt, generator=generator, **base_kwargs)

    return forward_loop


def _force_export_quantized_weights(backbone: torch.nn.Module, dtype: torch.dtype) -> int:
    """Convert fake-quantized BF16 weights to packed FP4 in-memory before save.

    ModelOpt stores weights as fake-quantized BF16 after mtq.quantize(); writing
    them to disk via save_pretrained would emit BF16 tensors with quantizer
    scales as side metadata.  We need true packed FP4 + scale tensors, which is
    what the upstream NVFP4 inference kernel expects.  Prefer mtq.compress()
    (public API) when available; fall back to walking modules and calling the
    private _export_quantized_weight helper.
    """
    try:
        import modelopt.torch.quantization as mtq

        if hasattr(mtq, "compress"):
            mtq.compress(backbone)
            return sum(1 for m in backbone.modules() if hasattr(m, "weight_quantizer"))
    except Exception as exc:
        print(f"[warn] mtq.compress() unavailable, falling back to private API: {exc}", file=sys.stderr)

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


def _wan22_quant_config_block() -> dict:
    """W4A8 metadata block: NVFP4 weights + FP8 activations.

    Written into each transformer's config.json so the vllm-omni factory's
    _detect_modelopt_method() routes the checkpoint to the NVFP4 kernel path.

    Includes both schemas: the compressed-tensors style (config_groups, ignore)
    for tooling that inspects per-group settings, and the flat ModelOpt schema
    (group_size, kv_cache_quant_algo, exclude_modules) required by upstream
    ModelOptNvFp4Config._from_config.
    """
    exclude_modules = [
        "condition_embedder*",
        "norm_out*",
        "output_scale_shift_prepare*",
        "patch_embedding*",
        "proj_out*",
        "scale_shift_table*",
        "timestep_proj_prepare*",
    ]
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "group",
                    "group_size": 16,
                    "symmetric": True,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": list(exclude_modules),
        "producer": {"name": "modelopt"},
        "quant_algo": "NVFP4",
        "quant_method": "modelopt_fp4",
        "group_size": 16,
        "kv_cache_quant_algo": None,
        "exclude_modules": list(exclude_modules),
    }


def _patch_quant_config(output_dir: Path) -> None:
    new_qc = _wan22_quant_config_block()
    for transformer_dir in ("transformer", "transformer_2"):
        cfg_path = output_dir / transformer_dir / "config.json"
        if not cfg_path.exists():
            continue
        with cfg_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        existing = cfg.get("quantization_config")
        block = copy.deepcopy(new_qc)
        if isinstance(existing, dict):
            producer = existing.get("producer")
            if isinstance(producer, dict):
                block["producer"] = producer
        cfg["quantization_config"] = block
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


def _save_pipeline_with_nvfp4_transformers(
    pipe: DiffusionPipeline,
    model_path: str,
    output_dir: Path,
    max_shard_size: str = "5GB",
) -> None:
    from modelopt.torch.export.diffusers_utils import hide_quantizers_from_state_dict

    src = Path(model_path)
    if not src.exists():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(model_path))

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(src, output_dir, ignore=shutil.ignore_patterns("transformer", "transformer_2"))

    for attr, backbone in _list_transformers(pipe):
        transformer_out = output_dir / attr
        with hide_quantizers_from_state_dict(backbone):
            backbone.save_pretrained(
                str(transformer_out),
                safe_serialization=True,
                max_shard_size=max_shard_size,
            )


def _summarize_export(output_dir: Path) -> None:
    print("Export summary:")
    for transformer_dir in ("transformer", "transformer_2"):
        cfg_path = output_dir / transformer_dir / "config.json"
        if not cfg_path.exists():
            continue
        with cfg_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config", {})
        act_bits = qc.get("config_groups", {}).get("group_0", {}).get("input_activations", {}).get("num_bits")
        print(
            f"  [{transformer_dir}] quant_method={qc.get('quant_method')} "
            f"quant_algo={qc.get('quant_algo')} act_bits={act_bits}"
        )


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ModelOpt NVFP4 quantization.")

    mtq = _require_modelopt()
    model_path, output_dir = _ensure_paths(args)
    dtype = _select_dtype(args.dtype)
    prompts = _build_prompts(args)

    print("Quantization plan:")
    print(f"  input:        {args.model}")
    print(f"  output:       {output_dir}")
    print(f"  dtype:        {dtype}")
    print(f"  height/width: {args.height}x{args.width}")
    print(f"  num_frames:   {args.num_frames}")
    print(f"  calib_size:   {len(prompts)}")
    print(f"  calib_steps:  {args.calib_steps}")
    print(f"  quantize_mha: {args.quantize_mha}")
    print("  quant_algo:   NVFP4 W4A8 (FP4 weights + FP8 activations)")
    print("  NOTE: Serving requires Blackwell (sm_100+). Calibration on H100 is fine.")

    pipe = _load_pipeline(model_path, dtype)
    transformers = _list_transformers(pipe)
    if not transformers:
        raise SystemExit("Pipeline has no transformer or transformer_2 attribute.")
    print(f"  found {len(transformers)} transformer(s) on pipeline: {', '.join(a for a, _ in transformers)}")
    if _is_image_conditioned(pipe):
        print("  pipeline accepts `image` -> using black placeholder for calibration")

    quant_config = _build_w4a8_quant_config(mtq)
    forward_loop = _build_forward_loop(pipe, args, prompts)

    # Quantize each DiT.  The forward_loop runs the full pipeline, so quantizers
    # on every attached transformer get calibrated during the same passes.
    for attr, backbone in transformers:
        print(f"\nQuantizing {attr}...")
        quantized = mtq.quantize(backbone, quant_config, forward_loop)
        if quantized is not None:
            setattr(pipe, attr, quantized)
            backbone = quantized
        _disable_known_problematic_quantizers(mtq, backbone, quantize_mha=args.quantize_mha)

    print("\nForcing NVFP4 weight serialization...")
    total_exported = 0
    for attr, backbone in _list_transformers(pipe):
        exported = _force_export_quantized_weights(backbone, dtype)
        print(f"  [{attr}] {exported} weights converted to NVFP4 in memory")
        total_exported += exported
    if total_exported == 0:
        raise SystemExit("No quantized weights were exported. Check the disable_quantizer regex.")

    print("\nSaving pipeline with NVFP4 transformer(s)...")
    _save_pipeline_with_nvfp4_transformers(pipe, model_path, output_dir)
    _patch_quant_config(output_dir)
    print(f"Saved to: {output_dir}")
    _summarize_export(output_dir)


if __name__ == "__main__":
    main()
