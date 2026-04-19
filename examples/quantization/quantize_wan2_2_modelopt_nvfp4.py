#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize Wan2.2 (TI2V-5B, 704x1280 T2V) to a ModelOpt NVFP4 Hugging Face checkpoint.

Near-identical to quantize_wan2_2_modelopt_fp8.py but uses ModelOpt's
NVFP4_DEFAULT_CFG (W4A4 FP4 with 16-element group-wise scales along the input
dim) instead of FP8_DEFAULT_CFG. Produces ~4x weight compression vs BF16.

IMPORTANT - SERVING REQUIRES BLACKWELL:
    NVFP4 kernels (CutlassNvFp4ScaledMMLinearKernel) require SM_100 / Blackwell
    hardware (B200, RTX 5090). Calibration runs fine on H100 - this script just
    writes FP4 weights + FP32 group scales to disk - but you cannot validate or
    serve the resulting checkpoint on H100. Publish to HF Hub and test on
    Blackwell when available.

Example:
    python examples/quantization/quantize_wan2_2_modelopt_nvfp4.py \\
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \\
        --output ./wan22-ti2v-modelopt-nvfp4 \\
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
    p.add_argument("--output", required=True, help="Output directory for the ModelOpt NVFP4 checkpoint.")
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--calib-steps", type=int, default=10)
    p.add_argument("--calib-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument(
        "--quantize-mha",
        action="store_true",
        help="Enable attention K/V/softmax quantizers. Off by default.",
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
    """ModelOpt NVFP4 metadata. 4-bit FP4 weights with 16-element group scales."""
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {"dynamic": False, "num_bits": 4, "type": "float"},
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
        "quant_algo": "NVFP4",
        "quant_method": "modelopt",
    }


def _patch_quant_config(output_dir: Path) -> None:
    cfg_path = output_dir / "transformer" / "config.json"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    new_qc = _wan22_quant_config_block()
    existing = cfg.get("quantization_config")
    if isinstance(existing, dict):
        producer = existing.get("producer")
        if isinstance(producer, dict):
            new_qc["producer"] = producer

    cfg["quantization_config"] = new_qc
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _save_pipeline_with_nvfp4_transformer(
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

    transformer_out = output_dir / "transformer"
    with hide_quantizers_from_state_dict(pipe.transformer):
        pipe.transformer.save_pretrained(
            str(transformer_out),
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )


def _summarize_export(output_dir: Path) -> None:
    cfg_path = output_dir / "transformer" / "config.json"
    if not cfg_path.exists():
        print(f"[warn] {cfg_path} missing.", file=sys.stderr)
        return
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config", {})
    print("Export summary:")
    print(f"  quant_method: {qc.get('quant_method')}")
    print(f"  quant_algo:   {qc.get('quant_algo')}")
    print(f"  config path:  {cfg_path}")


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
    print("  quant_algo:   NVFP4 (W4A4 FP4, 16-element group-wise scales)")
    print("  NOTE: Serving this checkpoint requires Blackwell (sm_100+). Calibration on H100 is fine.")

    pipe = _load_pipeline(model_path, dtype)
    backbone = pipe.transformer

    quant_config = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)

    forward_loop = _build_forward_loop(pipe, args, prompts)
    quantized = mtq.quantize(backbone, quant_config, forward_loop)
    if quantized is not None:
        pipe.transformer = quantized
        backbone = quantized

    _disable_known_problematic_quantizers(mtq, backbone, quantize_mha=args.quantize_mha)

    print("\nForcing NVFP4 weight serialization...")
    exported = _force_export_quantized_weights(backbone, dtype)
    print(f"  -> {exported} weights converted to NVFP4 in memory")
    if exported == 0:
        raise SystemExit("No quantized weights were exported. Check the disable_quantizer regex.")

    print("\nSaving pipeline with NVFP4 transformer...")
    _save_pipeline_with_nvfp4_transformer(pipe, model_path, output_dir)
    _patch_quant_config(output_dir)
    print(f"Saved to: {output_dir}")
    _summarize_export(output_dir)


if __name__ == "__main__":
    main()
