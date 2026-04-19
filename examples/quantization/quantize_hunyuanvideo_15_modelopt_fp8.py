#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize HunyuanVideo-1.5 (480p T2V) to a ModelOpt FP8 Hugging Face checkpoint.

Calibrates the DiT transformer using a small video prompt set and exports a
diffusers-style directory whose transformer carries ModelOpt FP8 metadata.
The exported checkpoint is consumable by vllm-omni's ModelOpt FP8 adapter
(see vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt_fp8.py).

Layers kept full precision match the #2728 / #2795 pattern: modulation,
AdaLayerNorm, entry/exit projections, embeddings, the token refiner path,
and final proj_out. MHA quantizers are off by default; HV-1.5 self-attention
empirically degrades under FP8 (see #2920 ablation).

Example:
    python examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py \\
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \\
        --output ./hv15-480p-modelopt-fp8 \\
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
    p.add_argument("--model", required=True, help="Input HV-1.5 diffusers directory or HF id.")
    p.add_argument("--output", required=True, help="Output directory for the ModelOpt FP8 checkpoint.")
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument(
        "--num-frames",
        type=int,
        default=33,
        help="Frames per calibration sample. 33 matches the typical short benchmark.",
    )
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument(
        "--calib-steps",
        type=int,
        default=10,
        help="Denoising steps per calibration prompt (10 is enough for amax statistics).",
    )
    p.add_argument("--calib-size", type=int, default=8, help="How many prompts to use for calibration.")
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
        help="Enable FP8 attention K/V/softmax quantizers. Off by default — empirically degrades HV-1.5 video output.",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return p


def _require_modelopt() -> tuple[Any, Any]:
    try:
        import modelopt.torch.quantization as mtq
        from modelopt.torch.export import export_hf_checkpoint
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "NVIDIA ModelOpt is not installed. Install with:\n"
            "  pip install 'nvidia-modelopt[all]'\n"
            f"Original error: {exc}"
        ) from exc
    return mtq, export_hf_checkpoint


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


# Layers to KEEP at full precision (mirror of the #2920 wiring + #2728/#2795 skip pattern).
# - x_embedder, image_embedder, context_embedder*, time_embed*, cond_type_embed: entry/embedding
# - norm_out, norm1*.linear, norm1_context*.linear, norm2*, norm2_context*: AdaLayerNorm modulation
# - proj_out: final output projection
# - token_refiner*: text-encoder refinement uses diffusers raw nn.Linear
def _filter_func_hv15(name: str) -> bool:
    pattern = re.compile(
        r"(proj_out.*|"
        r".*(x_embedder|image_embedder|context_embedder|context_embedder_2|"
        r"time_embed|cond_type_embed|"
        r"norm_out|norm1\.linear|norm1_context\.linear|norm2|norm2_context|"
        r"token_refiner).*)"
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
    mtq.disable_quantizer(backbone, _filter_func_hv15)
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

    def forward_loop(*_unused_args, **_unused_kwargs) -> None:
        with torch.inference_mode():
            for idx, prompt in enumerate(prompts):
                generator.manual_seed(args.seed + idx)
                pipe(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.calib_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    output_type="latent",
                )

    return forward_loop


def _summarize_export(output_dir: Path) -> None:
    cfg_path = output_dir / "transformer" / "config.json"
    if not cfg_path.exists():
        print(f"[warn] {cfg_path} missing.", file=sys.stderr)
        return
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        print("[warn] No quantization_config in transformer/config.json.", file=sys.stderr)
        return
    print("Export summary:")
    print(f"  quant_method: {qc.get('quant_method')}")
    print(f"  quant_algo:   {qc.get('quant_algo')}")
    producer = qc.get("producer")
    if isinstance(producer, dict):
        print(f"  producer:     {producer.get('name')} {producer.get('version')}")
    print(f"  config path:  {cfg_path}")


def _export_with_fallback(
    pipe: DiffusionPipeline,
    export_hf_checkpoint: Any,
    model_path: str,
    output_dir: Path,
) -> str:
    try:
        export_hf_checkpoint(pipe, export_dir=str(output_dir))
        return "pipeline"
    except Exception as exc:
        print(
            f"[warn] Whole-pipeline export failed, falling back to transformer-only export: {exc}",
            file=sys.stderr,
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        # Resolve the source directory: a local path stays as-is; an HF id resolves via snapshot_download.
        src = Path(model_path)
        if not src.exists():
            from huggingface_hub import snapshot_download

            src = Path(snapshot_download(model_path))
        shutil.copytree(src, output_dir, dirs_exist_ok=True)
        shutil.rmtree(output_dir / "transformer", ignore_errors=True)
        export_hf_checkpoint(pipe.transformer, export_dir=str(output_dir / "transformer"))
        return "transformer"


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ModelOpt FP8 quantization.")

    mtq, export_hf_checkpoint = _require_modelopt()
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

    pipe = _load_pipeline(model_path, dtype)
    backbone = pipe.transformer

    quant_config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)

    forward_loop = _build_forward_loop(pipe, args, prompts)
    quantized = mtq.quantize(backbone, quant_config, forward_loop)
    if quantized is not None:
        pipe.transformer = quantized
        backbone = quantized

    _disable_known_problematic_quantizers(mtq, backbone, quantize_mha=args.quantize_mha)

    export_mode = _export_with_fallback(pipe, export_hf_checkpoint, model_path, output_dir)
    print(f"Export mode: {export_mode}")
    _summarize_export(output_dir)

    print("\nNext: validate the checkpoint with vllm-omni:")
    print(
        "  python examples/offline_inference/text_to_video/text_to_video.py \\\n"
        f"    --model {output_dir} \\\n"
        "    --stage-configs-path vllm_omni/model_executor/stage_configs/hunyuan_video_15_dit_fp8.yaml \\\n"
        "    --prompt 'A dog running across a field of golden wheat.' \\\n"
        f"    --height {args.height} --width {args.width} --num-frames {args.num_frames} \\\n"
        "    --num-inference-steps 30 --guidance-scale 6.0 --seed 42 \\\n"
        "    --output outputs/hv15_modelopt_fp8.mp4 \\\n"
        "    --enforce-eager"
    )


if __name__ == "__main__":
    main()
