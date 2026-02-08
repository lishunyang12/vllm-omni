# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ablation study: FP8 quantization with different ignored_layers configs.

Measures generation time and saves images for visual quality comparison
across different ignored_layers combinations on Qwen-Image-2512.

Usage:
    python ablation_fp8_ignored_layers.py
    python ablation_fp8_ignored_layers.py --model Qwen/Qwen-Image-2512
    python ablation_fp8_ignored_layers.py --output-dir outputs/ablation_2512

Layer reference (QwenImageTransformerBlock):
    Attention (QwenImageCrossAttention):
        to_qkv       - Image-stream Q/K/V projection  (QKVParallelLinear)
        to_out        - Image-stream output projection  (RowParallelLinear)
        add_kv_proj   - Text-stream Q/K/V projection   (QKVParallelLinear)
        to_add_out    - Text-stream output projection   (RowParallelLinear)
    MLP (FeedForward):
        img_mlp       - Image-stream MLP (2 linear layers inside)
        txt_mlp       - Text-stream MLP  (2 linear layers inside)
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, logger
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
# Each entry: (config_name, ignored_layers_list_or_None)
#   - None means no quantization (BF16 baseline)
#   - []   means FP8 on ALL layers (no layers ignored)
#   - [..] means FP8 with those layers kept in BF16
ABLATION_CONFIGS: list[tuple[str, list[str] | None]] = [
    # ── Baselines ──
    ("bf16_baseline", None),
    ("fp8_all_layers", []),

    # ── Single-group ablations ──
    # Text-stream attention (most likely sensitive in dual-stream arch)
    ("fp8_skip_text_attn_kv", ["add_kv_proj"]),
    ("fp8_skip_text_attn_out", ["to_add_out"]),
    ("fp8_skip_text_attn_all", ["add_kv_proj", "to_add_out"]),

    # Image-stream attention
    ("fp8_skip_img_attn_qkv", ["to_qkv"]),
    ("fp8_skip_img_attn_out", ["to_out"]),
    ("fp8_skip_img_attn_all", ["to_qkv", "to_out"]),

    # MLP
    ("fp8_skip_img_mlp", ["img_mlp"]),
    ("fp8_skip_txt_mlp", ["txt_mlp"]),
    ("fp8_skip_all_mlp", ["img_mlp", "txt_mlp"]),

    # ── Combined ablations ──
    # All attention kept in BF16
    ("fp8_skip_all_attn", ["to_qkv", "to_out", "add_kv_proj", "to_add_out"]),
    # Text-stream entirely in BF16 (attention + MLP)
    ("fp8_skip_text_stream", ["add_kv_proj", "to_add_out", "txt_mlp"]),
    # All attention + text MLP in BF16 (conservative)
    ("fp8_skip_attn_and_txt_mlp",
     ["to_qkv", "to_out", "add_kv_proj", "to_add_out", "txt_mlp"]),
]

# Prompts for quality evaluation across different domains
DEFAULT_PROMPTS = [
    "a mountain landscape with a lake reflection at sunset, photorealistic",
    "a portrait of a woman with intricate jewelry, studio lighting",
    "a futuristic cityscape at night with neon lights and flying cars",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study for FP8 ignored_layers on Qwen-Image.")
    parser.add_argument(
        "--model", default="Qwen/Qwen-Image-2512",
        help="Model name or path.")
    parser.add_argument(
        "--output-dir", default="outputs/ablation_fp8",
        help="Directory to save images and results.")
    parser.add_argument(
        "--prompts", nargs="+", default=None,
        help="Custom prompts (overrides defaults).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Run only these config names (default: all).")
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="Disable torch.compile for stable benchmarking.")
    return parser.parse_args()


def build_omni(
    model: str,
    quantization: str | None,
    ignored_layers: list[str] | None,
    enforce_eager: bool,
) -> Omni:
    """Create an Omni instance with the given quantization settings."""
    quant_kwargs: dict[str, Any] = {}
    if quantization and ignored_layers:
        quant_kwargs["quantization_config"] = {
            "method": quantization,
            "ignored_layers": ignored_layers,
        }
    elif quantization:
        quant_kwargs["quantization"] = quantization

    return Omni(
        model=model,
        enforce_eager=enforce_eager,
        **quant_kwargs,
    )


def generate_image(
    omni: Omni,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_inference_steps: int,
    cfg_scale: float,
    guidance_scale: float,
) -> tuple[Any, float]:
    """Generate one image, return (PIL.Image, elapsed_seconds)."""
    generator = torch.Generator(
        device=current_omni_platform.device_type).manual_seed(seed)

    t0 = time.perf_counter()
    outputs = omni.generate(
        {"prompt": prompt},
        OmniDiffusionSamplingParams(
            height=height,
            width=width,
            generator=generator,
            true_cfg_scale=cfg_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ),
    )
    elapsed = time.perf_counter() - t0

    image = outputs[0].request_output[0].images[0]
    return image, elapsed


def run_ablation(args: argparse.Namespace) -> None:
    prompts = args.prompts or DEFAULT_PROMPTS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter configs if user specified a subset
    configs = ABLATION_CONFIGS
    if args.configs:
        selected = set(args.configs)
        configs = [(n, l) for n, l in configs if n in selected]
        if not configs:
            raise ValueError(
                f"No matching configs found. Available: "
                f"{[n for n, _ in ABLATION_CONFIGS]}")

    results: list[dict[str, Any]] = []

    print(f"\n{'=' * 70}")
    print(f"FP8 Ablation Study: {args.model}")
    print(f"  Configs to test : {len(configs)}")
    print(f"  Prompts          : {len(prompts)}")
    print(f"  Image size       : {args.width}x{args.height}")
    print(f"  Inference steps  : {args.num_inference_steps}")
    print(f"  Seed             : {args.seed}")
    print(f"  Output directory : {output_dir}")
    print(f"{'=' * 70}\n")

    for cfg_idx, (config_name, ignored_layers) in enumerate(configs):
        is_baseline = ignored_layers is None
        quantization = None if is_baseline else "fp8"

        print(f"\n[{cfg_idx + 1}/{len(configs)}] Config: {config_name}")
        if is_baseline:
            print("  Mode: BF16 (no quantization)")
        else:
            print(f"  Mode: FP8, ignored_layers={ignored_layers or '(none)'}")
        print("-" * 50)

        # Build model for this config
        omni = build_omni(
            model=args.model,
            quantization=quantization,
            ignored_layers=ignored_layers,
            enforce_eager=args.enforce_eager,
        )

        for p_idx, prompt in enumerate(prompts):
            short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
            print(f"  Prompt {p_idx + 1}/{len(prompts)}: {short_prompt}")

            image, elapsed = generate_image(
                omni=omni,
                prompt=prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                guidance_scale=args.guidance_scale,
            )

            # Save image
            img_filename = f"{config_name}_prompt{p_idx}.png"
            img_path = output_dir / img_filename
            image.save(img_path)
            print(f"    -> {elapsed:.2f}s  saved: {img_path}")

            results.append({
                "config": config_name,
                "quantization": quantization or "none",
                "ignored_layers": ",".join(ignored_layers) if ignored_layers else "",
                "prompt_index": p_idx,
                "prompt": prompt,
                "time_seconds": round(elapsed, 4),
                "image_path": str(img_path),
            })

        # Free GPU memory between configs
        del omni
        torch.cuda.empty_cache()

    # ── Save summary ──
    csv_path = output_dir / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary table ──
    print(f"\n\n{'=' * 70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Config':<30} {'Avg Time (s)':>12}  {'Ignored Layers'}")
    print("-" * 70)

    # Group by config
    from collections import defaultdict
    times_by_config: dict[str, list[float]] = defaultdict(list)
    layers_by_config: dict[str, str] = {}
    for r in results:
        times_by_config[r["config"]].append(r["time_seconds"])
        layers_by_config[r["config"]] = r["ignored_layers"] or "(none)"

    baseline_avg = None
    for config_name, times in times_by_config.items():
        avg = sum(times) / len(times)
        if config_name == "bf16_baseline":
            baseline_avg = avg
        speedup = ""
        if baseline_avg and config_name != "bf16_baseline":
            speedup = f"  ({baseline_avg / avg:.2f}x vs BF16)"
        print(f"  {config_name:<28} {avg:>10.2f}s  "
              f"{layers_by_config[config_name]}{speedup}")

    print(f"\nResults saved to:")
    print(f"  CSV  : {csv_path}")
    print(f"  JSON : {json_path}")
    print(f"  Images: {output_dir}/")


if __name__ == "__main__":
    run_ablation(parse_args())
