# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ablation study: FP8 quantization with different ignored_layers configs.

Spawns a separate subprocess for each config via text_to_image.py so that
GPU memory is fully released between runs (avoids OOM).  After all images
are generated, computes LPIPS (perceptual distance) between each FP8
variant and the BF16 baseline for the same prompt/seed pair.

Requirements (for metrics):
    pip install lpips

Usage:
    python ablation_fp8_ignored_layers.py
    python ablation_fp8_ignored_layers.py --model Qwen/Qwen-Image-2512
    python ablation_fp8_ignored_layers.py --output-dir outputs/ablation_2512
    python ablation_fp8_ignored_layers.py --skip-lpips   # images only

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
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    # Text-stream attention
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
    ("fp8_skip_all_attn", ["to_qkv", "to_out", "add_kv_proj", "to_add_out"]),
    ("fp8_skip_text_stream", ["add_kv_proj", "to_add_out", "txt_mlp"]),
    ("fp8_skip_attn_and_txt_mlp", ["to_qkv", "to_out", "add_kv_proj", "to_add_out", "txt_mlp"]),
]

# Diverse prompts spanning different domains for robust evaluation
DEFAULT_PROMPTS = [
    "a mountain landscape with a lake reflection at sunset, photorealistic",
    "a portrait of a woman with intricate jewelry, studio lighting",
    "a futuristic cityscape at night with neon lights and flying cars",
]

# Regexes to extract metrics from text_to_image.py stdout
_TIME_RE = re.compile(r"Total generation time:\s*([\d.]+)\s*seconds")
_MODEL_MEM_RE = re.compile(r"Model loaded memory:\s*([\d.]+)\s*GiB")
_PEAK_MEM_RE = re.compile(r"Peak memory:\s*([\d.]+)\s*GiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation study for FP8 ignored_layers on DiT models.")
    parser.add_argument("--model", default="Qwen/Qwen-Image-2512", help="Model name or path.")
    parser.add_argument("--output-dir", default="outputs/ablation_fp8", help="Directory to save images and results.")
    parser.add_argument("--prompts", nargs="+", default=None, help="Custom prompts (overrides defaults).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--configs", nargs="+", default=None, help="Run only these config names (default: all).")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile for stable benchmarking.")
    parser.add_argument("--skip-lpips", action="store_true", help="Skip LPIPS computation (generate images only).")
    return parser.parse_args()


def run_single_config(
    model: str,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_inference_steps: int,
    cfg_scale: float,
    guidance_scale: float,
    output_path: str,
    quantization: str | None,
    ignored_layers: list[str] | None,
    enforce_eager: bool,
) -> dict[str, float | None]:
    """Run text_to_image.py as a subprocess.

    Returns dict with keys: time_seconds, model_memory_gib, peak_memory_gib.
    Values are None on failure or if not reported.
    """
    script = str(Path(__file__).parent / "text_to_image.py")
    cmd = [
        sys.executable,
        script,
        "--model",
        model,
        "--prompt",
        prompt,
        "--seed",
        str(seed),
        "--height",
        str(height),
        "--width",
        str(width),
        "--num_inference_steps",
        str(num_inference_steps),
        "--cfg_scale",
        str(cfg_scale),
        "--guidance_scale",
        str(guidance_scale),
        "--output",
        output_path,
    ]
    if quantization:
        cmd += ["--quantization", quantization]
    if ignored_layers:
        cmd += ["--ignored-layers", ",".join(ignored_layers)]
    if enforce_eager:
        cmd += ["--enforce_eager"]

    metrics: dict[str, float | None] = {
        "time_seconds": None,
        "model_memory_gib": None,
        "peak_memory_gib": None,
    }

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    | {line}")
    if result.returncode != 0:
        print(f"    [ERROR] exit code {result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-10:]:
                print(f"    | {line}")
        return metrics

    stdout = result.stdout
    m = _TIME_RE.search(stdout)
    if m:
        metrics["time_seconds"] = float(m.group(1))
    else:
        print("    [WARN] Could not parse generation time from output")

    m = _MODEL_MEM_RE.search(stdout)
    if m:
        metrics["model_memory_gib"] = float(m.group(1))

    m = _PEAK_MEM_RE.search(stdout)
    if m:
        metrics["peak_memory_gib"] = float(m.group(1))

    return metrics


# ---------------------------------------------------------------------------
# LPIPS computation
# ---------------------------------------------------------------------------
def compute_lpips_scores(
    results: list[dict[str, Any]],
    baseline_name: str = "bf16_baseline",
) -> dict[tuple[str, int], float]:
    """Compute LPIPS between each config and the BF16 baseline.

    Returns a dict keyed by (config_name, prompt_index) -> lpips_score.
    """
    try:
        import lpips
        import torch
        from PIL import Image
        from torchvision import transforms
    except ImportError as e:
        print(f"\n[WARN] Cannot compute LPIPS: {e}")
        print("       Install with: pip install lpips torchvision")
        return {}

    print("\n" + "=" * 70)
    print("Computing LPIPS (perceptual distance vs BF16 baseline)...")
    print("=" * 70)

    # Build lookup: prompt_index -> baseline image path
    baseline_paths: dict[int, str] = {}
    for r in results:
        if r["config"] == baseline_name and r["image_path"]:
            baseline_paths[r["prompt_index"]] = r["image_path"]

    if not baseline_paths:
        print(f"[ERROR] No baseline images found for '{baseline_name}'")
        return {}

    # Initialise LPIPS (AlexNet backbone, lightweight and standard)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device)

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(0.5, 0.5),  # [-1, 1] as LPIPS expects
        ]
    )

    scores: dict[tuple[str, int], float] = {}
    for r in results:
        cfg = r["config"]
        p_idx = r["prompt_index"]
        if cfg == baseline_name or r["image_path"] is None:
            continue
        if p_idx not in baseline_paths:
            continue
        if not Path(r["image_path"]).exists():
            continue

        ref_img = Image.open(baseline_paths[p_idx]).convert("RGB")
        gen_img = Image.open(r["image_path"]).convert("RGB")

        ref_t = to_tensor(ref_img).unsqueeze(0).to(device)
        gen_t = to_tensor(gen_img).unsqueeze(0).to(device)

        with torch.no_grad():
            score = loss_fn(ref_t, gen_t).item()
        scores[(cfg, p_idx)] = score

    print(f"  Computed {len(scores)} LPIPS scores")
    return scores


def run_ablation(args: argparse.Namespace) -> None:
    prompts = args.prompts or DEFAULT_PROMPTS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter configs if user specified a subset
    configs = ABLATION_CONFIGS
    if args.configs:
        selected = set(args.configs)
        configs = [(name, layers) for name, layers in configs if name in selected]
        if not configs:
            raise ValueError(f"No matching configs found. Available: {[n for n, _ in ABLATION_CONFIGS]}")

    results: list[dict[str, Any]] = []
    total_runs = len(configs) * len(prompts)

    print(f"\n{'=' * 70}")
    print(f"FP8 Ablation Study: {args.model}")
    print(f"  Configs to test : {len(configs)}")
    print(f"  Prompts          : {len(prompts)}")
    print(f"  Total runs       : {total_runs}")
    print(f"  Image size       : {args.width}x{args.height}")
    print(f"  Inference steps  : {args.num_inference_steps}")
    print(f"  Seed             : {args.seed}")
    print(f"  Output directory : {output_dir}")
    print(f"  LPIPS            : {'skip' if args.skip_lpips else 'enabled'}")
    print("  Method           : subprocess per run (full GPU release)")
    print(f"{'=' * 70}\n")

    run_idx = 0
    for cfg_idx, (config_name, ignored_layers) in enumerate(configs):
        is_baseline = ignored_layers is None
        quantization = None if is_baseline else "fp8"

        print(f"\n[{cfg_idx + 1}/{len(configs)}] Config: {config_name}")
        if is_baseline:
            print("  Mode: BF16 (no quantization)")
        else:
            print(f"  Mode: FP8, ignored_layers={ignored_layers or '(none)'}")
        print("-" * 50)

        for p_idx, prompt in enumerate(prompts):
            run_idx += 1
            short_prompt = prompt[:50] + ("..." if len(prompt) > 50 else "")
            print(f"  [{run_idx}/{total_runs}] Prompt {p_idx}: {short_prompt}")

            img_filename = f"{config_name}_prompt{p_idx}.png"
            img_path = str(output_dir / img_filename)

            metrics = run_single_config(
                model=args.model,
                prompt=prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                guidance_scale=args.guidance_scale,
                output_path=img_path,
                quantization=quantization,
                ignored_layers=ignored_layers if ignored_layers else None,
                enforce_eager=args.enforce_eager,
            )

            elapsed = metrics["time_seconds"]
            mem_model = metrics["model_memory_gib"]
            mem_peak = metrics["peak_memory_gib"]
            parts = [f"{elapsed:.2f}s" if elapsed else "FAILED"]
            if mem_model is not None:
                parts.append(f"model={mem_model:.2f}GiB")
            if mem_peak is not None:
                parts.append(f"peak={mem_peak:.2f}GiB")
            print(f"    -> {', '.join(parts)}  saved: {img_path}")

            results.append(
                {
                    "config": config_name,
                    "quantization": quantization or "none",
                    "ignored_layers": ",".join(ignored_layers) if ignored_layers else "",
                    "prompt_index": p_idx,
                    "prompt": prompt,
                    "time_seconds": round(elapsed, 4) if elapsed else None,
                    "model_memory_gib": round(mem_model, 4) if mem_model else None,
                    "peak_memory_gib": round(mem_peak, 4) if mem_peak else None,
                    "image_path": img_path,
                }
            )

    # ── Compute LPIPS ──
    lpips_scores: dict[tuple[str, int], float] = {}
    if not args.skip_lpips:
        lpips_scores = compute_lpips_scores(results)
        for r in results:
            key = (r["config"], r["prompt_index"])
            r["lpips_vs_bf16"] = round(lpips_scores[key], 6) if key in lpips_scores else None

    # ── Save results ──
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

    has_lpips = bool(lpips_scores)
    header_lpips = f"  {'Mean LPIPS':>11}" if has_lpips else ""
    print(f"  {'Config':<28} {'Avg Time':>9} {'Model Mem':>10} {'Peak Mem':>10}{header_lpips}  {'Ignored Layers'}")
    print("-" * (90 if has_lpips else 78))

    times_by_config: dict[str, list[float]] = defaultdict(list)
    model_mem_by_config: dict[str, list[float]] = defaultdict(list)
    peak_mem_by_config: dict[str, list[float]] = defaultdict(list)
    lpips_by_config: dict[str, list[float]] = defaultdict(list)
    layers_by_config: dict[str, str] = {}
    for r in results:
        if r["time_seconds"] is not None:
            times_by_config[r["config"]].append(r["time_seconds"])
        if r.get("model_memory_gib") is not None:
            model_mem_by_config[r["config"]].append(r["model_memory_gib"])
        if r.get("peak_memory_gib") is not None:
            peak_mem_by_config[r["config"]].append(r["peak_memory_gib"])
        if r.get("lpips_vs_bf16") is not None:
            lpips_by_config[r["config"]].append(r["lpips_vs_bf16"])
        layers_by_config[r["config"]] = r["ignored_layers"] or "(none)"

    baseline_avg = None
    for config_name, times in times_by_config.items():
        avg_time = sum(times) / len(times)
        if config_name == "bf16_baseline":
            baseline_avg = avg_time

        speedup = ""
        if baseline_avg and config_name != "bf16_baseline":
            speedup = f"  ({baseline_avg / avg_time:.2f}x)"

        model_mem_col = ""
        if config_name in model_mem_by_config:
            avg_mm = sum(model_mem_by_config[config_name]) / len(model_mem_by_config[config_name])
            model_mem_col = f"{avg_mm:>8.2f}Gi"

        peak_mem_col = ""
        if config_name in peak_mem_by_config:
            avg_pm = sum(peak_mem_by_config[config_name]) / len(peak_mem_by_config[config_name])
            peak_mem_col = f"{avg_pm:>8.2f}Gi"

        lpips_col = ""
        if has_lpips and config_name in lpips_by_config:
            avg_lpips = sum(lpips_by_config[config_name]) / len(lpips_by_config[config_name])
            lpips_col = f"  {avg_lpips:>11.4f}"
        elif has_lpips and config_name == "bf16_baseline":
            lpips_col = f"  {'(ref)':>11}"

        layers = layers_by_config[config_name]
        print(
            f"  {config_name:<28} {avg_time:>7.2f}s {model_mem_col:>10} {peak_mem_col:>10}{lpips_col}  {layers}{speedup}"
        )

    # Highlight best FP8 config by LPIPS
    if has_lpips and lpips_by_config:
        best_cfg = min(lpips_by_config, key=lambda c: sum(lpips_by_config[c]) / len(lpips_by_config[c]))
        best_lpips = sum(lpips_by_config[best_cfg]) / len(lpips_by_config[best_cfg])
        print(f"\n  >> Best FP8 config by LPIPS: {best_cfg} (mean={best_lpips:.4f})")

    failed = [r["config"] for r in results if r["time_seconds"] is None]
    if failed:
        print(f"\n  FAILED: {', '.join(set(failed))}")

    print("\nResults saved to:")
    print(f"  CSV  : {csv_path}")
    print(f"  JSON : {json_path}")
    print(f"  Images: {output_dir}/")


if __name__ == "__main__":
    run_ablation(parse_args())
