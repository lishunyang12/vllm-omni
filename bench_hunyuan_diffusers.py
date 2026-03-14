# SPDX-License-Identifier: Apache-2.0
"""
Benchmark HunyuanVideo-1.5 on pure diffusers (baseline).

Based on: https://huggingface.co/docs/diffusers/api/pipelines/hunyuanvideo1_5

Usage:
    python bench_hunyuan_diffusers.py [--skip-i2v] [--i2v-image PATH]

Results are written to bench_results_diffusers.json
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch

SEED = 42
OUTPUT_DIR = "bench_outputs"

T2V_PROMPT = (
    "A little girl wearing a straw hat runs through a summer meadow "
    "full of wildflowers. A wide shot is used, with the camera panning "
    "right to follow her."
)
I2V_PROMPT = (
    "The camera follows the puppy as it runs forward on the grass, "
    "its four legs alternating steps, its tail held high and wagging "
    "side to side."
)

EXPERIMENTS = [
    {
        "name": "T2V 480p short BF16",
        "task": "t2v",
        "model": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "pipeline_cls": "HunyuanVideo15Pipeline",
        "height": 480, "width": 832, "frames": 33, "steps": 30,
        "guidance_scale": 6.0, "flow_shift": 5.0,
        "precision": "BF16",
    },
    {
        "name": "T2V 480p full BF16",
        "task": "t2v",
        "model": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "pipeline_cls": "HunyuanVideo15Pipeline",
        "height": 480, "width": 832, "frames": 121, "steps": 50,
        "guidance_scale": 6.0, "flow_shift": 5.0,
        "precision": "BF16",
    },
    {
        "name": "T2V 720p short FP8+tiling",
        "task": "t2v",
        "model": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        "pipeline_cls": "HunyuanVideo15Pipeline",
        "height": 720, "width": 1280, "frames": 33, "steps": 30,
        "guidance_scale": 6.0, "flow_shift": 9.0,
        "precision": "FP8+tiling",
    },
    {
        "name": "I2V 480p short BF16",
        "task": "i2v",
        "model": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
        "pipeline_cls": "HunyuanVideo15ImageToVideoPipeline",
        "height": 480, "width": 832, "frames": 33, "steps": 30,
        "guidance_scale": 6.0, "flow_shift": 5.0,
        "precision": "BF16",
    },
]


def _save_frames(frames, path):
    """Save raw frames as .pt for PSNR/SSIM comparison."""
    import PIL.Image
    if isinstance(frames, list) and len(frames) > 0:
        if isinstance(frames[0], PIL.Image.Image):
            arr = np.stack([np.asarray(f) for f in frames])
            torch.save(torch.from_numpy(arr), path)
            return
    if isinstance(frames, np.ndarray):
        torch.save(torch.from_numpy(frames), path)
    elif isinstance(frames, torch.Tensor):
        torch.save(frames.cpu(), path)
    else:
        arr = np.stack([np.asarray(f) for f in frames])
        torch.save(torch.from_numpy(arr), path)


def load_pipeline(exp):
    """Load a diffusers pipeline following official HunyuanVideo-1.5 docs."""
    import diffusers

    cls = getattr(diffusers, exp["pipeline_cls"])
    pipe = cls.from_pretrained(
        exp["model"],
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Enable VAE tiling (official pattern: pipe.vae.enable_tiling())
    pipe.vae.enable_tiling()

    # Set guidance_scale via guider (v1.5 uses guiders, not guidance_scale param)
    # Default guider is ClassifierFreeGuidance with guidance_scale=6.0
    if exp.get("guidance_scale") is not None and hasattr(pipe, "guider"):
        pipe.guider = pipe.guider.new(guidance_scale=exp["guidance_scale"])

    # Set flow_shift on scheduler
    if exp.get("flow_shift") is not None and hasattr(pipe, "scheduler"):
        if hasattr(pipe.scheduler, "_shift"):
            pipe.scheduler._shift = exp["flow_shift"]

    return pipe


def run_t2v(exp, pipe):
    """Run T2V with diffusers HunyuanVideo15Pipeline."""
    generator = torch.Generator("cuda").manual_seed(SEED)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    output = pipe(
        prompt=T2V_PROMPT,
        height=exp["height"],
        width=exp["width"],
        num_frames=exp["frames"],
        num_inference_steps=exp["steps"],
        generator=generator,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    frames = output.frames[0] if hasattr(output, "frames") else output
    return elapsed, peak_vram, frames


def run_i2v(exp, pipe, image_path):
    """Run I2V with diffusers HunyuanVideo15ImageToVideoPipeline."""
    import PIL.Image

    image = PIL.Image.open(image_path).convert("RGB")

    generator = torch.Generator("cuda").manual_seed(SEED)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    output = pipe(
        image=image,
        prompt=I2V_PROMPT,
        num_frames=exp["frames"],
        num_inference_steps=exp["steps"],
        generator=generator,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    frames = output.frames[0] if hasattr(output, "frames") else output
    return elapsed, peak_vram, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-i2v", action="store_true")
    parser.add_argument("--i2v-image", type=str, default="test_input.jpg")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--experiments", type=str, nargs="*", default=None,
                        help="Run only specific experiments by name substring")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    prev_model_key = None
    pipe = None

    for exp in EXPERIMENTS:
        if exp["task"] == "i2v" and args.skip_i2v:
            print(f"SKIP: {exp['name']} (--skip-i2v)")
            continue
        if args.experiments:
            if not any(s.lower() in exp["name"].lower() for s in args.experiments):
                print(f"SKIP: {exp['name']} (not in filter)")
                continue

        model_key = exp["model"]
        if model_key != prev_model_key:
            if pipe is not None:
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
            print(f"\n{'='*60}")
            print(f"Loading: {exp['model']} ({exp['precision']})")
            print(f"{'='*60}")
            pipe = load_pipeline(exp)
            prev_model_key = model_key

        print(f"\n>>> Running: {exp['name']}")
        print(f"    {exp['height']}x{exp['width']}, {exp['frames']}f, {exp['steps']}steps")

        try:
            if exp["task"] == "t2v":
                elapsed, peak_vram, frames = run_t2v(exp, pipe)
            else:
                elapsed, peak_vram, frames = run_i2v(exp, pipe, args.i2v_image)

            safe_name = exp["name"].replace(" ", "_").replace("+", "_")
            frames_path = os.path.join(args.output_dir, f"diffusers_{safe_name}_frames.pt")
            try:
                _save_frames(frames, frames_path)
                print(f"    Frames saved: {frames_path}")
            except Exception as e:
                print(f"    WARNING: Could not save frames: {e}")
                frames_path = None

            record = {
                "name": exp["name"],
                "framework": "diffusers",
                "task": exp["task"],
                "model": exp["model"],
                "resolution": f"{exp['height']}x{exp['width']}",
                "frames": exp["frames"],
                "steps": exp["steps"],
                "precision": exp["precision"],
                "latency_s": round(elapsed, 1),
                "peak_vram_gib": round(peak_vram, 2),
                "frames_path": frames_path,
            }
            results.append(record)

            print(f"    Latency:   {elapsed:.1f}s")
            print(f"    Peak VRAM: {peak_vram:.2f} GiB")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": exp["name"],
                "framework": "diffusers",
                "error": str(e),
            })

    results_path = os.path.join(args.output_dir, "bench_results_diffusers.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"{'='*60}")

    print(f"\n{'Name':<30} {'Latency':>10} {'VRAM':>12}")
    print("-" * 55)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} {'ERROR':>10} {'':>12}")
        else:
            print(f"{r['name']:<30} {r['latency_s']:>8.1f}s {r['peak_vram_gib']:>10.2f} GiB")


if __name__ == "__main__":
    main()
