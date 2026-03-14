# SPDX-License-Identifier: Apache-2.0
"""
Benchmark HunyuanVideo-1.5 on vLLM-OMNI.

Records latency, peak VRAM, and saves raw frames (.pt) for accuracy comparison.
Output videos are NOT saved — upload those manually.

Usage:
    python bench_hunyuan_vllm_omni.py [--skip-i2v] [--i2v-image PATH]

Results are written to bench_results_vllm_omni.json
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

# ── Shared test config ──────────────────────────────────────────────
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
SEED = 42
OUTPUT_DIR = "bench_outputs"

EXPERIMENTS = [
    # (name, task, model_id, height, width, frames, steps, guidance, flow_shift, precision, extra_kwargs)
    {
        "name": "T2V 480p short BF16",
        "task": "t2v",
        "model": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "height": 480, "width": 832, "frames": 33, "steps": 30,
        "guidance_scale": 6.0, "flow_shift": 5.0,
        "precision": "BF16", "quantization": None,
        "vae_use_tiling": False,
    },
    {
        "name": "T2V 480p full BF16",
        "task": "t2v",
        "model": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "height": 480, "width": 832, "frames": 121, "steps": 50,
        "guidance_scale": 6.0, "flow_shift": 5.0,
        "precision": "BF16", "quantization": None,
        "vae_use_tiling": True,
    },
    {
        "name": "T2V 480p full FP8",
        "task": "t2v",
        "model": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "height": 480, "width": 832, "frames": 121, "steps": 50,
        "guidance_scale": 6.0, "flow_shift": 5.0,
        "precision": "FP8", "quantization": "fp8",
        "vae_use_tiling": True,
    },
    {
        "name": "T2V 720p short FP8+tiling",
        "task": "t2v",
        "model": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        "height": 720, "width": 1280, "frames": 33, "steps": 30,
        "guidance_scale": 6.0, "flow_shift": 9.0,
        "precision": "FP8+tiling", "quantization": "fp8",
        "vae_use_tiling": True,
    },
    {
        "name": "I2V 480p short BF16",
        "task": "i2v",
        "model": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
        "height": 480, "width": 832, "frames": 33, "steps": 30,
        "guidance_scale": 6.0, "flow_shift": 5.0,
        "precision": "BF16", "quantization": None,
        "vae_use_tiling": False,
    },
]


def _save_frames(frames, path):
    """Save raw frames as .pt for PSNR/SSIM comparison."""
    if isinstance(frames, list):
        try:
            import PIL.Image
            if len(frames) > 0 and isinstance(frames[0], PIL.Image.Image):
                arr = np.stack([np.asarray(f) for f in frames])
                torch.save(torch.from_numpy(arr), path)
                return
        except Exception:
            pass
        arr = np.stack([np.asarray(f) for f in frames])
        torch.save(torch.from_numpy(arr), path)
    elif isinstance(frames, np.ndarray):
        torch.save(torch.from_numpy(frames), path)
    elif isinstance(frames, torch.Tensor):
        torch.save(frames.cpu(), path)


def _extract_frames(result):
    """Extract raw frame data from OmniRequestOutput."""
    from vllm_omni.outputs import OmniRequestOutput

    if isinstance(result, list):
        result = result[0] if result else None
    if isinstance(result, OmniRequestOutput):
        if result.is_pipeline_output and result.request_output is not None:
            inner = result.request_output
            if isinstance(inner, list):
                inner = inner[0] if inner else None
            if isinstance(inner, OmniRequestOutput):
                result = inner
        if isinstance(result, OmniRequestOutput) and result.images:
            return result.images
    return result


def run_t2v(exp, omni):
    """Run a T2V experiment."""
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED)

    prompt_dict = {"prompt": T2V_PROMPT}
    sampling_params = OmniDiffusionSamplingParams(
        height=exp["height"],
        width=exp["width"],
        generator=generator,
        guidance_scale=exp["guidance_scale"],
        num_inference_steps=exp["steps"],
        num_frames=exp["frames"],
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    result = omni.generate(prompt_dict, sampling_params)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    frames = _extract_frames(result)
    return elapsed, peak_vram, frames


def run_i2v(exp, omni, image_path):
    """Run an I2V experiment."""
    import PIL.Image

    image = PIL.Image.open(image_path).convert("RGB")
    image = image.resize((exp["width"], exp["height"]), PIL.Image.Resampling.LANCZOS)

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED)

    prompt_dict = {
        "prompt": I2V_PROMPT,
        "multi_modal_data": {"image": image},
    }
    sampling_params = OmniDiffusionSamplingParams(
        height=exp["height"],
        width=exp["width"],
        generator=generator,
        guidance_scale=exp["guidance_scale"],
        num_inference_steps=exp["steps"],
        num_frames=exp["frames"],
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    result = omni.generate(prompt_dict, sampling_params)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    frames = _extract_frames(result)
    return elapsed, peak_vram, frames


def build_omni(exp):
    """Build an Omni instance for the experiment."""
    parallel_config = DiffusionParallelConfig()
    kwargs = dict(
        model=exp["model"],
        vae_use_tiling=exp["vae_use_tiling"],
        parallel_config=parallel_config,
        enforce_eager=True,
    )
    if exp["flow_shift"] is not None:
        kwargs["flow_shift"] = exp["flow_shift"]
    if exp["quantization"] is not None:
        kwargs["quantization"] = exp["quantization"]
    return Omni(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-i2v", action="store_true", help="Skip I2V experiments")
    parser.add_argument("--i2v-image", type=str, default="test_input.jpg", help="Input image for I2V")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--experiments", type=str, nargs="*", default=None,
                        help="Run only specific experiments by name substring (e.g. '480p short' '720p')")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    # Group experiments by model to avoid reloading
    prev_model_key = None
    omni = None

    for exp in EXPERIMENTS:
        # Filter
        if exp["task"] == "i2v" and args.skip_i2v:
            print(f"SKIP: {exp['name']} (--skip-i2v)")
            continue
        if args.experiments:
            if not any(s.lower() in exp["name"].lower() for s in args.experiments):
                print(f"SKIP: {exp['name']} (not in filter)")
                continue

        model_key = (exp["model"], exp.get("quantization"))
        if model_key != prev_model_key:
            # Free previous model
            if omni is not None:
                del omni
                gc.collect()
                torch.cuda.empty_cache()
            print(f"\n{'='*60}")
            print(f"Loading model: {exp['model']} (quant={exp.get('quantization')})")
            print(f"{'='*60}")
            omni = build_omni(exp)
            prev_model_key = model_key

        print(f"\n>>> Running: {exp['name']}")
        print(f"    {exp['height']}x{exp['width']}, {exp['frames']}f, {exp['steps']}steps, "
              f"guidance={exp['guidance_scale']}, flow_shift={exp['flow_shift']}, "
              f"precision={exp['precision']}")

        try:
            if exp["task"] == "t2v":
                elapsed, peak_vram, frames = run_t2v(exp, omni)
            else:
                elapsed, peak_vram, frames = run_i2v(exp, omni, args.i2v_image)

            # Save frames for accuracy comparison
            safe_name = exp["name"].replace(" ", "_").replace("+", "_")
            frames_path = os.path.join(args.output_dir, f"vllm_{safe_name}_frames.pt")
            try:
                _save_frames(frames, frames_path)
                print(f"    Frames saved: {frames_path}")
            except Exception as e:
                print(f"    WARNING: Could not save frames: {e}")
                frames_path = None

            record = {
                "name": exp["name"],
                "framework": "vLLM-OMNI",
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
            results.append({
                "name": exp["name"],
                "framework": "vLLM-OMNI",
                "error": str(e),
            })

    # Save results
    results_path = os.path.join(args.output_dir, "bench_results_vllm_omni.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'Name':<30} {'Latency':>10} {'VRAM':>12}")
    print("-" * 55)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} {'ERROR':>10} {'':>12}")
        else:
            print(f"{r['name']:<30} {r['latency_s']:>8.1f}s {r['peak_vram_gib']:>10.2f} GiB")


if __name__ == "__main__":
    main()
