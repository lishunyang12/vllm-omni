# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, logger
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.utils.platform_utils import detect_device_type, is_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with Qwen-Image.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image",
        help="Diffusion model name or local path. Supported models: "
        "Qwen/Qwen-Image, Tongyi-MAI/Z-Image-Turbo, Qwen/Qwen-Image-2512",
    )
    parser.add_argument("--prompt", default="a cup of coffee on the table", help="Text prompt for image generation.")
    parser.add_argument(
        "--negative_prompt", default="", help="negative prompt for classifier-free conditional guidance."
    )
    parser.add_argument("--seed", type=int, default=142, help="Random seed for deterministic results.")
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="True classifier-free guidance scale specific to Qwen-Image.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image.")
    parser.add_argument(
        "--output",
        type=str,
        default="qwen_image_output.png",
        help="Path to save the generated image (PNG).",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps for the diffusion sampler.",
    )
    parser.add_argument(
        "--cache_backend",
        type=str,
        default=None,
        choices=["cache_dit", "tea_cache"],
        help=(
            "Cache backend to use for acceleration. "
            "Options: 'cache_dit' (DBCache + SCM + TaylorSeer), 'tea_cache' (Timestep Embedding Aware Cache). "
            "Default: None (no cache acceleration)."
        ),
    )
    parser.add_argument(
        "--ulysses_degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--ring_degree",
        type=int,
        default=1,
        help="Number of GPUs used for ring sequence parallelism.",
    )
    parser.add_argument(
        "--cfg_parallel_size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs used for classifier free guidance parallel size.",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism (TP) inside the DiT.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = detect_device_type()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Enable VAE memory optimizations on NPU
    vae_use_slicing = is_npu()
    vae_use_tiling = is_npu()

    # Configure cache based on backend type
    cache_config = None
    if args.cache_backend == "cache_dit":
        # cache-dit configuration: Hybrid DBCache + SCM + TaylorSeer
        # All parameters marked with [cache-dit only] in DiffusionCacheConfig
        cache_config = {
            # DBCache parameters [cache-dit only]
            "Fn_compute_blocks": 1,  # Optimized for single-transformer models
            "Bn_compute_blocks": 0,  # Number of backward compute blocks
            "max_warmup_steps": 4,  # Maximum warmup steps (works for few-step models)
            "residual_diff_threshold": 0.24,  # Higher threshold for more aggressive caching
            "max_continuous_cached_steps": 3,  # Limit to prevent precision degradation
            # TaylorSeer parameters [cache-dit only]
            "enable_taylorseer": False,  # Disabled by default (not suitable for few-step models)
            "taylorseer_order": 1,  # TaylorSeer polynomial order
            # SCM (Step Computation Masking) parameters [cache-dit only]
            "scm_steps_mask_policy": None,  # SCM mask policy: None (disabled), "slow", "medium", "fast", "ultra"
            "scm_steps_policy": "dynamic",  # SCM steps policy: "dynamic" or "static"
        }
    elif args.cache_backend == "tea_cache":
        # TeaCache configuration
        # All parameters marked with [tea_cache only] in DiffusionCacheConfig
        cache_config = {
            # TeaCache parameters [tea_cache only]
            "rel_l1_thresh": 0.2,  # Threshold for accumulated relative L1 distance
            # Note: coefficients will use model-specific defaults based on model_type
            #        (e.g., QwenImagePipeline or FluxPipeline)
        }

    # assert args.ring_degree == 1, "Ring attention is not supported yet"
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Check if profiling is requested via environment variable
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))
    omni = None

    # Time profiling for generation
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Cache backend: {args.cache_backend if args.cache_backend else 'None (no acceleration)'}")
    print(f"  Image size: {args.width}x{args.height}")
    print(
        f"  Parallel configuration: ulysses_degree={args.ulysses_degree}, ring_degree={args.ring_degree}, "
        f"cfg_parallel_size={args.cfg_parallel_size}, tensor_parallel_size={args.tensor_parallel_size}"
    )
    print(f"{'=' * 60}\n")

    # Initialize Omni with appropriate pipeline
    omni = Omni(
        model=args.model,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
    )
    print("Pipeline loaded")

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    generation_start = time.perf_counter()
    # Generate image
    generate_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "generator": generator,
        "true_cfg_scale": args.cfg_scale,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "num_outputs_per_prompt": args.num_images_per_prompt,
        "height": args.height,
        "width": args.width,
        "layers": args.layers,
        "resolution": args.resolution,
    }
    outputs = omni.generate(**generate_kwargs)
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    if profiler_enabled:
        print("\n[Profiler] Stopping profiler and collecting results...")
        profile_results = omni.stop_profile()
        if profile_results and isinstance(profile_results, dict):
            traces = profile_results.get("traces", [])
            tables = profile_results.get("tables", [])
            print("\n" + "=" * 60)
            print("PROFILING RESULTS:")
            for rank in range(max(len(traces), len(tables))):
                print(f"\nRank {rank}:")
                if rank < len(traces) and traces[rank]:
                    print(f" • Trace: {traces[rank]}")
                if rank < len(tables) and tables[rank]:
                    print(f" • Table: {tables[rank]}")
            print("=" * 60)
        else:
            print("[Profiler] No valid profiling data returned.")

    if not outputs:
        raise ValueError("No output generated from omni.generate()")
    logger.info("Outputs: %s", outputs)

    # Extract images from OmniRequestOutput
    first_output = outputs[0]
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output[0]
    # Check if this is a request output with images
    # Supports both direct access and namespace access if needed
    images = getattr(req_out, "images", None)
    if not images:
        raise ValueError("No images found in request_output")

    # Save output image(s)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "qwen_image_output"

    if args.num_images_per_prompt <= 1:
        img = images[0]
        # Check if this is a layered output (list of images)
        if isinstance(img, list):
            for sub_idx, sub_img in enumerate(img):
                save_path = output_path.parent / f"{stem}_{sub_idx}{suffix}"
                sub_img.save(save_path)
                print(f"Saved edited image to {os.path.abspath(save_path)}")
        else:
            img.save(output_path)
            print(f"Saved edited image to {os.path.abspath(output_path)}")
    else:
        for idx, img in enumerate(images):
            if isinstance(img, list):
                for sub_idx, sub_img in enumerate(img):
                    save_path = output_path.parent / f"{stem}_{idx}_{sub_idx}{suffix}"
                    sub_img.save(save_path)
                    print(f"Saved edited image to {os.path.abspath(save_path)}")
            else:
                save_path = output_path.parent / f"{stem}_{idx}{suffix}"
                img.save(save_path)
                print(f"Saved edited image to {os.path.abspath(save_path)}")

    # Explicitly close omni
    if omni is not None:
        print("\nCleaning up Omni instance...")
        omni.close()
        print("Cleanup completed.")


if __name__ == "__main__":
    main()
