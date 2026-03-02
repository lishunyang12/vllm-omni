# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import os
import time
import uuid
from pathlib import Path

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

USE_V1 = os.getenv("VLLM_OMNI_USE_V1") == "1"

if USE_V1:
    from vllm_omni.entrypoints.async_omni_v1 import AsyncOmniV1 as AsyncOmniCls
else:
    from vllm_omni.entrypoints.async_omni import AsyncOmni as AsyncOmniCls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with a diffusion model (async).")
    parser.add_argument(
        "--model",
        default="riverclouds/qwen_image_random",
        help="Diffusion model name or local path.",
    )
    parser.add_argument("--prompt", default="a cup of coffee on the table", help="Text prompt for image generation.")
    parser.add_argument(
        "--negative_prompt",
        default=None,
        help="Negative prompt for classifier-free conditional guidance.",
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
        default="output.png",
        help="Path to save the generated image.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--ulysses_degree",
        type=int,
        default=1,
        help="Ulysses sequence parallel degree.",
    )
    parser.add_argument(
        "--ring_degree",
        type=int,
        default=1,
        help="Ring attention degree.",
    )
    parser.add_argument(
        "--cfg_parallel_size",
        type=int,
        default=1,
        help="CFG parallel size.",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offloading.",
    )
    parser.add_argument(
        "--enable-layerwise-offload",
        action="store_true",
        help="Enable layerwise offloading on DiT modules.",
    )
    parser.add_argument(
        "--layerwise-num-gpu-layers",
        type=int,
        default=1,
        help="Number of layers to keep on GPU during layerwise offload.",
    )
    parser.add_argument(
        "--vae_use_slicing",
        action="store_true",
        help="Enable VAE slicing.",
    )
    parser.add_argument(
        "--vae_use_tiling",
        action="store_true",
        help="Enable VAE tiling.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    engine_kwargs = dict(
        enable_layerwise_offload=args.enable_layerwise_offload,
        layerwise_num_gpu_layers=args.layerwise_num_gpu_layers,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
    )

    engine = AsyncOmniCls(model=args.model, **engine_kwargs)

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        generator=generator,
        true_cfg_scale=args.cfg_scale,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=args.num_images_per_prompt,
    )

    prompt = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
    }

    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model:            {args.model}")
    print(f"  Prompt:           {args.prompt}")
    print(f"  Inference steps:  {args.num_inference_steps}")
    print(f"  Image size:       {args.width}x{args.height}")
    print(f"{'=' * 60}\n")

    request_id = str(uuid.uuid4())
    generation_start = time.perf_counter()
    output = None
    async for output in engine.generate(prompt, request_id, sampling_params_list=[sampling_params]):
        pass
    generation_end = time.perf_counter()

    print(f"Total generation time: {generation_end - generation_start:.4f}s")

    images = output.images
    if not images:
        raise ValueError("No images returned.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem
    suffix = output_path.suffix or ".png"

    if len(images) == 1:
        images[0].save(output_path)
        print(f"Saved image to {output_path}")
    else:
        for idx, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"
            img.save(save_path)
            print(f"Saved image to {save_path}")


if __name__ == "__main__":
    asyncio.run(main())
