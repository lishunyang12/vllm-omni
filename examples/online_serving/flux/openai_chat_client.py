#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FLUX.1-schnell OpenAI-compatible image generation client.

Usage:
    python openai_chat_client.py --prompt "A beautiful landscape" --output output.png
    python openai_chat_client.py --prompt "A sunset" --height 1024 --width 1024 --steps 4 --seed 42
"""

import argparse
import base64
from pathlib import Path

import requests


def generate_image(
    prompt: str,
    server_url: str = "http://localhost:8091",
    height: int | None = None,
    width: int | None = None,
    steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    num_outputs_per_prompt: int = 1,
) -> bytes | None:
    """Generate an image using the images generation API.

    Args:
        prompt: Text description of the image
        server_url: Server URL
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed
        negative_prompt: Negative prompt
        num_outputs_per_prompt: Number of images to generate

    Returns:
        Image bytes or None if failed
    """
    payload: dict[str, object] = {
        "prompt": prompt,
        "response_format": "b64_json",
        "n": num_outputs_per_prompt,
    }

    if width is not None and height is not None:
        payload["size"] = f"{width}x{height}"
    elif width is not None:
        payload["size"] = f"{width}x{width}"
    elif height is not None:
        payload["size"] = f"{height}x{height}"

    if steps is not None:
        payload["num_inference_steps"] = steps
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["seed"] = seed

    try:
        response = requests.post(
            f"{server_url}/v1/images/generations",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        items = data.get("data")
        if isinstance(items, list) and items:
            first = items[0].get("b64_json") if isinstance(items[0], dict) else None
            if isinstance(first, str):
                return base64.b64decode(first)

        print(f"Unexpected response format: {data}")
        return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="FLUX.1-schnell image generation client")
    parser.add_argument("--prompt", "-p", default="a cup of coffee on the table", help="Text prompt")
    parser.add_argument("--output", "-o", default="flux_output.png", help="Output file")
    parser.add_argument("--server", "-s", default="http://localhost:8091", help="Server URL")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (default 4 for schnell)")
    parser.add_argument("--guidance-scale", type=float, default=0.0, help="Guidance scale (default 0 for schnell)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--negative", help="Negative prompt")

    args = parser.parse_args()

    print(f"Generating image for: {args.prompt}")

    image_bytes = generate_image(
        prompt=args.prompt,
        server_url=args.server,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        negative_prompt=args.negative,
    )

    if image_bytes:
        output_path = Path(args.output)
        output_path.write_bytes(image_bytes)
        print(f"Image saved to: {output_path}")
        print(f"Size: {len(image_bytes) / 1024:.1f} KB")
    else:
        print("Failed to generate image")
        exit(1)


if __name__ == "__main__":
    main()
