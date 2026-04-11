# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate one image with FLUX.2-klein-4b-nvfp4 end-to-end.

Run on a Blackwell (or emulation-capable) GPU box:

    python tests/diffusion/quantization/gen_nvfp4_flux2.py \\
        --prompt "a photo of an astronaut riding a horse on Mars" \\
        --output /tmp/out.png

This script reuses the same TP / VllmConfig / forward-context boilerplate
as smoke_nvfp4_flux2.py but actually drives the full Flux2KleinPipeline
forward — text encode -> denoise -> VAE decode -> PIL — and saves the
result to disk. Intended for manual quality verification on a real GPU,
not for CI.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _make_minimal_vllm_config():
    from vllm.config import VllmConfig
    from vllm.config.parallel import ParallelConfig

    return VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    )


def _init_distributed_only() -> None:
    import torch.distributed as dist
    from vllm.distributed import init_distributed_environment

    if dist.is_available() and dist.is_initialized():
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
        backend="nccl",
    )


def _save_pil_image(pil_or_list, path: str) -> str:
    """Accept a PIL image, a list of PIL images, or a tensor and write to disk."""
    import PIL.Image

    img = pil_or_list[0] if isinstance(pil_or_list, list) else pil_or_list
    if not isinstance(img, PIL.Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(img).__name__}")
    img.save(path)
    return path


def _run(args: argparse.Namespace) -> int:
    import torch
    from vllm.config import LoadConfig, set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized
    from vllm.transformers_utils.config import get_hf_file_to_dict

    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        initialize_model_parallel as initialize_diffusion_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.model_loader.diffusers_loader import (
        DiffusersPipelineLoader,
    )
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    vllm_config = _make_minimal_vllm_config()
    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        # vllm-omni's diffusion side maintains its OWN parallel groups
        # (CFG / SP / VAE PP / data) on top of vLLM's TP/PP. Pipelines like
        # Flux2KleinPipeline.predict_noise_maybe_with_cfg() reach for the
        # CFG group at forward time, so we have to create singleton groups
        # for all the dimensions or that path asserts.
        initialize_diffusion_model_parallel(
            data_parallel_size=1,
            cfg_parallel_size=1,
            sequence_parallel_size=1,
            ulysses_degree=1,
            ring_degree=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )

        od = OmniDiffusionConfig(
            model=args.base,
            model_class_name="Flux2KleinPipeline",
            transformer_weights_path=args.nvfp4,
        )
        tf_cfg = get_hf_file_to_dict("transformer/config.json", od.model)
        if tf_cfg is None:
            print(f"FAIL: could not fetch transformer/config.json from {od.model}", file=sys.stderr)
            return 1
        od.tf_model_config = TransformerConfig.from_dict(tf_cfg)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = DiffusersPipelineLoader(load_config=LoadConfig(), od_config=od)
        load_t0 = time.perf_counter()
        pipe = loader.load_model(
            od_config=od,
            load_device=device.type,
            load_format="default",
            device=device,
        )
        print(f"[load] pipeline ready in {time.perf_counter() - load_t0:.1f}s")

        req = OmniDiffusionRequest(
            prompts=[args.prompt],
            sampling_params=OmniDiffusionSamplingParams(
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance,
                seed=args.seed,
            ),
            request_ids=["smoke-gen-0"],
        )

        gen_t0 = time.perf_counter()
        with set_forward_context(omni_diffusion_config=od):
            out = pipe.forward(req)
        print(f"[gen]  forward done in {time.perf_counter() - gen_t0:.1f}s")

        if out.error is not None:
            print(f"FAIL: pipeline error: {out.error}", file=sys.stderr)
            return 2
        if out.output is None:
            print("FAIL: pipeline returned no output", file=sys.stderr)
            return 3

        # The diffusion pipeline returns a tensor; the post-processing func
        # converts it to PIL. Some pipelines run post-process automatically;
        # if .output is already a list of PIL images, just save it.
        result = out.output
        if out.post_process_func is not None and not isinstance(result, list):
            result = out.post_process_func(result)
        path = _save_pil_image(result, args.output)
        print(f"[save] wrote {path}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--nvfp4", default="black-forest-labs/FLUX.2-klein-4b-nvfp4")
    parser.add_argument("--prompt", default="A photo of a capybara wearing sunglasses, riding a skateboard")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/flux2_klein_4b_nvfp4.png")
    args = parser.parse_args()

    _init_distributed_only()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
