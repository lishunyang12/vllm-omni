#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Profile HunyuanVideo-1.5 on both diffusers and vLLM-OMNI with torch.profiler.

Generates Chrome trace files (.json.gz) for side-by-side comparison in Perfetto.

Usage:
    # Profile diffusers only
    python bench_profile_comparison.py --framework diffusers

    # Profile vLLM-OMNI only (uses VLLM_TORCH_PROFILER_DIR)
    python bench_profile_comparison.py --framework vllm

    # Profile both
    python bench_profile_comparison.py --framework both

Output:
    ./profile_traces/diffusers_hunyuan15.json.gz
    ./profile_traces/vllm_omni_hunyuan15_rank0.json.gz

Open both in https://ui.perfetto.dev for side-by-side comparison.
"""

import argparse
import gc
import os
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=["diffusers", "vllm", "both"], default="both")
    parser.add_argument("--model", default="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    parser.add_argument("--prompt", default="A little girl wearing a straw hat runs through a summer meadow full of wildflowers.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./profile_traces")
    return parser.parse_args()


def profile_diffusers(args):
    """Profile diffusers with torch.profiler — single process, trace captures everything."""
    import diffusers

    print("=" * 60)
    print("Profiling: diffusers")
    print("=" * 60)

    pipe = diffusers.HunyuanVideo15Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.vae.enable_tiling()

    # Set guidance via built-in guider (v1.5 API)
    if hasattr(pipe, "guider"):
        pipe.guider = pipe.guider.new(guidance_scale=args.guidance_scale)

    # Set flow_shift
    if hasattr(pipe.scheduler, "_shift"):
        pipe.scheduler._shift = args.flow_shift

    generator = torch.Generator("cuda").manual_seed(args.seed)

    # Warmup
    print("Warmup run (2 steps)...")
    with torch.no_grad():
        _ = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=2,
            generator=generator,
        )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Profile run
    trace_path = os.path.join(args.output_dir, "diffusers_hunyuan15")
    print(f"Profile run ({args.num_inference_steps} steps)...")

    generator = torch.Generator("cuda").manual_seed(args.seed)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            output = pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            )

    torch.cuda.synchronize()

    trace_file = f"{trace_path}.json.gz"
    print(f"Exporting trace to {trace_file}...")
    prof.export_chrome_trace(trace_file)

    # Print top CUDA ops
    print("\nTop 20 CUDA operations by total time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    del pipe, output
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nDiffusers trace saved: {trace_file}")
    return trace_file


def profile_vllm_omni(args):
    """Profile vLLM-OMNI using its built-in profiler (runs in worker subprocess)."""
    print("=" * 60)
    print("Profiling: vLLM-OMNI")
    print("=" * 60)

    os.environ["VLLM_TORCH_PROFILER_DIR"] = args.output_dir

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    omni = Omni(
        model=args.model,
        flow_shift=args.flow_shift,
        enforce_eager=True,
        parallel_config=DiffusionParallelConfig(),
    )

    # Start profiler
    omni.start_profile()

    # Run generation
    print(f"Profile run ({args.num_inference_steps} steps)...")
    start = time.perf_counter()
    frames = omni.generate(
        {"prompt": args.prompt},
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
        ),
    )
    elapsed = time.perf_counter() - start
    print(f"Generation: {elapsed:.2f}s")

    # Stop profiler
    print("Stopping profiler (this may take a few minutes for trace export)...")
    result = omni.stop_profile()

    trace_file = None
    if result and isinstance(result, dict):
        traces = result.get("traces", [])
        if traces:
            trace_file = traces[0]
            print(f"vLLM-OMNI trace saved: {trace_file}")

    return trace_file


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    traces = {}

    if args.framework in ("diffusers", "both"):
        traces["diffusers"] = profile_diffusers(args)

    if args.framework in ("vllm", "both"):
        traces["vllm_omni"] = profile_vllm_omni(args)

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    for name, path in traces.items():
        print(f"  {name}: {path}")
    print(f"\nOpen traces in https://ui.perfetto.dev for side-by-side comparison.")
    print("\nKey things to compare:")
    print("  1. Attention kernel time (flash_attn vs sdpa)")
    print("  2. Linear/matmul kernel time (fused QKV vs separate Q/K/V)")
    print("  3. Total denoising loop time")
    print("  4. VAE decode time")
    print("  5. Memory usage patterns")


if __name__ == "__main__":
    main()
