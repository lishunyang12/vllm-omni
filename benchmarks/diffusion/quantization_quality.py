# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark quantization quality loss for diffusion models (image & video).

Generates outputs with BF16 (baseline) and a quantized config using the same
seed, then computes LPIPS perceptual distance between them. Results are printed
as a Markdown table ready to paste into a PR description.

Requirements:
    pip install lpips Pillow numpy

Image example (text-to-image):
    python benchmarks/diffusion/quantization_quality.py \
        --model Tongyi-MAI/Z-Image-Turbo \
        --task t2i \
        --quantization fp8 \
        --prompts \
            "an aerial view of a coral reef with crystal clear turquoise water" \
            "a campfire in a dark forest with sparks rising into a starry sky" \
            "a gourmet dessert plate with chocolate mousse and gold leaf" \
        --height 1024 --width 1024 \
        --num-inference-steps 50 --seed 42

Video example (text-to-video):
    python benchmarks/diffusion/quantization_quality.py \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --task t2v \
        --quantization fp8 \
        --prompts \
            "A serene lakeside sunrise with mist over the water" \
            "A cat walking across a wooden bridge in autumn" \
        --height 720 --width 1280 \
        --num-frames 81 --num-inference-steps 40 --seed 42

Video example (text-to-video) with offline quant:
    python benchmarks/diffusion/quantization_quality.py \
        --use-offline-quant \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --model-quant-checkpoint /vllm-omni/wan22-t2v-modelopt-fp8\
        --task t2v \
        --quantization fp8 \
        --prompts \
            "A serene lakeside sunrise with mist over the water" \
            "A cat walking across a wooden bridge in autumn" \
        --height 720 --width 1280 \
        --num-frames 81 --num-inference-steps 40 --seed 42

Video example (image-to-video) with offline quant:
    python benchmarks/diffusion/quantization_quality.py \
        --use-offline-quant \
        --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
        --model-quant-checkpoint /vllm-omni/wan22-i2v-modelopt-fp8\
        --task i2v \
        --quantization fp8 \
        --prompts \
            "An astronaut riding a horse across the surface of Mars, red dust swirling, cinematic wide shot." \
            "A skateboarder doing a kickflip in an urban plaza, slow motion, golden hour lighting." \
        --height 720 --width 1280 \
        --image /path/to/ref_images/ \
        --num-frames 81 --num-inference-steps 40 --seed 42 --vae-use-tiling

Multiple quantization methods:
    python benchmarks/diffusion/quantization_quality.py \
        --model Tongyi-MAI/Z-Image-Turbo \
        --task t2i \
        --quantization fp8 int8 bitsandbytes \
        --prompts "a cup of coffee on the table" \
        --height 1024 --width 1024 \
        --num-inference-steps 50 --seed 42

Output directory structure (--output-dir, default: ./quant_bench_output):
    quant_bench_output/
        baseline/           # BF16 outputs
        <method>/           # Quantized outputs per method
        results.md          # Markdown table
"""

import argparse
import gc
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch


def compute_lpips_images(
    baseline_images: list,
    quantized_images: list,
    net: str = "alex",
) -> list[float]:
    """Compute LPIPS between paired lists of PIL images."""
    import lpips
    from torchvision import transforms

    loss_fn = lpips.LPIPS(net=net).eval()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    scores = []
    for img_bl, img_qt in zip(baseline_images, quantized_images):
        t_bl = transform(img_bl.convert("RGB")).unsqueeze(0)
        t_qt = transform(img_qt.convert("RGB")).unsqueeze(0)
        if torch.cuda.is_available():
            t_bl, t_qt = t_bl.cuda(), t_qt.cuda()
        with torch.no_grad():
            score = loss_fn(t_bl, t_qt).item()
        scores.append(score)
    return scores


def compute_lpips_video(
    baseline_frames: np.ndarray,
    quantized_frames: np.ndarray,
    net: str = "alex",
) -> float:
    """Compute mean per-frame LPIPS for a video pair.

    Args:
        baseline_frames: (F, H, W, C) float array in [0, 1].
        quantized_frames: same shape.

    Returns:
        Mean LPIPS across all frames.
    """
    import lpips

    loss_fn = lpips.LPIPS(net=net).eval()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    num_frames = min(len(baseline_frames), len(quantized_frames))
    scores = []
    for i in range(num_frames):
        # Convert (H, W, C) float [0,1] -> (1, C, H, W) float [-1, 1]
        f_bl = torch.from_numpy(baseline_frames[i]).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        f_qt = torch.from_numpy(quantized_frames[i]).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        if torch.cuda.is_available():
            f_bl, f_qt = f_bl.cuda(), f_qt.cuda()
        with torch.no_grad():
            score = loss_fn(f_bl, f_qt).item()
        scores.append(score)
    return float(np.mean(scores))


def _build_omni_kwargs(args, quantization=None):
    """Build kwargs dict for Omni() constructor."""
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if quantization:
        kwargs = {
            "model": args.model_quant_checkpoint if args.use_offline_quant else args.model,
            "parallel_config": parallel_config,
            "enforce_eager": args.enforce_eager,
            "quantization_config":quantization,
            "vae_use_tiling":args.vae_use_tiling,
            "enable_diffusion_pipeline_profiler": True,
        }
    else:
        kwargs = {
            "model": args.model,
            "parallel_config": parallel_config,
            "enforce_eager": args.enforce_eager,
            "vae_use_tiling":args.vae_use_tiling,
            "enable_diffusion_pipeline_profiler": True,
        }

    return kwargs

def _load_reference_images(spec: str | None) -> list[Any]:
    """Load PIL.Image list from a directory or a single file path."""
    if spec is None:
        return []
    from PIL import Image

    p = Path(spec).expanduser()
    if not p.exists():
        raise SystemExit(f"--reference-images path not found: {p}")
    if p.is_file():
        return [Image.open(p).convert("RGB")]
    image_paths = sorted(
        f for f in p.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if not image_paths:
        raise SystemExit(f"No image files (jpg/jpeg/png/webp) found in {p}")
    return [Image.open(f).convert("RGB") for f in image_paths]

def _extract_denoise_seconds(first) -> float | None:
    """Pull denoise-only time out of OmniRequestOutput.stage_durations.

    The pipeline profiler records keys like '<PipelineClass>.diffuse'
    (cuda-synced timer wrapped around the denoising loop). Match by suffix so
    this stays model-agnostic.
    """
    stage_durations = getattr(first, "stage_durations", {}) or {}
    return next(
        (v for k, v in stage_durations.items() if k.endswith(".diffuse")),
        None,
    )


def _generate_image(omni, args, prompt, seed):
    """Generate a single image and return (PIL.Image, wall_seconds, memory_gib, denoise_seconds)."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs = omni.generate(
        {"prompt": prompt},
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
        ),
    )
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    first = outputs[0]
    peak_mem = getattr(first, "peak_memory_mb", 0.0) / 1024  # MB -> GiB
    denoise_seconds = _extract_denoise_seconds(first)
    req_out = first.request_output if hasattr(first, "request_output") else first
    img = req_out.images[0]
    return img, elapsed, peak_mem, denoise_seconds


def _generate_video(omni, args, prompt, seed,image=None):
    """Generate a video and return (np.ndarray [F,H,W,C], wall_seconds, memory_gib, denoise_seconds)."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.platforms import current_omni_platform

    request = {"prompt": prompt, "negative_prompt": args.negative_prompt}
    if image is not None:
        request["multi_modal_data"] = {"image": image}
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs = omni.generate(
        request,
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

    first = outputs[0]
    peak_mem = getattr(first, "peak_memory_mb", 0.0) / 1024  # MB -> GiB
    denoise_seconds = _extract_denoise_seconds(first)
    if hasattr(first, "request_output") and isinstance(first.request_output, list):
        inner = first.request_output[0]
        if isinstance(inner, OmniRequestOutput) and hasattr(inner, "images"):
            frames = inner.images[0] if inner.images else None
        else:
            frames = inner
    elif hasattr(first, "images") and first.images:
        frames = first.images
    else:
        raise ValueError("Could not extract video frames from output.")

    if isinstance(frames, torch.Tensor):
        video = frames.detach().cpu()
        if video.dim() == 5:
            video = video[0].permute(1, 2, 3, 0) if video.shape[1] in (3, 4) else video[0]
        elif video.dim() == 4 and video.shape[0] in (3, 4):
            video = video.permute(1, 2, 3, 0)
        if video.is_floating_point():
            video = video.clamp(-1, 1) * 0.5 + 0.5
        frames_array = video.float().numpy()
    else:
        frames_array = np.asarray(frames)
        if frames_array.ndim ==6:# wan2.2: inner.images = [ndarray[1,F,H,W,C]]
            frames_array = frames_array[0]
        if frames_array.ndim == 5:
            frames_array = frames_array[0]

    return frames_array, elapsed, peak_mem, denoise_seconds


def _unload_omni(omni):
    """Delete Omni instance and free GPU memory."""
    del omni
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_benchmark(args):
    from vllm_omni.entrypoints.omni import Omni

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_video = args.task in ("t2v", "i2v", "ti2v")
    is_image_conditioned = args.task == "i2v" or (args.task == "ti2v" and args.images is not None)
    prompts = args.prompts
    seed = args.seed
    input_images = _load_reference_images(args.images) if is_image_conditioned else None

    # Determine configs to benchmark
    configs = []  # list of (label, quantization_method)
    for method in args.quantization:
        configs.append((method, method))

    # --- Baseline run ---
    print("\n" + "=" * 60)
    print("Running BF16 baseline...")
    print("=" * 60)
    bl_kwargs = _build_omni_kwargs(args, quantization=None)
    omni_bl = Omni(**bl_kwargs)

    baseline_outputs = {}  # prompt -> (output, wall_time, mem, denoise_seconds)
    for i,prompt in enumerate(prompts):
        print(f"  Generating: {prompt[:60]}...")
        if is_video and input_images is not None:
            out, t, mem, dt = _generate_video(omni_bl, args, prompt, seed,input_images[i % len(input_images)])
        elif is_video:
            out, t, mem, dt = _generate_video(omni_bl, args, prompt, seed)
        else:
            out, t, mem, dt = _generate_image(omni_bl, args, prompt, seed)
        baseline_outputs[prompt] = (out, t, mem, dt)

    bl_avg_time = np.mean([v[1] for v in baseline_outputs.values()])
    bl_denoise_times = [v[3] for v in baseline_outputs.values() if v[3] is not None]
    bl_avg_denoise = float(np.mean(bl_denoise_times)) if bl_denoise_times else None
    bl_mem = baseline_outputs[prompts[0]][2]  # use first prompt's memory
    _unload_omni(omni_bl)
    del omni_bl # must explicitly del omni_bl otherwise will cause oom when running next model

    # Save baseline outputs
    bl_dir = output_dir / "baseline"
    bl_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(prompts):
        out = baseline_outputs[prompt][0]
        if is_video:
            try:
                from diffusers.utils import export_to_video

                frames_list = list(out) if isinstance(out, np.ndarray) and out.ndim == 4 else out
                export_to_video(frames_list, str(bl_dir / f"prompt_{i}.mp4"), fps=args.fps)
            except ImportError:
                np.save(bl_dir / f"prompt_{i}.npy", out)
        else:
            out.save(bl_dir / f"prompt_{i}.png")

    # --- Quantized runs ---
    all_results = []  # list of dicts

    for config_label, quant_method in configs:
        print(f"\n{'=' * 60}")
        print(f"Running: {config_label}...")
        print("=" * 60)

        qt_kwargs = _build_omni_kwargs(args, quantization=quant_method)
        omni_qt = Omni(**qt_kwargs)

        qt_outputs = {}
        for i,prompt in enumerate(prompts):
            print(f"  Generating: {prompt[:60]}...")
            if is_video and input_images is not None:
                out, t, mem, dt = _generate_video(omni_qt, args, prompt, seed,input_images[i % len(input_images)])
            elif is_video:
                out, t, mem, dt = _generate_video(omni_qt, args, prompt, seed)
            else:
                out, t, mem, dt = _generate_image(omni_qt, args, prompt, seed)
            qt_outputs[prompt] = (out, t, mem, dt)

        qt_avg_time = np.mean([v[1] for v in qt_outputs.values()])
        qt_denoise_times = [v[3] for v in qt_outputs.values() if v[3] is not None]
        qt_avg_denoise = float(np.mean(qt_denoise_times)) if qt_denoise_times else None
        qt_mem = qt_outputs[prompts[0]][2]
        _unload_omni(omni_qt)
        del omni_qt  # must explicitly del omni_qt otherwise will cause oom when running next model
        # Save quantized outputs
        qt_dir = output_dir / config_label.replace(" ", "_")
        qt_dir.mkdir(parents=True, exist_ok=True)

        # Compute LPIPS per prompt
        per_prompt = []
        for i, prompt in enumerate(prompts):
            bl_out = baseline_outputs[prompt][0]
            qt_out = qt_outputs[prompt][0]
            if is_video:
                lpips_score = compute_lpips_video(bl_out, qt_out, net=args.lpips_net)
                try:
                    from diffusers.utils import export_to_video

                    frames_list = list(qt_out) if isinstance(qt_out, np.ndarray) and qt_out.ndim == 4 else qt_out
                    export_to_video(frames_list, str(qt_dir / f"prompt_{i}.mp4"), fps=args.fps)
                except ImportError:
                    np.save(qt_dir / f"prompt_{i}.npy", qt_out)
            else:
                lpips_score = compute_lpips_images([bl_out], [qt_out], net=args.lpips_net)[0]
                qt_out.save(qt_dir / f"prompt_{i}.png")
            per_prompt.append({"prompt": prompt, "lpips": lpips_score})

        mean_lpips = np.mean([p["lpips"] for p in per_prompt])
        speedup = bl_avg_time / qt_avg_time if qt_avg_time > 0 else float("inf")
        mem_reduction = (bl_mem - qt_mem) / bl_mem * 100
        # Throughput uses denoise-only time (cuda-synced via DiffusionPipelineProfiler);
        # falls back to wall time only if the profiler didn't surface it.
        qt_throughput_basis = qt_avg_denoise if qt_avg_denoise is not None else qt_avg_time
        qt_throughput = qt_throughput_basis / args.num_inference_steps if args.num_inference_steps > 0 else float("inf")

        all_results.append(
            {
                "config": config_label,
                "avg_time": qt_avg_time,
                "avg_denoise": qt_avg_denoise,
                "speedup": speedup,
                "throughput_its": qt_throughput,
                "memory_gib": qt_mem,
                "mem_reduction_pct": mem_reduction,
                "mean_lpips": mean_lpips,
                "per_prompt": per_prompt,
            }
        )

    # --- Print results ---
    print("\n\n")
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Summary table
    model = args.model_quant_checkpoint if args.use_offline_quant else args.model
    lines = []
    lines.append(f"## Quantization Quality Benchmark — {model.split('/')[-1]}")
    lines.append(
        f"Setup: {args.height}x{args.width}, {args.num_inference_steps} steps, "
        f"seed={args.seed}, LPIPS ({args.lpips_net})"
    )
    if is_video:
        lines.append(f"Video: {args.num_frames} frames")
    lines.append("")
    lines.append("### Summary")
    lines.append("")
    # Throughput uses denoise-only time (cuda-synced via DiffusionPipelineProfiler);
    # falls back to wall time only if the profiler didn't surface it.
    bl_throughput_basis = bl_avg_denoise if bl_avg_denoise is not None else bl_avg_time
    bl_throughput = bl_throughput_basis / args.num_inference_steps if args.num_inference_steps > 0 else float("inf")
    lines.append(
        "| Config | Avg Time | Speedup | Throughput (s/it) "
        "| Peak VRAM (GiB) | Peak VRAM Reduction | Mean LPIPS |"
    )
    lines.append(
        "|--------|----------|---------|--------------------"
        "|-----------------|---------------------|------------|"
    )
    lines.append(
        f"| BF16 baseline | {bl_avg_time:.2f}s | 1.00x | {bl_throughput:.3f} "
        f"| {bl_mem:.2f} | — | (ref) |"
    )
    for r in all_results:
        lines.append(
            f"| {r['config']} | {r['avg_time']:.2f}s | {r['speedup']:.2f}x "
            f"| {r['throughput_its']:.3f} "
            f"| {r['memory_gib']:.2f} | {r['mem_reduction_pct']:.0f}% "
            f"| {r['mean_lpips']:.4f} |"
        )
    lines.append("")
    lines.append("> LPIPS < 0.01 = imperceptible, > 0.1 = clearly noticeable.")
    lines.append(
        "> Throughput (s/it) = denoise time / num_inference_steps "
        "(cuda-synced via DiffusionPipelineProfiler; excludes text encode + VAE decode)."
    )
    lines.append(
        "> Peak VRAM = `max_memory_allocated` during one generate (model + activations + latents + VAE buffers)."
    )
    lines.append(
        "> For model resident size on disk/VRAM, run `examples/quantization/check_modelopt_fp8_export.py "
        "--output <ckpt> --baseline <hf-id-or-path>`."
    )
    lines.append("")

    # Per-prompt table
    if len(prompts) > 1:
        lines.append("### Per-Prompt LPIPS")
        lines.append("")
        header = "| Prompt |"
        sep = "|--------|"
        for r in all_results:
            header += f" {r['config']} |"
            sep += "--------|"
        lines.append(header)
        lines.append(sep)
        for i, prompt in enumerate(prompts):
            short = prompt[:50] + "..." if len(prompt) > 50 else prompt
            row = f"| {short} |"
            for r in all_results:
                row += f" {r['per_prompt'][i]['lpips']:.4f} |"
            lines.append(row)
        lines.append("")

    md = "\n".join(lines)
    print(md)

    # Save markdown
    results_path = output_dir / "results.md"
    results_path.write_text(md, encoding="utf-8")
    print(f"\nResults saved to {results_path}")
    print(f"Baseline outputs in {bl_dir}")
    for r in all_results:
        qt_dir = output_dir / r["config"].replace(" ", "_")
        print(f"Quantized outputs in {qt_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark quantization quality loss for diffusion models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model name or local path.")
    parser.add_argument("--model-quant-checkpoint",default=None, help="Offline quantization model checkpoint")
    parser.add_argument("--use-offline-quant",action="store_true",help="compare with offline quantized model checkpoint")
    parser.add_argument(
        "--task",
        default="t2i",
        choices=["t2i", "t2v", "i2v", "ti2v"],
        help="Task type: t2i (text-to-image), t2v (text-to-video), i2v / ti2v (image-to-video).",
    )
    parser.add_argument(
        "--quantization",
        nargs="+",
        required=True,
        help="One or more quantization methods to benchmark (e.g. fp8 int8 bitsandbytes).",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["a cup of coffee on the table"],
        help="One or more prompts to generate.",
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Path to input images (required for i2v and ti2v tasks).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=81, help="Number of video frames (t2v only).")
    parser.add_argument("--fps", type=int, default=24, help="Video FPS for saving (t2v only).")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG scale (used for video).")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt for video generation. Wan2.2 I2V degenerates to a static frame "
             "without the official anti-static negative prompt; other pipelines work with empty.",
    )
    parser.add_argument("--output-dir", type=str, default="./quant_bench_output", help="Directory to save outputs.")
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone network.",
    )
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--vae-use-tiling",
        action="store_true",
        help="Enable VAE tiling for memory optimization.Specifically for bf16 model",
    )
    args = parser.parse_args()
    if args.task in ("i2v") and args.images is None:
        parser.error(f"--task {args.task} requires --images")
    if args.use_offline_quant and not args.model_quant_checkpoint:
        parser.error("--use-offline-quant requires --model-quant-checkpoint")
    return args


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
