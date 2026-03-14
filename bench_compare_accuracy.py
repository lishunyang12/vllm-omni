# SPDX-License-Identifier: Apache-2.0
"""
Compare frames from vLLM-OMNI and diffusers benchmarks.

Computes PSNR and SSIM for each matching experiment pair.

Usage:
    python bench_compare_accuracy.py [--output-dir bench_outputs]

Requires: pip install scikit-image
"""

import argparse
import json
import os

import numpy as np
import torch


def compute_psnr(a, b, data_range=255.0):
    """Compute PSNR between two arrays."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(data_range**2 / mse)


def compute_ssim(a, b, data_range=255):
    """Compute SSIM. Falls back to manual if skimage unavailable."""
    try:
        from skimage.metrics import structural_similarity
        # Average over frames
        ssims = []
        for i in range(min(len(a), len(b))):
            s = structural_similarity(a[i], b[i], channel_axis=-1, data_range=data_range)
            ssims.append(s)
        return float(np.mean(ssims))
    except ImportError:
        print("WARNING: scikit-image not installed, skipping SSIM")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="bench_outputs")
    args = parser.parse_args()

    # Find matching pairs
    pairs = [
        ("T2V_480p_short_BF16", "T2V 480p short BF16"),
        ("T2V_480p_full_BF16", "T2V 480p full BF16"),
        ("T2V_720p_short_FP8_tiling", "T2V 720p short FP8+tiling"),
        ("I2V_480p_short_BF16", "I2V 480p short BF16"),
    ]

    results = []
    print(f"\n{'Test':<30} {'PSNR (dB)':>12} {'SSIM':>10} {'Match Frames':>14}")
    print("-" * 70)

    for safe_name, display_name in pairs:
        vllm_path = os.path.join(args.output_dir, f"vllm_{safe_name}_frames.pt")
        diff_path = os.path.join(args.output_dir, f"diffusers_{safe_name}_frames.pt")

        if not os.path.exists(vllm_path):
            print(f"{display_name:<30} {'N/A (no vllm)':>12}")
            continue
        if not os.path.exists(diff_path):
            print(f"{display_name:<30} {'N/A (no diffusers)':>12}")
            continue

        vllm_frames = torch.load(vllm_path, weights_only=True).numpy()
        diff_frames = torch.load(diff_path, weights_only=True).numpy()

        print(f"  vllm shape: {vllm_frames.shape}, dtype: {vllm_frames.dtype}")
        print(f"  diff shape: {diff_frames.shape}, dtype: {diff_frames.dtype}")

        # Handle different layouts:
        # diffusers typically saves (F, H, W, C) uint8
        # vllm-omni may save (F, C, H, W) or (F, H, W, C)
        def normalize_layout(arr):
            """Ensure frames are (F, H, W, C) layout."""
            if arr.ndim == 5:
                # (B, F, ...) -> (F, ...)
                arr = arr[0]
            if arr.ndim == 4:
                # Check if channel-first: (F, C, H, W) where C is 3 or 4
                if arr.shape[1] in (3, 4) and arr.shape[1] < arr.shape[2]:
                    arr = np.transpose(arr, (0, 2, 3, 1))
            if arr.ndim == 3:
                # Single frame (H, W, C)
                arr = arr[np.newaxis]
            return arr

        vllm_frames = normalize_layout(vllm_frames)
        diff_frames = normalize_layout(diff_frames)

        # Ensure same frame count
        n = min(len(vllm_frames), len(diff_frames))
        vllm_frames = vllm_frames[:n]
        diff_frames = diff_frames[:n]

        # Normalize to uint8 range
        if vllm_frames.dtype in (np.float32, np.float64, np.float16):
            if vllm_frames.max() <= 1.0:
                vllm_frames = (vllm_frames * 255).clip(0, 255).astype(np.uint8)
            else:
                vllm_frames = vllm_frames.clip(0, 255).astype(np.uint8)
        if diff_frames.dtype in (np.float32, np.float64, np.float16):
            if diff_frames.max() <= 1.0:
                diff_frames = (diff_frames * 255).clip(0, 255).astype(np.uint8)
            else:
                diff_frames = diff_frames.clip(0, 255).astype(np.uint8)

        # Resize if dimensions don't match
        if vllm_frames.shape[1:3] != diff_frames.shape[1:3]:
            print(f"  WARNING: frame sizes differ: vllm={vllm_frames.shape[1:3]} vs diff={diff_frames.shape[1:3]}")
            print(f"  Skipping accuracy comparison for this test")
            continue

        vllm_frames = vllm_frames.astype(np.float64)
        diff_frames = diff_frames.astype(np.float64)

        # Per-frame PSNR
        psnrs = [compute_psnr(vllm_frames[i], diff_frames[i]) for i in range(n)]
        avg_psnr = float(np.mean(psnrs))

        # SSIM
        avg_ssim = compute_ssim(
            vllm_frames.astype(np.uint8),
            diff_frames.astype(np.uint8),
        )

        ssim_str = f"{avg_ssim:.4f}" if avg_ssim is not None else "N/A"
        print(f"{display_name:<30} {avg_psnr:>10.2f} {ssim_str:>10} {n:>14}")

        results.append({
            "name": display_name,
            "psnr_db": round(avg_psnr, 2),
            "ssim": round(avg_ssim, 4) if avg_ssim is not None else None,
            "num_frames_compared": n,
        })

    results_path = os.path.join(args.output_dir, "bench_accuracy.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
