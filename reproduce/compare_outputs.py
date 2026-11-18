"""Compare saved DiT tensors; these are not decoded video quality metrics."""

import argparse
import json
from pathlib import Path

import torch


def compare(actual, reference):
    assert len(actual) == len(reference) == 2
    results = []
    for name, a, b in zip(("video", "audio"), actual, reference):
        assert a.shape == b.shape and a.dtype == b.dtype
        af, bf = a.double().reshape(-1), b.double().reshape(-1)
        error = af - bf
        results.append({
            "output": name, "shape": list(a.shape), "dtype": str(a.dtype),
            "finite": bool(torch.isfinite(af).all() and torch.isfinite(bf).all()),
            "bitwise_equal": torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)),
            "max_abs": error.abs().max().item(),
            "rmse": error.square().mean().sqrt().item(),
            "relative_l2": (error.norm() / bf.norm().clamp_min(1e-12)).item(),
            "cosine": torch.nn.functional.cosine_similarity(af, bf, dim=0).item(),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = {}
    for root in args.roots:
        reports[root.name] = {}
        for rank in range(2):
            tensors = {
                (variant, repeat): torch.load(
                    root / f"output-{variant}-{repeat}-rank-{rank}.pt", map_location="cpu", weights_only=True
                )
                for variant in ("dense", "sol") for repeat in ("first", "last")
            }
            reports[root.name][f"rank{rank}"] = {
                "sol_vs_dense": compare(tensors["sol", "last"], tensors["dense", "last"]),
                **{f"{variant}_last_vs_first": compare(tensors[variant, "last"], tensors[variant, "first"])
                   for variant in ("dense", "sol")},
            }
    args.output.write_text(json.dumps({
        "scope": "Full-forward output tensors only; no SSIM, PSNR, LPIPS or decoded quality assessment.",
        "comparisons": reports,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
