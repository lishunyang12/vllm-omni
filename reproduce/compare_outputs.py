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
    parser.add_argument("--cross-process", action="store_true", help="Also compare fresh runs against the first root")
    args = parser.parse_args()
    reports = {}
    for root in args.roots:
        reports[root.name] = {}
        protocol = json.loads((root / "protocol.json").read_text())
        for rank in range(protocol["engine"]["num_gpus"]):
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
    cross_process = {}
    if args.cross_process:
        reference = args.roots[0]
        reference_protocol = json.loads((reference / "protocol.json").read_text())
        count = reference_protocol["engine"]["num_gpus"]
        for root in args.roots[1:]:
            protocol = json.loads((root / "protocol.json").read_text())
            assert protocol["engine"] == reference_protocol["engine"]
            assert protocol["request_sha256"] == reference_protocol["request_sha256"]
            cross_process[root.name] = {}
            for rank in range(count):
                for variant in ("dense", "sol"):
                    filename = f"output-{variant}-last-rank-{rank}.pt"
                    cross_process[root.name][f"{variant}_rank{rank}_vs_{reference.name}"] = compare(
                        torch.load(root / filename, map_location="cpu", weights_only=True),
                        torch.load(reference / filename, map_location="cpu", weights_only=True),
                    )
    args.output.write_text(json.dumps({
        "scope": "Full-forward output tensors only; no SSIM, PSNR, LPIPS or decoded quality assessment.",
        "comparisons": reports,
        "fresh_process_comparisons": cross_process,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
