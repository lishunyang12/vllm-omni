#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump raw scale values from a ModelOpt NVFP4 diffusers checkpoint.

Reads the transformer's safetensors directly (no vllm-omni / vLLM load path)
and prints the shapes and a few values for one representative attention block's
scales — so we can triage whether calibration produced sane numbers before
debugging the serving-side loader / kernel.

Expected NVFP4 scale layout (per vLLM ModelOptNvFp4LinearMethod):
    input_scale      : F32 scalar   (per-tensor activation global scale)
    weight_scale     : F8_E4M3 [out, in/16]   (per-group block scale)
    weight_scale_2   : F32 scalar   (per-tensor weight global scale)

Sanity ranges:
    input_scale       typical O(1e-3) to O(1e0)
    weight_scale_2    typical O(1e-4) to O(1e-1)
    alpha = input * weight_2  typical O(1e-7) to O(1e-1)
    weight_scale values   well within FP8 range (roughly |x| < 10)

Example:
    python examples/quantization/inspect_nvfp4_scales.py \\
        --model shunyang90/HunyuanVideo-1.5-480p-ModelOpt-NVFP4 \\
        --layer-prefix transformer_blocks.0.attn \\
        --shards to_q to_k to_v to_out.0
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def _resolve_model_dir(model: str) -> Path:
    p = Path(model)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    print(f"Downloading (or using cached) {model!r} ...")
    return Path(snapshot_download(model, allow_patterns=["transformer/*"]))


def _read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def _build_shard_index(transformer_dir: Path) -> dict[str, Path]:
    """Map tensor name -> path of safetensors file containing it."""
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as f:
            idx = json.load(f)
        return {k: transformer_dir / v for k, v in idx["weight_map"].items()}

    # Single-shard checkpoint: one .safetensors file, build index from its header
    files = sorted(transformer_dir.glob("*.safetensors"))
    if len(files) != 1:
        raise SystemExit(f"Expected index.json or a single safetensors file under {transformer_dir}")
    header = _read_safetensors_header(files[0])
    return {k: files[0] for k in header.keys()}


def _read_tensor(path: Path, name: str):
    """Read a single tensor from a safetensors file."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _describe_scalar(t) -> str:
    import torch  # local import: avoid requiring torch just to parse CLI

    if not isinstance(t, torch.Tensor):
        return str(t)
    if t.numel() == 1:
        return f"{t.item():.6g}"
    return f"min={t.min().item():.6g} max={t.max().item():.6g} mean={t.float().mean().item():.6g}"


def _inspect_shard(shard_index: dict[str, Path], full_name: str) -> None:
    if full_name not in shard_index:
        print(f"  [MISSING] {full_name}")
        return
    path = shard_index[full_name]
    header = _read_safetensors_header(path)
    info = header.get(full_name, {})
    dtype = info.get("dtype", "?")
    shape = info.get("shape", [])
    print(f"  {full_name}")
    print(f"    dtype={dtype}  shape={tuple(shape)}  file={path.name}")

    # Materialize the tensor for scalar / small tensors; skip huge ones
    total = 1
    for d in shape:
        total *= int(d) if d else 1
    if total <= 4096 or full_name.endswith(("input_scale", "weight_scale_2")):
        try:
            t = _read_tensor(path, full_name)
            print(f"    value: {_describe_scalar(t)}")
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"    (could not materialize: {exc})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id or local diffusers dir")
    p.add_argument(
        "--layer-prefix",
        default="transformer_blocks.0.attn",
        help="Attention block prefix (default: transformer_blocks.0.attn)",
    )
    p.add_argument(
        "--shards",
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"],
        help="Shard suffixes to inspect (default: to_q to_k to_v to_out.0)",
    )
    p.add_argument(
        "--scale-suffixes",
        nargs="+",
        default=["input_scale", "weight_scale_2", "weight_scale"],
        help="Which scale tensors to print (default: input_scale weight_scale_2 weight_scale)",
    )
    args = p.parse_args()

    root = _resolve_model_dir(args.model)
    transformer_dir = root / "transformer" if (root / "transformer").exists() else root

    shard_index = _build_shard_index(transformer_dir)
    print(f"Inspecting: {transformer_dir}")
    print(f"Layer:      {args.layer_prefix}")
    print(f"Shards:     {args.shards}\n")

    # Also inspect the weight itself for one shard so we can see the packed U8 shape
    first_shard = args.shards[0]
    weight_name = f"{args.layer_prefix}.{first_shard}.weight"
    print(f"[weight   ] {first_shard}:")
    _inspect_shard(shard_index, weight_name)
    print()

    for suffix in args.scale_suffixes:
        print(f"[{suffix}]:")
        for shard in args.shards:
            full_name = f"{args.layer_prefix}.{shard}.{suffix}"
            _inspect_shard(shard_index, full_name)
        print()

    print(
        "Sanity: input_scale / weight_scale_2 should be finite, positive, O(1e-4..1e0). "
        "weight_scale values should be modest (|x| < ~10). "
        "If any are 0 / NaN / Inf → calibration bug."
    )


if __name__ == "__main__":
    main()
