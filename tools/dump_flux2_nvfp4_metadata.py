#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump NVFP4 metadata and quantization_config for a merged FLUX.2 directory.

Used to diagnose exclude-list mismatches between the BFL checkpoint's
embedded ``_quantization_metadata`` and the ``quantization_config``
block that ``prepare_flux2_nvfp4.py`` injects into ``transformer/config.json``.

Usage:
    python tools/dump_flux2_nvfp4_metadata.py \\
        --merged-dir /workspace/flux2-dev-nvfp4-merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-dir",
        required=True,
        type=Path,
        help="Path to the merged directory (output of prepare_flux2_nvfp4.py).",
    )
    args = parser.parse_args()

    merged = args.merged_dir.resolve()
    cfg_path = merged / "transformer" / "config.json"
    weight_path = merged / "transformer" / "diffusion_pytorch_model.safetensors"

    # --- 1. injected quantization_config block --------------------------
    print("=" * 70)
    print(f"[1] {cfg_path}")
    print("=" * 70)
    if not cfg_path.exists():
        print(f"  !! missing: {cfg_path}")
    else:
        with open(cfg_path) as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config")
        if qc is None:
            print("  !! config.json has no 'quantization_config' block")
        else:
            print(json.dumps(qc, indent=2))

    # --- 2. embedded _quantization_metadata header ----------------------
    print()
    print("=" * 70)
    print(f"[2] {weight_path}")
    print("=" * 70)
    if not weight_path.exists():
        print(f"  !! missing: {weight_path}")
        return 1

    try:
        from safetensors import safe_open
    except ImportError:
        print("  !! safetensors not installed")
        return 1

    with safe_open(str(weight_path), framework="pt") as f:
        meta = f.metadata() or {}

    print("metadata top-level keys:", list(meta.keys()))
    raw = meta.get("_quantization_metadata")
    if raw is None:
        print("  !! no '_quantization_metadata' field in safetensors header")
        return 0

    if isinstance(raw, str):
        try:
            q = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  !! _quantization_metadata is not valid JSON: {e}")
            print(f"  raw first 500 bytes: {raw[:500]!r}")
            return 1
    else:
        q = raw

    print()
    print("top-level keys in _quantization_metadata:", list(q.keys()) if isinstance(q, dict) else type(q))

    if isinstance(q, dict):
        layers = q.get("layers") or q.get("layer") or {}
        if not isinstance(layers, dict):
            print(f"  !! 'layers' is not a dict: type={type(layers).__name__}")
        else:
            print(f"\ntotal layer entries: {len(layers)}")
            print("\n-- first 5 layer entries --")
            for name in list(layers)[:5]:
                print(f"  {name!r}:")
                print(f"    {json.dumps(layers[name], indent=2)[:400]}")

            # Key question: are txt_attn layers in here?
            txt_attn_hits = [name for name in layers if "txt_attn" in name]
            print(f"\n-- txt_attn entries: {len(txt_attn_hits)} --")
            for name in txt_attn_hits[:6]:
                info = layers[name]
                fmt = info.get("format") if isinstance(info, dict) else info
                print(f"  {name!r} format={fmt!r}")

            img_attn_hits = [name for name in layers if "img_attn" in name]
            print(f"\n-- img_attn entries: {len(img_attn_hits)} --")
            for name in img_attn_hits[:3]:
                info = layers[name]
                fmt = info.get("format") if isinstance(info, dict) else info
                print(f"  {name!r} format={fmt!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
