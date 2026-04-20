#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dequantize one NVFP4 Linear weight back to BF16 and compare against the
original unquantized weight.

Purpose: definitively verify the end-to-end byte-level packing of our NVFP4
checkpoint. If the dequantized weight is close to the original BF16 weight,
ModelOpt's export + scales are fine, and any quality issue must be downstream
(vLLM loader / kernel). If the dequantized weight is nowhere near the
original, the calibration / export produced a non-standard layout.

The FP4 E2M1 codebook (6 bits represent 16 values; here we use the 4-bit
version): each 4-bit code maps to one of 16 fixed fp32 values. We try both
"low nibble first" and "high nibble first" packings and pick whichever gives
the lower MSE against the original.

Example:
    python examples/quantization/dequant_nvfp4_compare.py \\
        --original hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \\
        --nvfp4 shunyang90/HunyuanVideo-1.5-480p-ModelOpt-NVFP4 \\
        --param transformer_blocks.0.attn.to_q.weight
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch


# FP4 E2M1 codebook: 16 values (1 sign bit, 2 exponent, 1 mantissa).
# Ordering is the NVIDIA / ModelOpt / vLLM convention (sign-magnitude).
FP4_E2M1_CODEBOOK = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _resolve(model: str, subfolder: str | None = "transformer") -> Path:
    p = Path(model)
    if p.exists():
        return p / subfolder if (subfolder and (p / subfolder).exists()) else p
    from huggingface_hub import snapshot_download

    allow = [f"{subfolder}/*"] if subfolder else None
    root = Path(snapshot_download(model, allow_patterns=allow))
    return root / subfolder if (subfolder and (root / subfolder).exists()) else root


def _read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def _tensor_from_checkpoint(root: Path, name: str) -> torch.Tensor | None:
    from safetensors import safe_open

    idx_path = root / "diffusion_pytorch_model.safetensors.index.json"
    if idx_path.exists():
        with idx_path.open(encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
        if name not in weight_map:
            return None
        path = root / weight_map[name]
    else:
        # Single shard fallback: search all safetensors
        path = None
        for f in sorted(root.glob("*.safetensors")):
            header = _read_safetensors_header(f)
            if name in header:
                path = f
                break
        if path is None:
            return None

    with safe_open(str(path), framework="pt", device="cpu") as sf:
        return sf.get_tensor(name)


def _unpack_fp4(packed_u8: torch.Tensor, low_first: bool) -> torch.Tensor:
    """Unpack U8 bytes to FP4 index pairs.

    low_first=True:  byte = (code[2i+1] << 4) | code[2i]
    low_first=False: byte = (code[2i] << 4) | code[2i+1]

    Returns int64 indices shape [..., 2*cols].
    """
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    if low_first:
        # [low, high, low, high, ...] → element 2i = low[i], 2i+1 = high[i]
        interleaved = torch.stack([low, high], dim=-1)
    else:
        interleaved = torch.stack([high, low], dim=-1)
    return interleaved.reshape(*packed_u8.shape[:-1], -1).long()


def _dequantize(
    packed_weight: torch.Tensor,
    weight_scale_fp8: torch.Tensor,
    weight_scale_2: torch.Tensor,
    low_first: bool,
    group_size: int = 16,
) -> torch.Tensor:
    """Return BF16 dequantized weight."""
    # 1. Unpack FP4 indices
    fp4_indices = _unpack_fp4(packed_weight, low_first=low_first)  # [M, 2*packed_K]
    # 2. Look up codebook
    codebook = FP4_E2M1_CODEBOOK.to(device=packed_weight.device)
    fp4_values = codebook[fp4_indices]  # [M, full_K] fp32
    # 3. Apply per-group scale
    M, full_K = fp4_values.shape
    assert full_K % group_size == 0, f"K={full_K} not divisible by group_size={group_size}"
    num_groups = full_K // group_size
    ws = weight_scale_fp8.to(torch.float32).to(fp4_values.device)  # [M, num_groups]
    assert ws.shape == (M, num_groups), f"weight_scale shape {tuple(ws.shape)} != expected ({M}, {num_groups})"
    fp4_grouped = fp4_values.reshape(M, num_groups, group_size)
    dequant = fp4_grouped * ws.unsqueeze(-1) * float(weight_scale_2.item())
    return dequant.reshape(M, full_K)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", required=True, help="Original BF16 diffusers repo id or path")
    p.add_argument("--nvfp4", required=True, help="NVFP4 diffusers repo id or path")
    p.add_argument(
        "--param",
        default="transformer_blocks.0.attn.to_q.weight",
        help="Full parameter name (excluding the quantization suffix)",
    )
    p.add_argument("--group-size", type=int, default=16)
    args = p.parse_args()

    orig_root = _resolve(args.original, subfolder="transformer")
    nvfp4_root = _resolve(args.nvfp4, subfolder="transformer")

    print(f"Original root: {orig_root}")
    print(f"NVFP4 root:    {nvfp4_root}")
    print(f"Param:         {args.param}")
    print()

    orig = _tensor_from_checkpoint(orig_root, args.param)
    if orig is None:
        raise SystemExit(f"Original tensor {args.param!r} not found under {orig_root}")
    packed = _tensor_from_checkpoint(nvfp4_root, args.param)
    if packed is None:
        raise SystemExit(f"NVFP4 tensor {args.param!r} not found under {nvfp4_root}")

    scale_name = args.param.replace(".weight", ".weight_scale")
    scale2_name = args.param.replace(".weight", ".weight_scale_2")
    ws = _tensor_from_checkpoint(nvfp4_root, scale_name)
    ws2 = _tensor_from_checkpoint(nvfp4_root, scale2_name)
    if ws is None or ws2 is None:
        raise SystemExit(f"Missing scale(s): {scale_name!r} / {scale2_name!r}")

    print(f"  original:       dtype={orig.dtype}  shape={tuple(orig.shape)}")
    print(f"  packed:         dtype={packed.dtype} shape={tuple(packed.shape)}")
    print(f"  weight_scale:   dtype={ws.dtype}     shape={tuple(ws.shape)}")
    print(f"  weight_scale_2: dtype={ws2.dtype}    value={ws2.item():.6g}")
    print()

    orig_f32 = orig.to(torch.float32)
    orig_mean_abs = orig_f32.abs().mean().item()
    orig_max_abs = orig_f32.abs().max().item()
    print(f"  original |w|:   mean={orig_mean_abs:.6g}  max={orig_max_abs:.6g}")

    for low_first in (True, False):
        dq = _dequantize(packed, ws, ws2, low_first=low_first, group_size=args.group_size)
        diff = (dq - orig_f32).abs()
        mse = (diff**2).mean().item()
        mean_abs_err = diff.mean().item()
        max_abs_err = diff.max().item()
        # Relative MSE compared to signal energy
        rel = mse / max(1e-30, (orig_f32**2).mean().item())
        label = "LOW-nibble first" if low_first else "HIGH-nibble first"
        print(f"  [{label}]  MSE={mse:.4g}  mean|err|={mean_abs_err:.4g}  max|err|={max_abs_err:.4g}  rel_MSE={rel:.4g}")

    print()
    print("Expectation if packing is correct: one ordering gives rel_MSE ~ 0.01-0.1")
    print("(typical NVFP4 quantization error). If BOTH orderings give rel_MSE ~ 1.0,")
    print("the export layout is fundamentally wrong, not just an endianness flip.")


if __name__ == "__main__":
    main()
