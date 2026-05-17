#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify a ModelOpt FP8 diffusers checkpoint exported by
quantize_hunyuanvideo_15_modelopt_fp8.py (or any sibling quantize_*.py).

Three checks:
  A. transformer/config.json has a sane quantization_config block.
  B. transformer/*.safetensors contains FP8 (float8_e4m3fn) tensors.
  C. transformer disk size is materially smaller than a BF16 baseline.

Example:
    python examples/quantization/check_modelopt_fp8_export.py \\
        --output ./hv15-480p-modelopt-fp8

    # Optional: compare disk size against a local or HF BF16 baseline.
    python examples/quantization/check_modelopt_fp8_export.py \\
        --output ./hv15-480p-modelopt-fp8 \\
        --baseline hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _check_config(transformer_dir: Path) -> int:
    """Returns 0 on pass, 1 on fail. Prints findings."""
    cfg_path = transformer_dir / "config.json"
    if not cfg_path.exists():
        print(f"[FAIL] {cfg_path} missing.")
        return 1

    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        print(f"[FAIL] No `quantization_config` block in {cfg_path}.")
        return 1

    print(f"[A] quantization_config from {cfg_path}:")
    print(json.dumps(qc, indent=2))

    issues = []
    if qc.get("quant_method") != "modelopt":
        issues.append(f"quant_method={qc.get('quant_method')!r} (expected 'modelopt')")

    quant_algo = qc.get("quant_algo")
    if quant_algo not in ("FP8", "FP8_PB_WO"):
        issues.append(
            f"quant_algo={quant_algo!r} (expected 'FP8' for per-tensor or "
            "'FP8_PB_WO' for 128x128 block-wise — other algos aren't routed by "
            "vllm-omni's adapter today)"
        )

    # Cross-check that the saved weight strategy and the dispatch field agree.
    # Producer scripts can in principle drift apart (e.g. metadata says "block"
    # but quant_algo still claims "FP8"), and that lands as an AssertionError at
    # weight load time because the runtime LinearMethod expects scalar scales but
    # finds 4D block ones. Failing here is much friendlier.
    cfg_groups = qc.get("config_groups", {})
    weight_strategies = {
        (group or {}).get("weights", {}).get("strategy")
        for group in cfg_groups.values()
        if isinstance(group, dict)
    }
    weight_strategies.discard(None)
    if weight_strategies == {"block"} and quant_algo != "FP8_PB_WO":
        issues.append(
            f"weights.strategy='block' but quant_algo={quant_algo!r}. Per-block "
            "weight scales require FP8_PB_WO so upstream vLLM dispatches to "
            "ModelOptFp8PbWoLinearMethod; FP8 routes to per-tensor and crashes "
            "on the 4D weight_scale at weight load time."
        )
    elif quant_algo == "FP8_PB_WO" and weight_strategies != {"block"}:
        issues.append(
            f"quant_algo='FP8_PB_WO' but weights.strategy={weight_strategies!r} "
            "(expected {'block'}). FP8_PB_WO consumers expect 4D per-block scales."
        )

    if issues:
        print("[A] WARN — config looks incomplete:")
        for issue in issues:
            print(f"    - {issue}")
        return 2
    print(f"[A] PASS — config looks correct (quant_algo={quant_algo}).")
    return 0


def _read_safetensors_header(path: Path) -> dict:
    """Read the JSON header of a safetensors file. Bypass-safe — doesn't materialize tensors.

    Returns {tensor_name: {'dtype': 'F8_E4M3', 'shape': [...], 'data_offsets': [...]}}.
    Header dtype strings: F8_E4M3, F8_E5M2, BF16, F16, F32, F64, I8, I16, I32, I64, BOOL, U8, ...
    """
    import struct

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def _classify_weight_scale_granularity(weight_scale_shapes: list[list[int]]) -> str:
    """Infer per-tensor vs per-channel vs per-block from sample weight_scale shapes.

    ModelOpt block-wise produces shapes like `[16, 1, 16, 1]` (broadcasting dims of 1
    interleaved with block-count dims). We count "meaningful" dims — ones with size > 1 —
    and classify: 0 meaningful dims = per-tensor (scalar), 1 = per-channel, 2+ = per-block.
    """
    if not weight_scale_shapes:
        return "no weight_scale tensors found"

    def meaningful_dims(shape: list[int]) -> int:
        return sum(1 for d in shape if d > 1)

    per_tensor = sum(1 for s in weight_scale_shapes if meaningful_dims(s) == 0)
    per_channel = sum(1 for s in weight_scale_shapes if meaningful_dims(s) == 1)
    per_block = sum(1 for s in weight_scale_shapes if meaningful_dims(s) >= 2)
    total = len(weight_scale_shapes)
    if per_tensor == total:
        return "per-tensor (all scalar scales)"
    if per_channel == total:
        return "per-channel (1 meaningful dim)"
    if per_block == total:
        return "per-block (2+ meaningful dims — e.g. [M//bm, 1, N//bn, 1] for tiles)"
    return f"mixed: per-tensor={per_tensor}, per-channel={per_channel}, per-block={per_block} of {total}"


def _check_safetensors(transformer_dir: Path) -> int:
    """Returns 0 on pass, 1 on fail. Reads on-disk dtype from the safetensors header."""
    files = sorted(transformer_dir.glob("*.safetensors"))
    if not files:
        print(f"[FAIL] No *.safetensors in {transformer_dir}.")
        return 1

    header_dtype_counts: Counter[str] = Counter()
    sample_fp8_keys: list[str] = []
    sample_scale_keys: list[str] = []
    weight_scale_shapes: list[list[int]] = []
    sample_weight_scale_entries: list[tuple[str, list[int]]] = []
    for f in files:
        try:
            header = _read_safetensors_header(f)
        except Exception as exc:
            print(f"[B] WARN — could not parse header of {f}: {exc}")
            continue
        for k, info in header.items():
            dtype = info.get("dtype", "?")
            header_dtype_counts[dtype] += 1
            if dtype.startswith("F8") and len(sample_fp8_keys) < 5:
                sample_fp8_keys.append(k)
            if k.endswith(("_scale", ".weight_scale", ".input_scale", "_scale_inv")) and len(sample_scale_keys) < 5:
                sample_scale_keys.append(k)
            if k.endswith(".weight_scale"):
                weight_scale_shapes.append(info.get("shape", []))
                if len(sample_weight_scale_entries) < 5:
                    sample_weight_scale_entries.append((k, info.get("shape", [])))

    print(f"\n[B] On-disk dtype counts across {len(files)} safetensors file(s) (from header, not get_tensor):")
    for dtype, count in sorted(header_dtype_counts.items(), key=lambda kv: -kv[1]):
        marker = "  <-- FP8" if dtype.startswith("F8") else ""
        print(f"    {dtype:10s} {count:>6d}{marker}")

    fp8_count = sum(c for d, c in header_dtype_counts.items() if d.startswith("F8"))
    if fp8_count == 0:
        print("[B] FAIL — no FP8 tensors on disk. Calibration likely did not actually quantize the weights.")
        return 1

    print(f"[B] PASS — {fp8_count} FP8 tensors stored on disk.")
    if sample_fp8_keys:
        print(f"    sample FP8 tensors:   {sample_fp8_keys[:3]}")
    if sample_scale_keys:
        print(f"    sample scale tensors: {sample_scale_keys[:3]}")
    print("    (Note: torch's get_tensor() may return these as bf16 views on some versions —")
    print("     irrelevant; vLLM's loader uses native FP8 ops.)")

    # Weight-scale granularity — per-tensor (scalar) vs per-channel (1-D) vs per-block (N-D).
    print(f"\n    weight_scale granularity: {_classify_weight_scale_granularity(weight_scale_shapes)}")
    for key, shape in sample_weight_scale_entries[:3]:
        print(f"      {key}: shape {shape}")
    return 0


def _disk_size_gib(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024**3)


def _transformer_subdirs(root: Path) -> list[Path]:
    """Return [<root>/transformer, <root>/transformer_2] for those that exist.

    Wan2.2 MoE A14B (T2V/I2V) and Wan2.2-VACE-A14B export TWO transformer
    subfolders; single-transformer checkpoints just have `transformer/`.
    Falls back to `[root]` if neither exists (e.g., a baseline directory
    that wasn't structured as a diffusers repo).
    """
    found = [root / name for name in ("transformer", "transformer_2") if (root / name).is_dir()]
    return found if found else [root]


def _check_size_vs_baseline(transformer_dir: Path, baseline: str | None) -> int:
    """Returns 0 always (informational only)."""
    # transformer_dir is <fp8_root>/transformer; walk one level up so we can
    # also pick up transformer_2/ for Wan2.2 MoE A14B checkpoints.
    fp8_root = transformer_dir.parent
    fp8_subdirs = _transformer_subdirs(fp8_root)
    fp8_size = sum(_disk_size_gib(p) for p in fp8_subdirs)
    fp8_label = " + ".join(p.name for p in fp8_subdirs)
    print(f"\n[C] FP8 transformer disk size ({fp8_label}): {fp8_size:.2f} GiB")

    if baseline is None:
        print("[C] SKIP — pass --baseline <path or HF id> to compare against BF16.")
        return 0

    baseline_path = Path(baseline)
    if not baseline_path.exists():
        # Treat `baseline` as an HF repo id and read from the local cache.
        # Don't trigger a download: this script is meant to run AFTER
        # quantize_*_modelopt_fp8.py, which already pulled the whole repo
        # into the cache. local_files_only=True makes that assumption
        # explicit — if the cache is empty we surface a clear error rather
        # than silently kicking off a multi-GB download.
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
        except ImportError:
            print("[C] SKIP — huggingface_hub not installed and baseline not a local path.")
            return 0
        try:
            baseline_path = Path(snapshot_download(baseline, local_files_only=True))
        except LocalEntryNotFoundError:
            print(
                f"[C] SKIP — '{baseline}' not found in local HF cache. "
                "Run the matching quantize_*_modelopt_fp8.py first (it caches the BF16 repo), "
                "or pass --baseline <local-dir>."
            )
            return 0
        print(f"    Resolved baseline from HF cache: {baseline_path}")

    bf16_subdirs = _transformer_subdirs(baseline_path)
    bf16_size = sum(_disk_size_gib(p) for p in bf16_subdirs)
    if bf16_size == 0:
        print(f"[C] WARN — baseline transformer dir empty: {baseline_path}")
        return 0

    bf16_label = " + ".join(p.name for p in bf16_subdirs)
    reduction = (1 - fp8_size / bf16_size) * 100
    print(f"[C] BF16 baseline transformer disk size ({bf16_label}): {bf16_size:.2f} GiB ({baseline_path})")
    print(f"[C] Disk reduction: {reduction:.1f}%  (FP8 is {fp8_size / bf16_size:.0%} of BF16)")
    if reduction < 30:
        print("[C] WARN — FP8 should typically reduce disk by ~40-50%; <30% suggests partial quantization.")

    # Whole-repo view: includes VAE / text_encoder / tokenizer / scheduler /
    # top-level metadata. Quantization only touches transformer(s) so this
    # reduction is always smaller than the transformer-only one — but it's
    # what the deployment footprint actually is.
    fp8_total = _disk_size_gib(fp8_root)
    bf16_total = _disk_size_gib(baseline_path)
    if bf16_total > 0:
        total_reduction = (1 - fp8_total / bf16_total) * 100
        print(
            f"[C] Whole-repo: FP8 {fp8_total:.2f} GiB / BF16 {bf16_total:.2f} GiB "
            f"(reduction {total_reduction:.1f}%, deployment footprint)"
        )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="Path to the exported ModelOpt FP8 checkpoint root.")
    p.add_argument(
        "--baseline",
        default=None,
        help="Optional BF16 baseline (local diffusers dir or HF id) for disk-size comparison.",
    )
    args = p.parse_args()

    out_root = Path(args.output).expanduser().resolve()
    transformer_dir = out_root / "transformer"
    if not transformer_dir.exists():
        print(f"[FAIL] {transformer_dir} does not exist.")
        sys.exit(1)

    print(f"Checking: {out_root}\n")

    fail = 0
    fail |= _check_config(transformer_dir)
    fail |= _check_safetensors(transformer_dir)
    _check_size_vs_baseline(transformer_dir, args.baseline)

    print()
    if fail == 0:
        print("=" * 60)
        print("ALL CHECKS PASSED — checkpoint looks ready for vllm-omni serving.")
    elif fail == 1:
        print("=" * 60)
        print("FAILURES detected — calibration may need to be re-run.")
        sys.exit(1)
    else:
        print("=" * 60)
        print("WARNINGS only — checkpoint may serve but with caveats. See [A] above.")


if __name__ == "__main__":
    main()
