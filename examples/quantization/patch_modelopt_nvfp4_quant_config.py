#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retrofit older ModelOpt NVFP4 checkpoints with the flat schema fields.

Upstream ``ModelOptNvFp4Config._from_config`` requires ``group_size``,
``kv_cache_quant_algo`` and ``exclude_modules`` at the top level of each
transformer's ``config.json#quantization_config``.  Earlier revisions of the
calibration scripts only emitted the compressed-tensors-style ``config_groups``
+ ``ignore`` block, which makes the checkpoint fail to load with::

    ValueError: NVFP4 quantization requires the following fields in
    hf_quant_config.json: ['group_size', 'kv_cache_quant_algo', 'exclude_modules']

This script patches an existing checkpoint in place by reading ``group_size``
from the ``config_groups.group_0.weights`` entry and copying ``ignore`` into
``exclude_modules``.  Weights are not touched -- recalibration is unnecessary.

Example::

    python examples/quantization/patch_modelopt_nvfp4_quant_config.py \\
        /workspace/vllm-omni/wan22-t2v-a14b-modelopt-nvfp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRANSFORMER_SUBDIRS = ("transformer", "transformer_2")


def _patch_one(cfg_path: Path, *, dry_run: bool) -> bool:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        print(f"  [skip] {cfg_path}: no quantization_config block")
        return False

    weights_block = qc.get("config_groups", {}).get("group_0", {}).get("weights", {})
    group_size = qc.get("group_size", weights_block.get("group_size", 16))
    exclude_modules = qc.get("exclude_modules") or list(qc.get("ignore", []))
    kv_cache_quant_algo = qc.get("kv_cache_quant_algo", None)

    needs_update = "group_size" not in qc or "exclude_modules" not in qc or "kv_cache_quant_algo" not in qc
    if not needs_update:
        print(f"  [ok]   {cfg_path}: already has flat schema fields")
        return False

    qc["group_size"] = group_size
    qc["exclude_modules"] = exclude_modules
    qc["kv_cache_quant_algo"] = kv_cache_quant_algo
    cfg["quantization_config"] = qc

    if dry_run:
        print(
            f"  [dry]  {cfg_path}: would add group_size={group_size}, "
            f"kv_cache_quant_algo={kv_cache_quant_algo}, "
            f"exclude_modules ({len(exclude_modules)} entries)"
        )
        return True

    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"  [done] {cfg_path}: added flat schema fields")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Path to a ModelOpt NVFP4 checkpoint directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change but don't write.")
    args = parser.parse_args()

    root = Path(args.checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    candidates: list[Path] = []
    for sub in TRANSFORMER_SUBDIRS:
        cfg = root / sub / "config.json"
        if cfg.exists():
            candidates.append(cfg)
    if not candidates:
        single = root / "config.json"
        if single.exists():
            candidates.append(single)

    if not candidates:
        raise SystemExit(f"No transformer config.json found under {root}")

    print(f"Patching {len(candidates)} config(s) under {root}:")
    changed = 0
    for cfg_path in candidates:
        if _patch_one(cfg_path, dry_run=args.dry_run):
            changed += 1

    if args.dry_run:
        print(f"\nDry run complete: {changed} file(s) would change.")
    else:
        print(f"\nDone: {changed} file(s) updated.")
    sys.exit(0)


if __name__ == "__main__":
    main()
