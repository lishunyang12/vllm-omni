#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare a self-contained FLUX.2 NVFP4 model directory for vllm-omni.

BFL ships NVFP4 quantized FLUX.2 weights as a *flat* checkpoint repo (only
``.safetensors`` files at the repo root), without the diffusers component
layout (``transformer/``, ``vae/``, ``text_encoder/``, ``scheduler/``,
``model_index.json``) that ``vllm-omni`` expects. Using the NVFP4 repo
directly requires juggling two repos at load time, which doesn't compose
with vllm-omni's normal loader.

This script combines:

    <base-repo>                                      (e.g. FLUX.2-klein-4B)
       ├── model_index.json
       ├── scheduler/
       ├── text_encoder/
       ├── tokenizer/
       ├── vae/
       └── transformer/
            ├── config.json                          ← ARCH params
            └── diffusion_pytorch_model.safetensors  ← BF16 weights (replaced)

    <nvfp4-repo>                                     (e.g. FLUX.2-klein-4b-nvfp4)
       └── flux-2-klein-4b-nvfp4.safetensors         ← BFL-format NVFP4 weights

into a single directory layout that ``--model`` can consume:

    <output-dir>/
       ├── model_index.json
       ├── scheduler/ ...
       ├── text_encoder/ ...
       ├── tokenizer/ ...
       ├── vae/ ...
       └── transformer/
            ├── config.json                          ← ARCH params + quantization_config block
            └── diffusion_pytorch_model.safetensors  ← BFL NVFP4 weights (symlinked)

The BFL-format weights keep their original tensor names; vllm-omni's
``bfl_mapping`` helper handles the BFL → diffusers rename at load time.

After running this script, inference uses the standard loader path with a
single ``--model`` argument. No new config fields, no special loader, no
engine_args whitelist changes.

    python examples/offline_inference/text_to_image/text_to_image.py \\
        --model ./flux2-klein-4b-nvfp4-merged \\
        --prompt "a photo of an astronaut riding a horse on Mars"
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _download_repo(repo_id: str, allow_patterns: list[str] | None = None) -> Path:
    """Download a HF repo to the local cache and return the snapshot path."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
    )
    return Path(local)


def _parse_quant_metadata(nvfp4_safetensors_path: Path) -> dict:
    """Read _quantization_metadata from the NVFP4 safetensors header and
    build a ``quantization_config`` dict ready to inject into the merged
    ``transformer/config.json``.
    """
    import sys

    # Isolate from vllm/vllm-omni imports — this script is a pure-python
    # prep helper that should work even when vllm's C extensions are missing.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from vllm_omni.diffusion.utils.nvfp4_utils import (  # noqa: E402
        parse_nvfp4_quant_metadata,
    )

    quant_dict = parse_nvfp4_quant_metadata(str(nvfp4_safetensors_path))
    if quant_dict is None:
        raise RuntimeError(
            f"Could not parse NVFP4 metadata from {nvfp4_safetensors_path}. Is this really a ModelOpt-NVFP4 checkpoint?"
        )
    qc = dict(quant_dict["quantization"])
    # Mark which vllm-omni factory entry to dispatch through. We use the
    # FLUX.2-specific variant which adds a one-byte nibble swap on top of
    # upstream's ModelOptNvFp4LinearMethod — NVIDIA's BFL NVFP4 ckps pack
    # FP4 nibbles in the opposite order to the LLM path (see sgl-project/
    # sglang#20137 for the same workaround).
    qc["quant_method"] = "modelopt_fp4_flux"
    return qc


def _copy_or_link(src: Path, dst: Path, *, use_symlink: bool) -> None:
    """Materialize *src* at *dst* either as a symlink (fast, no extra disk)
    or a real copy (safer, portable)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HF repo ID or local path for the BF16 base model "
        "(provides text_encoder, vae, scheduler, tokenizer, transformer/config.json).",
    )
    parser.add_argument(
        "--nvfp4",
        default="black-forest-labs/FLUX.2-klein-4b-nvfp4",
        help="HF repo ID or local path for the BFL-format NVFP4 transformer weights.",
    )
    parser.add_argument(
        "--nvfp4-file",
        default=None,
        help="Specific safetensors filename inside --nvfp4. When omitted we "
        "pick the first non-'mixed' variant via resolve_nvfp4_checkpoint_file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to materialize the merged model directory. Pass this as --model to the inference example.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy weight files instead of symlinking (portable, uses 2x disk).",
    )
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1. Download the BF16 base — we need the non-transformer components
    #    (VAE / TE / scheduler / tokenizer) and the transformer's config.json.
    print(f"[1/4] Downloading base repo: {args.base}")
    base_dir = _download_repo(args.base)
    print(f"       → {base_dir}")

    # 2. Symlink every base component into out/. The only thing we skip is
    #    the BF16 transformer weight file, which gets replaced in step 4.
    print(f"[2/4] Mirroring base layout into {out}")
    for src in base_dir.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(base_dir)
        if rel.parts[0] == "transformer" and rel.name.endswith(".safetensors"):
            # Handled in step 4
            continue
        _copy_or_link(src, out / rel, use_symlink=not args.copy)

    # 3. Download the NVFP4 weights and pick the right variant.
    print(f"[3/4] Downloading NVFP4 weights: {args.nvfp4}")
    nvfp4_dir = _download_repo(args.nvfp4, allow_patterns=["*.safetensors"])
    if args.nvfp4_file is not None:
        nvfp4_file = nvfp4_dir / args.nvfp4_file
    else:
        import sys

        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from vllm_omni.diffusion.utils.nvfp4_utils import (  # noqa: E402
            resolve_nvfp4_checkpoint_file,
        )

        picked = resolve_nvfp4_checkpoint_file(str(nvfp4_dir))
        if picked is None:
            raise RuntimeError(f"No *.safetensors found under {nvfp4_dir}")
        nvfp4_file = nvfp4_dir / picked
    if not nvfp4_file.exists():
        raise FileNotFoundError(f"NVFP4 weight file missing: {nvfp4_file}")
    print(f"       → {nvfp4_file.name} ({nvfp4_file.stat().st_size / 1e9:.1f} GB)")

    # 4. Place the NVFP4 weight at the diffusers transformer path and patch
    #    transformer/config.json with the quantization_config block so that
    #    vllm-omni's TransformerConfig.from_dict auto-detects the quant
    #    method at load time (mirrors the AutoRound flow from PR #1777).
    print("[4/4] Writing transformer/config.json + NVFP4 weight")
    tf_weight_dst = out / "transformer" / "diffusion_pytorch_model.safetensors"
    _copy_or_link(nvfp4_file, tf_weight_dst, use_symlink=not args.copy)

    tf_config_path = out / "transformer" / "config.json"
    if not tf_config_path.exists():
        raise FileNotFoundError(
            f"Expected {tf_config_path} from the base repo. Does --base ({args.base}) point to a diffusers-layout repo?"
        )
    with open(tf_config_path) as f:
        tf_cfg = json.load(f)
    tf_cfg["quantization_config"] = _parse_quant_metadata(nvfp4_file)
    with open(tf_config_path, "w") as f:
        json.dump(tf_cfg, f, indent=2)

    print()
    print("Done. Run with:")
    print()
    print("  python examples/offline_inference/text_to_image/text_to_image.py \\")
    print(f"    --model {out} \\")
    print('    --prompt "a photo of an astronaut riding a horse on Mars" \\')
    print("    --num-inference-steps 20 \\")
    print(f"    --output {out.parent / 'out.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
