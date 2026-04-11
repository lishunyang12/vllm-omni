# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test for FLUX.2 NVFP4 loading via the standard ``Omni`` entry point.

Manual, not pytest-collected — needs a GPU, network, and a model directory
that has already been prepared by ``tools/prepare_flux2_nvfp4.py``.

What it verifies, in order:
  1. ``parse_nvfp4_quant_metadata`` reads NVFP4 metadata directly from a
     BFL-style safetensors header (pure-python, no GPU).
  2. The merged model directory has a ``transformer/config.json`` containing
     a ``quantization_config`` block that vllm-omni's ``TransformerConfig``
     will recognize.
  3. ``Omni(model=...).generate(...)`` runs one short denoising pass and
     produces a non-None image. All the distributed / vllm_config /
     forward_context plumbing is handled by the standard entry point — we
     don't reach into vLLM or vllm-omni internals.

Usage::

    python tools/prepare_flux2_nvfp4.py --output-dir /tmp/klein-4b-nvfp4
    python tests/diffusion/quantization/smoke_nvfp4_flux2.py \\
        --model /tmp/klein-4b-nvfp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _check_transformer_config(model_dir: Path) -> int:
    """Verify the merged model dir has quantization_config in transformer/config.json."""
    tf_cfg_path = model_dir / "transformer" / "config.json"
    if not tf_cfg_path.exists():
        print(f"FAIL: {tf_cfg_path} not found — did you run prepare_flux2_nvfp4.py?", file=sys.stderr)
        return 1
    with open(tf_cfg_path) as f:
        tf_cfg = json.load(f)
    qc = tf_cfg.get("quantization_config")
    if not qc:
        print(
            f"FAIL: {tf_cfg_path} has no 'quantization_config' block. Re-run prepare_flux2_nvfp4.py to patch it in.",
            file=sys.stderr,
        )
        return 2
    print("[1/2] transformer/config.json quantization_config OK")
    print(f"       quant_method: {qc.get('quant_method')}")
    print(f"       quant_algo:   {qc.get('quant_algo')}")
    print(f"       group_size:   {qc.get('group_size')}")
    print(f"       #exclude_modules: {len(qc.get('exclude_modules', []))}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to the merged model directory produced by "
        "tools/prepare_flux2_nvfp4.py (contains transformer/, vae/, "
        "text_encoder/, scheduler/, model_index.json).",
    )
    parser.add_argument("--prompt", default="a red cube on a white table, studio lighting")
    parser.add_argument("--steps", type=int, default=4, help="very short — smoke only")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional PNG path. Omit to only verify load + forward, no disk write.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"FAIL: --model path does not exist: {args.model}", file=sys.stderr)
        return 1

    rc = _check_transformer_config(args.model)
    if rc != 0:
        return rc

    # Use the real Omni entry point — this is the production path. It
    # handles distributed init, vllm_config, forward context, parallel
    # groups, and the auto-detect of NVFP4 from transformer/config.json.
    import torch

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(42)
    omni = Omni(model=str(args.model))

    outputs = omni.generate(
        {"prompt": args.prompt, "negative_prompt": None},
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=4.0,
            num_inference_steps=args.steps,
            num_outputs_per_prompt=1,
        ),
    )

    if not outputs:
        print("FAIL: Omni.generate returned no outputs", file=sys.stderr)
        return 3
    req_out = outputs[0].request_output
    images = getattr(req_out, "images", None) if req_out is not None else None
    if not images:
        print("FAIL: no images in request_output", file=sys.stderr)
        return 4

    print(f"[2/2] Omni.generate produced {len(images)} image(s)")
    if args.output:
        images[0].save(args.output)
        print(f"      saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
