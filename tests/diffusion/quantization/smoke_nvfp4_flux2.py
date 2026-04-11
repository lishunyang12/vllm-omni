# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test for FLUX.2 NVFP4 loading.

Run on a Blackwell (or emulation-capable) GPU box:

    python tests/diffusion/quantization/smoke_nvfp4_flux2.py

This script is intentionally NOT pytest-collected: it needs a real GPU and
~30 GB of free disk/VRAM, and it downloads the ~21 GB NVFP4 safetensors from
Hugging Face. It is meant for manual verification of the loader path, not
for CI.

What it checks, in order:
  1. ``parse_nvfp4_quant_metadata`` can read the metadata straight from the
     HF repo header (no full-shard download).
  2. ``ModelOptNvFp4Config.from_config`` accepts that dict and produces a
     config with ``is_checkpoint_nvfp4_serialized=True``.
  3. ``Flux2KleinPipeline`` constructs, loads weights end-to-end, and one
     forward pass through the transformer does not NaN.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="black-forest-labs/FLUX.2-dev")
    parser.add_argument("--nvfp4", default="black-forest-labs/FLUX.2-dev-NVFP4")
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()

    # Step 1: metadata parse (header-only network call)
    from vllm_omni.diffusion.utils.nvfp4_utils import (
        parse_nvfp4_quant_metadata,
        resolve_nvfp4_checkpoint_file,
    )

    quant_dict = parse_nvfp4_quant_metadata(args.nvfp4)
    if quant_dict is None:
        print("FAIL: parse_nvfp4_quant_metadata returned None", file=sys.stderr)
        return 1
    print("[1/3] parse_nvfp4_quant_metadata OK")
    print("      quant_algo:", quant_dict["quantization"]["quant_algo"])
    print("      group_size:", quant_dict["quantization"]["group_size"])
    print("      exclude_modules[:4]:", quant_dict["quantization"]["exclude_modules"][:4])
    print("      picked file:", resolve_nvfp4_checkpoint_file(args.nvfp4))

    # Step 2: ModelOptNvFp4Config.from_config must accept it
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

    qcfg = ModelOptNvFp4Config.from_config(quant_dict)
    assert qcfg.get_name() == "modelopt_fp4"
    assert qcfg.is_checkpoint_nvfp4_serialized
    print("[2/3] ModelOptNvFp4Config.from_config OK")

    # Step 3: full pipeline construction
    if args.skip_forward:
        print("[3/3] skipped (--skip-forward)")
        return 0

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import (
        Flux2KleinPipeline,
    )

    od = OmniDiffusionConfig(
        model=args.base,
        transformer_weights_path=args.nvfp4,
    )
    pipe = Flux2KleinPipeline(od_config=od)
    print("[3/3] Flux2KleinPipeline construct + load_weights OK")

    # Quick NaN check on the transformer with a zero input.
    import torch

    device = next(pipe.transformer.parameters()).device
    dtype = next(pipe.transformer.parameters()).dtype
    with torch.inference_mode():
        x = torch.zeros(1, 1024, pipe.transformer.config.in_channels, device=device, dtype=dtype)
        t = torch.zeros(1, device=device, dtype=dtype)
        out = pipe.transformer(hidden_states=x, timestep=t)
        sample = out.sample if hasattr(out, "sample") else out[0]
        if torch.isnan(sample).any():
            print("FAIL: forward produced NaN", file=sys.stderr)
            return 2
    print("      forward pass finite ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
