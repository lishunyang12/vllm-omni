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
import os
import sys


def _make_minimal_vllm_config():
    """Build a minimal VllmConfig sufficient for parallel-linear construction."""
    from vllm.config import VllmConfig
    from vllm.config.parallel import ParallelConfig

    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    return VllmConfig(parallel_config=parallel_config)


def _init_distributed_only() -> None:
    """Bring up torch.distributed in single-rank mode (no model-parallel yet).

    Flux2 transformer linears (QKVParallelLinear / ColumnParallelLinear / ...)
    call ``get_tensor_model_parallel_world_size()`` in their ``__init__``,
    which asserts that the TP group exists. We split this from the
    model-parallel init because the latter must run *inside* a
    ``set_current_vllm_config(...)`` context.
    """
    import torch.distributed as dist
    from vllm.distributed import init_distributed_environment

    if dist.is_available() and dist.is_initialized():
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
        backend="nccl",
    )


def _run_pipeline_under_vllm_config(args: argparse.Namespace) -> int:
    """Step 3 body: must run entirely inside the set_current_vllm_config ctx.

    vLLM custom ops (RMSNorm, NVFP4 linear, ...) read the global vllm_config
    in their ``__init__`` and again on every forward call, so the context
    has to stay alive for the entire pipeline construction *and* the smoke
    forward pass.
    """
    from vllm.config import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized

    vllm_config = _make_minimal_vllm_config()
    with set_current_vllm_config(vllm_config):
        # Model-parallel init must run inside the context — it reads the
        # current vllm_config to wire up DP/PP/TP/EP groups.
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

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
        print("      forward pass finite")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    # Defaults target the FLUX.2-klein-4B variant because that is the model
    # the Flux2KleinPipeline (with its hardcoded Qwen3 text encoder) is built
    # for. FLUX.2-dev uses a Mistral-3 text encoder and lives behind the
    # separate Flux2Pipeline class which does not yet have NVFP4 plumbing.
    parser.add_argument("--base", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--nvfp4", default="black-forest-labs/FLUX.2-klein-4b-nvfp4")
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

    _init_distributed_only()
    return _run_pipeline_under_vllm_config(args)


if __name__ == "__main__":
    raise SystemExit(main())
