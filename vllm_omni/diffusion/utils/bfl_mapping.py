# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BFL (Black Forest Labs) ↔ diffusers weight name remapping for FLUX.2.

Used by both ``flux2`` (dev) and ``flux2_klein`` transformer implementations
to load checkpoints saved in the BFL native naming convention (the format
used by NVIDIA's FLUX.2-dev-NVFP4 release and by GGUF community quants).

The two transformer classes are independent but share the same diffusers
parameter naming, so the same mapping table works for both.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.model_executor.models.utils import WeightsMapper

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Static mapping tables (BFL → diffusers).
# ---------------------------------------------------------------------------

_BFL_TO_DIFFUSERS_PREFIX = {
    "single_blocks.": "single_transformer_blocks.",
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "final_layer.linear": "proj_out",
    "final_layer.adaLN_modulation.1": "norm_out.linear",
}

_BFL_TO_DIFFUSERS_DOUBLE_BLOCK = {
    "double_blocks.": "transformer_blocks.",
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
    # Fused QKV projections
    "img_attn.qkv": "attn.to_qkv",
    "txt_attn.qkv": "attn.add_kv_proj",
}

_BFL_TO_DIFFUSERS_SINGLE_BLOCK = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}

BFL_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_prefix=_BFL_TO_DIFFUSERS_PREFIX,
    orig_to_new_substr=_BFL_TO_DIFFUSERS_DOUBLE_BLOCK | _BFL_TO_DIFFUSERS_SINGLE_BLOCK,
)


# ---------------------------------------------------------------------------
# Streaming helpers used in Flux2 transformer load_weights().
# ---------------------------------------------------------------------------


def peek_bfl_format(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> tuple[bool, Iterable[tuple[str, torch.Tensor]]]:
    """Detect BFL checkpoint format by peeking the first few entries.

    Only consumes a handful of items from the iterator and chains them back
    so the caller still gets a complete stream. Avoids materializing the full
    state dict (which on FLUX.2-dev-NVFP4 would mean ~700+ tensor handles
    held in a Python list during load).
    """
    iterator = iter(weights)
    peeked: list[tuple[str, torch.Tensor]] = []
    is_bfl = False
    for _ in range(4):
        try:
            entry = next(iterator)
        except StopIteration:
            break
        peeked.append(entry)
        if entry[0].startswith(("double_blocks.", "single_blocks.")):
            is_bfl = True
            break

    return is_bfl, itertools.chain(peeked, iterator)


def apply_bfl_mapping(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Remap BFL checkpoint names to diffusers format and handle special cases.

    Special cases:
    - RMSNorm: BFL stores as ``.scale``, diffusers expects ``.weight``.
    - adaLN modulation final layer: BFL stores ``[scale, shift]`` along the
      output dim, diffusers expects ``[shift, scale]`` — the two halves are
      swapped here.
    """
    import torch  # local import; this helper is only invoked at load time

    for name, weight in BFL_WEIGHTS_MAPPER.apply(weights):
        if name.endswith(".scale"):
            name = name[: -len(".scale")] + ".weight"
        if name == "norm_out.linear.weight":
            shift, scale = weight.chunk(2, dim=0)
            weight = torch.cat([scale, shift], dim=0)
        yield name, weight
