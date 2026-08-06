# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative OffloadPlan for distributed layerwise offload.

Models can declare a static ``_offload_plan`` or return an instance-derived
plan from ``get_offload_plan()``. When present, it replaces heuristic block
discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn


@dataclass(frozen=True)
class OffloadPlan:
    """Optional declarative metadata for distributed layerwise offload.

    Pipelines can declare a static ``_offload_plan`` or implement
    ``get_offload_plan()`` when topology depends on the loaded partition.
    The offloader uses that metadata instead of heuristic block discovery,
    avoiding model-specific logic in the backend.

    If not declared, the offloader falls back to:
    1. ``_layerwise_offload_blocks_attrs`` on each DiT module class.
    2. Heuristic search for ``layers`` / ``blocks`` / ``h`` attributes.

    Attributes:
        on_demand_component_paths: Encoder or VAE paths whose pipeline
            lifecycle loads them only for their active stage.
        block_attrs: Maps DiT path → tuple of block-list attribute names.
            e.g. ``{"transformer": ("gen_layers",),
                    "transformer.language_model": ("layers",)}``
        offload_submodules: Maps child name → block-list attribute name,
            for large non-DiT submodules within a DiT that should be
            independently offloaded with their own hooks.
            e.g. ``{"context_encoder": "layers"}``
        resident_dit_paths: DiT paths whose leading blocks may be kept on the
            device when ``dlo_resident_layers`` is nonzero. Keeping this
            model-declared avoids applying a consumer-GPU tuning knob to
            auxiliary or dual DiTs unintentionally.
        encoder_block_attrs: Maps encoder paths to rank-local block-list paths.
            These blocks are streamed with ordinary layerwise hooks, never
            with the DiT AllGather group.
    """

    on_demand_component_paths: frozenset[str] = field(default_factory=frozenset)

    block_attrs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    offload_submodules: dict[str, str] = field(default_factory=dict)
    resident_dit_paths: frozenset[str] = field(default_factory=frozenset)
    encoder_block_attrs: dict[str, tuple[str, ...]] = field(default_factory=dict)


def get_offload_plan(pipeline: nn.Module) -> OffloadPlan | None:
    """Retrieve static or instance-derived offload metadata."""
    plan_factory = getattr(pipeline, "get_offload_plan", None)
    plan = plan_factory() if callable(plan_factory) else getattr(pipeline, "_offload_plan", None)
    if plan is not None and not isinstance(plan, OffloadPlan):
        raise TypeError(f"get_offload_plan() must return OffloadPlan or None, got {type(plan).__name__}")
    return plan


def supports_mmap_loading(pipeline: nn.Module) -> bool:
    """Whether the pipeline supports the direct DLO+AllGather mmap loader.

    The direct path requires checkpoint-key remapping and bypasses ordinary
    weight-loader callbacks. Rank-local DLO always uses the regular
    ``load_weights()`` path with mmap-backed source tensors.

    This gate is shared by ``diffusers_loader.py`` and
    ``DistributedLayerwiseOffloadBackend.enable()`` so the regular loader is
    skipped if and only if the backend replaces it.
    """
    supports_direct_mmap = getattr(
        pipeline,
        "_supports_allgather_mmap_loading",
        getattr(pipeline, "_supports_mmap_loading", True),
    )
    return bool(supports_direct_mmap) and any(
        callable(getattr(type(m), "_remap_ckpt_key", None)) for m in pipeline.modules()
    )
