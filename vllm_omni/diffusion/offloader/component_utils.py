# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared component-plan helpers for diffusion offload backends."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain
from operator import attrgetter
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from vllm.logger import init_logger

from .block_discovery import get_blocks_from_dit
from .config import DIT_COMPONENT, TEXT_ENCODER_COMPONENT
from .offload_plan import OffloadPlan
from .tensor_utils import set_tensor_storage

if TYPE_CHECKING:
    from .base import OffloadConfig
    from .module_collector import PipelineModules

logger = init_logger(__name__)


def get_encoder_block_groups(
    module: nn.Module,
    name: str,
    plan: OffloadPlan | None,
) -> list[nn.ModuleList]:
    """Resolve the streamable block lists declared for one encoder."""
    if plan is None:
        return []

    groups: list[nn.ModuleList] = []
    for block_path in plan.encoder_block_attrs.get(name, ()):
        try:
            blocks = attrgetter(block_path)(module)
        except AttributeError:
            logger.warning("Encoder offload path %s.%s was not found", name, block_path)
            continue
        if not isinstance(blocks, nn.ModuleList) or len(blocks) <= 1:
            logger.warning("Encoder offload path %s.%s is not a streamable block list", name, block_path)
            continue
        groups.append(blocks)
    return groups


def get_streamable_dit_blocks(
    module: nn.Module,
    name: str,
    config: OffloadConfig,
    device: torch.device,
    plan: OffloadPlan | None,
) -> tuple[list[str], list[nn.Module]] | None:
    """Resolve one selected DiT's blocks with shared compatibility behavior."""
    planned_attrs = None if plan is None else plan.block_attrs.get(name)
    block_attrs, blocks = get_blocks_from_dit(module, planned_attrs)
    if blocks:
        return block_attrs, blocks
    if config.components_explicit:
        raise ValueError(f"Selected DiT {name!r} has no streamable layerwise-offload blocks")
    logger.warning(
        "Target layers (blocks) not found. Skipping offloading on %s (%s)",
        name,
        module.__class__.__name__,
    )
    module.to(device)
    return None


def iter_streamable_dits(
    modules: PipelineModules,
    config: OffloadConfig,
    device: torch.device,
    plan: OffloadPlan | None,
) -> Iterator[tuple[str, nn.Module, list[str], list[nn.Module]]]:
    """Yield selected DiTs whose block metadata resolves successfully."""
    if not config.offloads(DIT_COMPONENT):
        return
    for name, module in zip(modules.dit_names, modules.dits):
        logger.info("Applying hooks on %s (%s)", name, module.__class__.__name__)
        resolved = get_streamable_dit_blocks(module, name, config, device, plan)
        if resolved is not None:
            block_attrs, blocks = resolved
            yield name, module, block_attrs, blocks


def move_non_block_state_to_device(
    module: nn.Module,
    block_groups: list[nn.ModuleList],
    device: torch.device,
) -> None:
    """Keep component state outside streamed block lists resident on device."""
    block_tensors = {
        id(tensor)
        for blocks in block_groups
        for block in blocks
        for tensor in chain(block.parameters(), block.buffers())
    }
    for tensor in chain(module.parameters(), module.buffers()):
        if id(tensor) in block_tensors:
            continue
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        if local.device != device:
            set_tensor_storage(tensor, local.to(device, non_blocking=True))


def validate_on_demand_component(module: nn.Module, name: str) -> None:
    """Require the explicit lifecycle used by pipeline-managed components."""
    if not callable(getattr(module, "load_to_device", None)) or not callable(getattr(module, "offload_to_cpu", None)):
        raise ValueError(
            f"Component {name!r} declares on-demand offload but must implement load_to_device() and offload_to_cpu()"
        )


def prepare_component(
    module: nn.Module,
    name: str,
    *,
    device: torch.device,
    stage_on_demand: bool,
    blockwise: bool,
    staged_components: list[nn.Module],
) -> None:
    """Stage a selected component or keep its non-streamed form resident."""
    if stage_on_demand:
        validate_on_demand_component(module, name)
        getattr(module, "offload_to_cpu")()
        staged_components.append(module)
        logger.info("Prepared %s for pipeline-managed staged offload", name)
    elif not blockwise:
        module.to(device)


def prepare_pipeline_components(
    modules: PipelineModules,
    config: OffloadConfig,
    plan: OffloadPlan | None,
    *,
    device: torch.device,
    staged_components: list[nn.Module],
    enable_encoder_blocks: Callable[[nn.Module, str, OffloadPlan | None], bool],
) -> None:
    """Apply the shared encoder/VAE/resident placement policy."""
    if plan is not None:
        for encoder, name in zip(modules.encoders, modules.encoder_names):
            if config.should_offload_encoder(name, plan) and name in plan.on_demand_component_paths:
                validate_on_demand_component(encoder, name)

    selected_encoder_ready = False
    for encoder, name in zip(modules.encoders, modules.encoder_names):
        selected = config.should_offload_encoder(name, plan)
        blockwise = selected and enable_encoder_blocks(encoder, name, plan)
        stage_on_demand = bool(selected and plan is not None and name in plan.on_demand_component_paths)
        prepare_component(
            encoder,
            name,
            device=device,
            stage_on_demand=stage_on_demand,
            blockwise=blockwise,
            staged_components=staged_components,
        )
        selected_encoder_ready = selected_encoder_ready or blockwise or stage_on_demand

    if config.components_explicit and config.offloads(TEXT_ENCODER_COMPONENT) and not selected_encoder_ready:
        raise ValueError(
            "Selected text_encoder layerwise offload requires a model-declared streamable or on-demand plan"
        )

    for vae, name in zip(modules.vaes, modules.vae_names):
        legacy_staged = not config.components_explicit and plan is not None and name in plan.on_demand_component_paths
        prepare_component(
            vae,
            name,
            device=device,
            stage_on_demand=legacy_staged,
            blockwise=False,
            staged_components=staged_components,
        )

    for name, module in zip(modules.resident_names, modules.resident_modules):
        try:
            module.to(device)
        except Exception as exc:
            logger.debug("Failed to move resident module %s to %s: %s", name, device, exc)

    if not config.offloads(DIT_COMPONENT):
        for dit in modules.dits:
            dit.to(device)
