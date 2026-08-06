# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-demand module staging backed by an immutable pinned CPU snapshot."""

from __future__ import annotations

from itertools import chain
from typing import Any

import torch
from torch import nn

from vllm_omni.platforms import current_omni_platform

from .tensor_utils import make_offload_placeholder, set_tensor_storage


class PinnedModuleStager:
    """Stage an immutable module without copying device weights back to CPU.

    ``nn.Module.to("cpu")`` performs a device-to-host copy for every parameter
    after each forward. In inference the weights are immutable, so retain one
    pinned CPU master instead. ``load`` materializes fresh device tensors from
    that master; ``offload`` only rebinds the Parameters/Buffers to it and
    releases the device allocations.
    """

    def __init__(
        self,
        module: nn.Module,
        device: torch.device,
        *,
        pin_memory: bool = True,
        copy_stream: Any | None = None,
    ) -> None:
        self.device = device
        self.copy_stream = copy_stream or current_omni_platform.Stream()
        self.loaded = False
        self._entries: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._device_tensors: list[torch.Tensor] = []

        # parameters()/buffers() remove duplicate objects by default, which
        # preserves tied weights and buffer aliases when their storage moves.
        for target in chain(module.parameters(), module.buffers()):
            master = target.detach()
            if master.device.type != "cpu":
                master = master.to("cpu")
            if pin_memory and not master.is_pinned():
                master = master.pin_memory()
            set_tensor_storage(target, master)
            self._entries.append((target, master))

    def load(self) -> None:
        if self.loaded:
            return

        # Allocate on the compute stream so the caching allocator can reuse
        # memory released by the preceding encoder/DiT stage.
        device_tensors = [torch.empty_like(master, device=self.device) for _, master in self._entries]
        compute_stream = current_omni_platform.current_stream()
        self.copy_stream.wait_stream(compute_stream)
        ready = current_omni_platform.Event()
        with current_omni_platform.stream(self.copy_stream):
            for device_tensor, (_, master) in zip(device_tensors, self._entries):
                device_tensor.copy_(master, non_blocking=master.is_pinned())
            ready.record(self.copy_stream)

        for device_tensor, (target, _) in zip(device_tensors, self._entries):
            set_tensor_storage(target, device_tensor)
        compute_stream.wait_event(ready)
        self._device_tensors = device_tensors
        self.loaded = True

    def offload(self) -> None:
        if not self.loaded:
            return

        # The module has completed on the compute stream. Synchronize once at
        # the stage boundary, then release HBM by rebinding to the unchanged
        # host masters. No device-to-host transfer is necessary.
        current_omni_platform.synchronize()
        for target, master in self._entries:
            set_tensor_storage(target, master)
        self._device_tensors.clear()
        self.loaded = False
        current_omni_platform.empty_cache()


def shard_and_pin_tensors(
    params: dict[str, nn.Parameter],
    buffers: dict[str, torch.Tensor],
    *,
    shard_count: int,
    shard_rank: int,
    pin_memory: bool,
) -> tuple[dict[torch.dtype, torch.Tensor], dict[torch.dtype, list[dict[str, Any]]]]:
    """Flatten tensors by dtype and retain one pinned, equal-sized shard.

    Metadata offsets refer to the full flattened buffer so callers can either
    reconstruct it with an AllGather or, with shard_count=1, restore the
    tensors directly. Source tensors are rebound to empty placeholders after
    their local shard has been copied.
    """
    if shard_count < 1:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if not 0 <= shard_rank < shard_count:
        raise ValueError(f"shard_rank {shard_rank} is outside [0, {shard_count})")

    dtype_grouped: dict[torch.dtype, dict[str, torch.Tensor]] = {}
    for name, tensor in chain(params.items(), buffers.items()):
        dtype_grouped.setdefault(tensor.dtype, {})[name] = tensor

    cpu_shards: dict[torch.dtype, torch.Tensor] = {}
    dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}
    for dtype, named_tensors in dtype_grouped.items():
        tensors_with_local: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for name, tensor in named_tensors.items():
            local_tensor = tensor.to_local() if hasattr(tensor, "to_local") else tensor
            mmap_transform = getattr(tensor, "mmap_weight_transform", None)
            if callable(mmap_transform) and getattr(tensor, "mmap_weight_transform_pending", False):
                local_tensor = mmap_transform(local_tensor)
            tensors_with_local.append((name, tensor, local_tensor))

        total_numel = sum(local.numel() for _, _, local in tensors_with_local)
        shard_size = (total_numel + shard_count - 1) // shard_count
        shard_start = shard_rank * shard_size
        shard_end = min(shard_start + shard_size, total_numel)
        shard = torch.zeros(shard_size, dtype=dtype, device="cpu")
        metadata: list[dict[str, Any]] = []

        current_offset = 0
        for name, original_tensor, local_tensor in tensors_with_local:
            numel = local_tensor.numel()
            metadata.append(
                {
                    "name": name,
                    "offset": current_offset,
                    "numel": numel,
                    "shape": local_tensor.shape,
                }
            )

            overlap_start = max(current_offset, shard_start)
            overlap_end = min(current_offset + numel, shard_end)
            if overlap_start < overlap_end:
                src_start = overlap_start - current_offset
                src_end = overlap_end - current_offset
                dst_start = overlap_start - shard_start
                dst_end = overlap_end - shard_start
                shard[dst_start:dst_end].copy_(local_tensor.flatten()[src_start:src_end])

            set_tensor_storage(original_tensor, make_offload_placeholder(original_tensor))
            current_offset += numel

        if pin_memory:
            shard = shard.pin_memory()
        cpu_shards[dtype] = shard
        dtype_metadata[dtype] = metadata

    return cpu_shards, dtype_metadata


class PinnedResidentLayerGroup:
    """Keep selected, rank-local layers in pinned memory between requests."""

    def __init__(
        self,
        blocks: list[nn.Module],
        device: torch.device,
        copy_stream: Any,
        pin_memory: bool,
    ) -> None:
        self.device = device
        self.copy_stream = copy_stream
        self.loaded = False
        self._states: list[dict[str, Any]] = []
        self._device_buffers: list[dict[torch.dtype, torch.Tensor]] = []

        for block in blocks:
            params = dict(block.named_parameters())
            buffers = dict(block.named_buffers())
            cpu_shards, metadata = shard_and_pin_tensors(
                params,
                buffers,
                shard_count=1,
                shard_rank=0,
                pin_memory=pin_memory,
            )
            self._states.append(
                {
                    "targets": {**params, **buffers},
                    "cpu_shards": cpu_shards,
                    "metadata": metadata,
                }
            )

    def load(self) -> None:
        if self.loaded:
            return

        device_buffers = [
            {
                dtype: torch.empty(cpu_shard.shape, dtype=dtype, device=self.device)
                for dtype, cpu_shard in state["cpu_shards"].items()
            }
            for state in self._states
        ]

        compute_stream = current_omni_platform.current_stream()
        self.copy_stream.wait_stream(compute_stream)
        ready = current_omni_platform.Event()
        with current_omni_platform.stream(self.copy_stream):
            for state, block_buffers in zip(self._states, device_buffers):
                for dtype, cpu_shard in state["cpu_shards"].items():
                    block_buffers[dtype].copy_(cpu_shard, non_blocking=cpu_shard.is_pinned())
            ready.record(self.copy_stream)

        for state, block_buffers in zip(self._states, device_buffers):
            for dtype, metadata in state["metadata"].items():
                device_buffer = block_buffers[dtype]
                for tensor_metadata in metadata:
                    offset = tensor_metadata["offset"]
                    numel = tensor_metadata["numel"]
                    set_tensor_storage(
                        state["targets"][tensor_metadata["name"]],
                        device_buffer[offset : offset + numel].view(tensor_metadata["shape"]),
                    )

        compute_stream.wait_event(ready)
        self._device_buffers = device_buffers
        self.loaded = True

    def offload(self) -> None:
        if not self.loaded:
            return

        current_omni_platform.synchronize()
        for state in self._states:
            for target in state["targets"].values():
                set_tensor_storage(target, make_offload_placeholder(target))
        self._device_buffers.clear()
        self.loaded = False
        current_omni_platform.empty_cache()


__all__ = ["PinnedModuleStager", "PinnedResidentLayerGroup", "shard_and_pin_tensors"]
