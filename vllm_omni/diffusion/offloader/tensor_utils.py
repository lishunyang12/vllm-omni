# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared tensor utilities for distributed layerwise offload.

These helpers are used by both DistributedLayerwiseOffloadHook and
DistributedLayerwiseOffloadBackend, and can be reused by other
offload backends.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any

import torch
from torch.distributed.tensor import DTensor


@dataclass(frozen=True)
class TensorStorageSpec:
    """One tensor's target and physical-storage layout."""

    name: str
    target: torch.Tensor
    value: torch.Tensor
    storage_numel: int
    stride: tuple[int, ...]


def dtype_size(dtype: torch.dtype) -> int:
    """Return element size in bytes for a torch.dtype."""
    return torch.empty(1, dtype=dtype).element_size()


def group_named_tensors_by_dtype(
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
) -> dict[torch.dtype, list[tuple[str, torch.Tensor]]]:
    """Group parameters and buffers by dtype while preserving their order."""
    grouped: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in chain(params.items(), buffers.items()):
        grouped.setdefault(tensor.dtype, []).append((name, tensor))
    return grouped


def physical_storage_numel(tensor: torch.Tensor) -> int:
    """Return the storage span needed to preserve ``tensor``'s stride."""
    if tensor.numel() == 0:
        return 0
    return 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))


def flatten_physical_storage(
    tensor: torch.Tensor,
    storage_numel: int | None = None,
) -> torch.Tensor:
    """Pack a possibly strided tensor in physical storage order."""
    if tensor.is_contiguous():
        return tensor.flatten()
    storage_numel = physical_storage_numel(tensor) if storage_numel is None else storage_numel
    storage = torch.zeros(storage_numel, dtype=tensor.dtype, device=tensor.device)
    torch.as_strided(storage, size=tensor.shape, stride=tensor.stride()).copy_(tensor)
    return storage


def describe_tensor_storage(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    transforms: Mapping[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> list[TensorStorageSpec]:
    """Resolve local tensors and their transport-preserving layouts."""
    specs: list[TensorStorageSpec] = []
    for name, target in named_tensors:
        value = target.to_local() if hasattr(target, "to_local") else target
        transform = (transforms or {}).get(id(target))
        if callable(transform):
            value = transform(value)
        specs.append(
            TensorStorageSpec(
                name=name,
                target=target,
                value=value,
                storage_numel=physical_storage_numel(value),
                stride=value.stride(),
            )
        )
    return specs


def tensor_storage_metadata(spec: TensorStorageSpec, offset: int, *, include_device: bool = False) -> dict[str, Any]:
    """Build reconstruction metadata shared by offload transports."""
    metadata: dict[str, Any] = {
        "name": spec.name,
        "offset": offset,
        "numel": spec.storage_numel,
        "shape": spec.value.shape,
        "stride": spec.stride,
    }
    if include_device:
        metadata["device"] = spec.value.device
    return metadata


def is_dtensor(t: torch.Tensor) -> bool:
    """Check if tensor is a DTensor."""
    return isinstance(t, DTensor)


def set_tensor_storage(target: torch.Tensor, value: torch.Tensor) -> None:
    """Replace target's underlying storage with value (zero-copy)."""
    if is_dtensor(target):
        target._local_tensor = value
    else:
        target.data = value


def make_offload_placeholder(tensor: torch.Tensor) -> torch.Tensor:
    """Create a zero-element placeholder to free GPU memory."""
    if is_dtensor(tensor):
        local_shape = tuple(tensor.to_local().shape)
        return torch.empty(local_shape, device="meta", dtype=tensor.dtype)
    return torch.empty((0,), device=tensor.device, dtype=tensor.dtype)


def clear_tensor_storage(tensors: Iterable[torch.Tensor]) -> None:
    """Release tensor residency after all replacement backing is ready."""
    for tensor in tensors:
        set_tensor_storage(tensor, make_offload_placeholder(tensor))


def clear_block_storage(
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    ready_event: Any | None,
) -> None:
    """Wait for an outstanding transfer, then release one block's residency."""
    if ready_event is not None:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.current_stream().wait_event(ready_event)
    clear_tensor_storage(chain(params.values(), buffers.values()))


def is_materialized_tensor(t: torch.Tensor) -> bool:
    """Check if tensor holds real data (not meta or empty placeholder)."""
    if is_dtensor(t):
        local_t = t.to_local()
        return not local_t.is_meta
    return not t.is_meta and t.data.numel() > 0


def restore_tensor_storage(
    target: torch.Tensor,
    value: torch.Tensor,
    *,
    device: torch.device | str,
) -> None:
    """Detach ``target`` from an offloader-owned backing with a real copy."""
    restored = torch.empty_strided(
        value.shape,
        value.stride(),
        dtype=value.dtype,
        device=device,
    )
    restored.copy_(value)
    set_tensor_storage(target, restored)
