# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded host-memory staging for distributed layerwise offload."""

from __future__ import annotations

import errno
import mmap
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from .tensor_utils import dtype_size

_FALLOCATE_UNSUPPORTED_ERRNOS = {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}


class FileBackedShardStore:
    """Store finalized rank-local shards in private file-backed mappings.

    The cache files hold the persistent copy.  Their pages may be cached by
    the OS, but unlike anonymous or pinned tensors those pages are reclaimable
    under host-memory pressure.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix=f"vllm-omni-dlo-{os.getpid()}-",
            dir=root_path,
        )
        self.cache_dir = Path(self._temp_dir.name)
        self.allocated_bytes = 0
        self._entries: dict[int, tuple[mmap.mmap, int, Path, int]] = {}
        self._next_file_id = 0
        self._closed = False

    def allocate(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("file-backed DLO shard store is closed")
        if numel <= 0:
            raise ValueError(f"file-backed DLO shard must contain elements, got {numel}")

        nbytes = numel * dtype_size(dtype)
        path = self.cache_dir / f"shard-{self._next_file_id:06d}.bin"
        self._next_file_id += 1
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, nbytes)
            if hasattr(os, "posix_fallocate"):
                try:
                    os.posix_fallocate(fd, 0, nbytes)
                except OSError as exc:
                    if exc.errno not in _FALLOCATE_UNSUPPORTED_ERRNOS:
                        raise
            mapping = mmap.mmap(fd, nbytes, access=mmap.ACCESS_WRITE)
        except Exception:
            os.close(fd)
            path.unlink(missing_ok=True)
            raise

        tensor = torch.frombuffer(mapping, dtype=dtype, count=numel)
        self._entries[id(tensor)] = (mapping, fd, path, nbytes)
        self.allocated_bytes += nbytes
        return tensor

    def finalize(self, tensor: torch.Tensor) -> None:
        """Write back a completed shard and make its clean pages reclaimable."""
        entry = self._entries.get(id(tensor))
        if entry is None:
            return
        mapping, fd, _, nbytes = entry
        mapping.flush()
        try:
            if hasattr(mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                mapping.madvise(mmap.MADV_DONTNEED)
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                os.posix_fadvise(fd, 0, nbytes, os.POSIX_FADV_DONTNEED)
        except (OSError, ValueError):
            pass

    def prefetch(self, shard_sets: Iterable[dict[torch.dtype, torch.Tensor]]) -> None:
        """Best-effort OS readahead for upcoming file-backed shards."""
        seen: set[int] = set()
        for shards in shard_sets:
            for tensor in shards.values():
                tensor_id = id(tensor)
                if tensor_id in seen:
                    continue
                seen.add(tensor_id)
                entry = self._entries.get(tensor_id)
                if entry is None:
                    continue
                mapping, fd, _, nbytes = entry
                try:
                    if hasattr(mapping, "madvise") and hasattr(mmap, "MADV_WILLNEED"):
                        mapping.madvise(mmap.MADV_WILLNEED)
                    elif hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_WILLNEED"):
                        os.posix_fadvise(fd, 0, nbytes, os.POSIX_FADV_WILLNEED)
                except (OSError, ValueError):
                    # Readahead is an optimization.  The following tensor copy
                    # remains the correctness path when the filesystem rejects it.
                    continue

    def close(self) -> None:
        if self._closed:
            return
        for mapping, fd, _, _ in self._entries.values():
            mapping.close()
            os.close(fd)
        self._entries.clear()
        self._temp_dir.cleanup()
        self._closed = True


class PinnedHostStagingPool:
    """A fixed-size ring of shared CPU staging buffers."""

    def __init__(
        self,
        buffers: list[dict[torch.dtype, torch.Tensor]],
        allocated_bytes: int,
    ) -> None:
        self.buffers = buffers
        self.allocated_bytes = allocated_bytes
        self.slot_events: list[Any | None] = [None] * len(buffers)
        self._next_slot = 0

    @classmethod
    def from_shard_sets(
        cls,
        shard_sets: Iterable[dict[torch.dtype, torch.Tensor]],
        *,
        pin_memory: bool,
        budget_bytes: int,
        buffer_count: int = 2,
    ) -> PinnedHostStagingPool:
        if buffer_count < 2:
            raise ValueError(f"dlo_pinned_staging_buffer_count must be >= 2, got {buffer_count}")

        max_numels: dict[torch.dtype, int] = {}
        for shards in shard_sets:
            for dtype, shard in shards.items():
                max_numels[dtype] = max(max_numels.get(dtype, 0), shard.numel())

        required_bytes = buffer_count * sum(numel * dtype_size(dtype) for dtype, numel in max_numels.items())
        if required_bytes > budget_bytes:
            raise ValueError(
                f"{buffer_count} pinned host staging slots require {required_bytes} bytes, "
                f"but dlo_host_memory_budget_gib allows {budget_bytes} bytes"
            )

        buffers: list[dict[torch.dtype, torch.Tensor]] = []
        for _ in range(buffer_count):
            slot = {
                dtype: torch.empty(numel, dtype=dtype, device="cpu", pin_memory=pin_memory)
                for dtype, numel in max_numels.items()
            }
            buffers.append(slot)
        return cls(buffers, required_bytes)

    def stage(
        self,
        sources: dict[torch.dtype, torch.Tensor],
    ) -> tuple[int, dict[torch.dtype, torch.Tensor]]:
        slot = self._next_slot
        self._next_slot = (slot + 1) % len(self.buffers)

        previous_event = self.slot_events[slot]
        if previous_event is not None:
            previous_event.synchronize()
            self.slot_events[slot] = None

        staged: dict[torch.dtype, torch.Tensor] = {}
        for dtype, source in sources.items():
            target = self.buffers[slot][dtype][: source.numel()]
            target.copy_(source)
            staged[dtype] = target
        return slot, staged

    def mark_in_flight(self, slot: int, event: Any) -> None:
        self.slot_events[slot] = event
