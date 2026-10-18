# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in FlashInfer PR #4876 all-RDMA Ulysses transport.

Each live Q/K/V/gate/output has its own registered slot. Slots are reused by
successive layers and requests; callers must consume them before the next
exchange into that slot. Close collectively before destroying process groups.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _resolve_process_group
from vllm.logger import init_logger

logger = init_logger(__name__)


def is_flashinfer_pcie_enabled() -> bool:
    backend = os.getenv("VLLM_OMNI_ULYSSES_A2A_BACKEND", "nccl").strip().lower()
    if backend not in ("nccl", "flashinfer-pcie"):
        raise ValueError(f"Unsupported VLLM_OMNI_ULYSSES_A2A_BACKEND={backend!r}")
    return backend == "flashinfer-pcie"


def ensure_flashinfer_pcie_available() -> None:
    try:
        from flashinfer.comm import UlyssesCommunicator
        from flashinfer.comm.ulysses import missing_ulysses_pcie_dependencies
    except ImportError as exc:
        raise RuntimeError("flashinfer-pcie requires FlashInfer PR #4876") from exc
    missing = missing_ulysses_pcie_dependencies()
    if missing:
        raise RuntimeError("FlashInfer PR #4876 dependencies missing: " + ", ".join(missing))
    for name in ("allocate_output", "input_buffer", "scatter_heads", "gather_heads", "close"):
        if not hasattr(UlyssesCommunicator, name):
            raise RuntimeError(f"FlashInfer PR #4876 method missing: {name}")
    if os.getenv("FLASHINFER_ULYSSES_PCIE_ROUTE") != "rdma":
        raise ValueError("flashinfer-pcie requires FLASHINFER_ULYSSES_PCIE_ROUTE=rdma")


def _shape(x: torch.Tensor, op: str, world_size: int) -> tuple[int, int, int, int]:
    if x.ndim != 4 or x.shape[0] != 1 or any(size <= 0 for size in x.shape):
        raise ValueError("FlashInfer all-RDMA Ulysses requires positive [1,S,H,D] operands")
    batch, seq, heads, dim = x.shape
    if op == "scatter_heads":
        if heads % world_size:
            raise ValueError("Ulysses heads must be divisible by world size")
        return batch, seq * world_size, heads // world_size, dim
    if op == "gather_heads":
        if seq % world_size:
            raise ValueError("Ulysses sequence must be divisible by world size")
        return batch, seq // world_size, heads * world_size, dim
    raise ValueError(f"Unsupported Ulysses operation: {op}")


@dataclass
class _State:
    group: Any
    world_size: int
    capacity: int
    communicator: Any
    outputs: dict = field(default_factory=dict)

    def exchange(self, x: torch.Tensor, op: str, slot: str) -> torch.Tensor:
        shape = _shape(x, op, self.world_size)
        if x.nbytes > self.capacity:
            raise ValueError(f"RDMA operand {x.nbytes} exceeds capacity {self.capacity}")
        key = (slot, op, x.dtype)
        if key not in self.outputs:
            # Registration validates only geometry, never reads the prototype.
            prototype = x if x.is_contiguous() else torch.empty_like(x, memory_format=torch.contiguous_format)
            self.outputs[key] = self.communicator.allocate_output(prototype, op=op, dtype=x.dtype)
        strides = (shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1)
        out = self.outputs[key].as_strided(shape, strides)
        source = x
        if not x.is_contiguous():
            source = self.communicator.input_buffer(out, x.shape)
            source.copy_(x)
        return getattr(self.communicator, op)(source, out=out, dtype=x.dtype)


_STATES: dict[tuple[str, torch.device], _State] = {}


def _exchange(x: torch.Tensor, group_name: str, world_size: int, op: str, slot: str) -> torch.Tensor:
    _shape(x, op, world_size)
    if world_size not in (2, 4, 8) or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("This FlashInfer RDMA integration requires 2/4/8 CUDA ranks and BF16 operands")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FlashInfer RDMA requires eager execution; CUDA graph capture is unsupported")
    group = _resolve_process_group(group_name)
    if dist.get_world_size(group) != world_size:
        raise ValueError("FlashInfer Ulysses process-group size mismatch")
    key = (group_name, x.device)
    state = _STATES.get(key)
    if state is None:
        ensure_flashinfer_pcie_available()
        capacity = int(os.getenv("VLLM_OMNI_FLASHINFER_ULYSSES_MAX_BYTES", str(x.nbytes)))
        if capacity < x.nbytes or capacity <= 0:
            raise ValueError(f"RDMA capacity {capacity} cannot hold operand {x.nbytes}")
        from flashinfer.comm import UlyssesCommunicator

        communicator = UlyssesCommunicator(group, max_bytes=capacity, dtype=x.dtype, backend="pcie", device=x.device)
        if communicator.transport != "rdma":
            observed = communicator.transport
            communicator.close()
            raise RuntimeError(f"Required FlashInfer RDMA route unavailable: transport={observed!r}")
        state = _State(group, world_size, capacity, communicator)
        _STATES[key] = state
        logger.info(
            "FlashInfer PR4876 Ulysses armed: rank=%d group=%s world=%d transport=rdma capacity=%d",
            dist.get_rank(group),
            group_name,
            world_size,
            capacity,
        )
    elif state.group is not group:
        raise RuntimeError("Process-group identity changed for a live RDMA communicator")
    return state.exchange(x, op, slot)


@torch.library.custom_op("vllm_omni_flashinfer_ulysses::scatter_heads", mutates_args=(), device_types="cuda")
def flashinfer_ulysses_qkv_fwd(x: torch.Tensor, group_name: str, world_size: int, slot: str) -> torch.Tensor:
    if slot not in ("q", "k", "v", "g"):
        raise ValueError(f"Unsupported Ulysses operand slot: {slot}")
    return _exchange(x, group_name, world_size, "scatter_heads", slot)


@flashinfer_ulysses_qkv_fwd.register_fake
def _(x: torch.Tensor, group_name: str, world_size: int, slot: str) -> torch.Tensor:
    return x.new_empty(_shape(x, "scatter_heads", world_size))


@torch.library.custom_op("vllm_omni_flashinfer_ulysses::gather_heads", mutates_args=(), device_types="cuda")
def flashinfer_ulysses_o_rev(x: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    return _exchange(x, group_name, world_size, "gather_heads", "o")


@flashinfer_ulysses_o_rev.register_fake
def _(x: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    return x.new_empty(_shape(x, "gather_heads", world_size))


def clear_flashinfer_ulysses_communicators() -> None:
    for key in sorted(_STATES, key=lambda item: (item[0], str(item[1]))):
        state = _STATES[key]
        state.communicator.close()
        state.outputs.clear()
        del _STATES[key]
