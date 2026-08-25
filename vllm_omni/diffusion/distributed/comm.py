# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# DeepSpeed Team & Jiarui Fang
#  from https://github.com/feifeibear/long-context-attention/blob/main/yunchang/comm/all_to_all.py
import os
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

__all__ = ["all_to_all_4D", "all_to_all_5D", "SeqAllToAll4D", "SeqAllToAll5D", "RingComm"]

logger = init_logger(__name__)

_ULYSSES_TRANSPORT = os.getenv("VLLM_OMNI_ULYSSES_TRANSPORT", "nccl").lower()
if _ULYSSES_TRANSPORT not in {"nccl", "pitched", "packed"}:
    raise ValueError(
        f"VLLM_OMNI_ULYSSES_TRANSPORT must be one of 'nccl', 'pitched', or 'packed', got {_ULYSSES_TRANSPORT!r}"
    )
_FAST_ULYSSES_GROUPS: dict[tuple[int, str, int], Any] = {}
_FAST_ULYSSES_ALLOW_NON_NVLINK = os.getenv("VLLM_OMNI_FAST_ULYSSES_ALLOW_NON_NVLINK", "0").lower() in {
    "1",
    "true",
    "yes",
}


def _get_fast_ulysses_group(group, backend: str):
    if not current_omni_platform.is_cuda():
        raise RuntimeError(f"Ulysses transport {backend!r} requires CUDA")
    process_group = group if group is not None else dist.group.WORLD
    device_index = current_omni_platform.current_device()
    key = (id(process_group), backend, device_index)
    fast_group = _FAST_ULYSSES_GROUPS.get(key)
    if fast_group is None:
        try:
            from fast_ulysses import UlyssesGroup
        except ImportError as exc:
            raise RuntimeError(
                f"VLLM_OMNI_ULYSSES_TRANSPORT={backend} requires the optional "
                "fast-ulysses package built against this PyTorch/CUDA environment"
            ) from exc
        fast_group = UlyssesGroup(
            process_group=process_group,
            device=device_index,
            require_nvlink=backend == "pitched" and not _FAST_ULYSSES_ALLOW_NON_NVLINK,
            backend=backend,
        )
        _FAST_ULYSSES_GROUPS[key] = fast_group
        logger.info(
            "Initialized fast-ulysses transport backend=%s group=%s device=%d allow_non_nvlink=%s",
            backend,
            getattr(process_group, "group_name", "unknown"),
            device_index,
            _FAST_ULYSSES_ALLOW_NON_NVLINK,
        )
    return fast_group


def _fast_all_to_all_4d(input: Tensor, scatter_idx: int, gather_idx: int, group, use_sync: bool) -> Tensor:
    if (scatter_idx, gather_idx) == (2, 1):
        mode = 0
    elif (scatter_idx, gather_idx) == (1, 2):
        mode = 1
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")
    output = _get_fast_ulysses_group(group, _ULYSSES_TRANSPORT).all_to_all_4d(input, mode=mode)
    if use_sync:
        current_omni_platform.synchronize()
    return output


def all_to_all_4D(
    input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group (torch.distributed.ProcessGroup): torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if seq_world_size > 1 and _ULYSSES_TRANSPORT != "nccl":
        return _fast_all_to_all_4d(input, scatter_idx, gather_idx, group, use_sync)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs).transpose(0, 2).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_omni_platform.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)->
        #  (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_omni_platform.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -transpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        return all_to_all_4D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)


def all_to_all_5D(
    input: torch.tensor, scatter_idx: int = 3, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.tensor:
    """
    all-to-all for QKV
    forward (bs, seqlen/N, 3, hc, hs) -> (bs, seqlen, 3, hc/N, hs)

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group (torch.distributed.ProcessGroup): torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, 3, hc, hs)
    """
    assert input.dim() == 5, f"input must be 5D tensor, got {input.dim()} and shape {input.shape}"

    if dist.get_world_size(group) > 1 and _ULYSSES_TRANSPORT != "nccl":
        raise RuntimeError(
            f"VLLM_OMNI_ULYSSES_TRANSPORT={_ULYSSES_TRANSPORT} supports 4D Ulysses only; "
            "the selected model issued a 5D collective"
        )

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 3 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, 3, hc, hs) output: (bs, seqlen, 3, hc/P, hs)
        bs, shard_seqlen, t_cnt, hc, hs = input.shape

        assert t_cnt == 3
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, 3, hc, hs) -reshape-> (bs, seq_len/P, 3, P, hc/P, hs) -transpose(0,3)->
        #  (P, seq_len/P, 3, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen, 3, seq_world_size, shard_hc, hs).transpose(0, 3).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, 3, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, 3, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_omni_platform.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, 3, bs, shard_hc, hs)

        # (seq_len, 3, bs, hc/P, hs) -trans-> (bs, seq_len, 3, hc/P, hs)
        output = output.transpose(0, 2).transpose(1, 2).contiguous()

        return output.reshape(bs, seqlen, 3, shard_hc, hs).contiguous()
    elif scatter_idx == 1 and gather_idx == 3:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, _, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, 3, hc/P, hs) -reshape-> (bs, P, seq_len/P, 3, hc/P, hs) -transpose(0, 4)->
        # (hc/P, P, seqlen/P, 3, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, 3, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, 3, shard_hc, hs)
            .transpose(0, 4)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, 3, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_omni_platform.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, 3, bs, hs)

        # (hc, seqlen/N, bs, hs) -transpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 3).contiguous()

        return output.reshape(bs, shard_seqlen, 3, hc, hs).contiguous()
    else:
        raise RuntimeError("scatter_idx must be 1 or 3 and gather_idx must be 1 or 3")


class SeqAllToAll5D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int = 3,
        gather_idx: int = 1,
        use_sync: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync

        return all_to_all_5D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)


class RingComm:
    """Ring communication utility for Ring Attention P2P communication."""

    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops: list[Any] = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: torch.Tensor | None = None) -> torch.Tensor:
        # Ensure to_send is contiguous for P2P
        if not to_send.is_contiguous():
            to_send = to_send.contiguous()

        if recv_tensor is None:
            # Create a contiguous buffer for receiving
            res = torch.empty_like(to_send, memory_format=torch.contiguous_format)
            # print(f"send_recv: empty_like {to_send.shape}")
        else:
            res = recv_tensor
            if not res.is_contiguous():
                res = res.contiguous()

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []
