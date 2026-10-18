# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared pooling, routing and provider calls for explicit block-sparse attention."""

import os

import torch
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def _flashinfer_sm120_tile128(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    logical_blocks: int,
) -> torch.Tensor:
    """Execute a logical tile-128 mask with SM120's 64-token compute blocks.

    Both query halves inherit the same selected KV tiles. Each selected KV
    tile contributes both halves to ONE attention reduction, preserving its
    softmax denominator. Pooling, selection and compression remain tile-128.
    """
    try:
        from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_fwd
    except ImportError as exc:
        raise RuntimeError(
            "FastH3 Omni on SM120 requires FlashInfer with bsa_attn_sm120_blk64_fwd; "
            "install a FlashInfer build that provides this API"
        ) from exc

    sizes = variable_block_sizes[:logical_blocks].to(torch.int32)
    sizes64 = torch.stack((sizes.clamp(max=64), (sizes - 64).clamp(0, 64)), dim=-1).flatten()
    mask64 = block_map[..., :logical_blocks, :logical_blocks].repeat_interleave(2, -2).repeat_interleave(2, -1)
    # Partial tiles can have an entirely empty second half. Do not route it.
    mask64 &= sizes64.view(1, 1, 1, -1) > 0
    counts = mask64.sum(-1).to(torch.int32).contiguous()
    num_blocks = logical_blocks * 2
    ids = torch.arange(num_blocks, device=query.device, dtype=torch.int32)
    indices = torch.where(mask64, ids, num_blocks).sort(dim=-1).values
    indices = torch.where(indices == num_blocks, -1, indices).contiguous()

    length = logical_blocks * 128
    out, _ = bsa_attn_sm120_blk64_fwd(
        query[:, :length].contiguous(),
        key[:, :length].contiguous(),
        value[:, :length].contiguous(),
        indices,
        num_blocks,
        block_sizes=sizes64,
        q2k_block_nums=counts,
    )
    if out.shape[1] != query.shape[1]:
        out = torch.nn.functional.pad(out, (0, 0, 0, 0, 0, query.shape[1] - out.shape[1]))
    return out


if not hasattr(torch.ops.vllm_omni, "fastvideo_block_sparse_attn_bshd"):

    @torch.library.custom_op("vllm_omni::fastvideo_block_sparse_attn_bshd", mutates_args=())
    def fastvideo_block_sparse_attn_bshd(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        logical_blocks: int,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Run explicit block-sparse attention on BSHD tensors in 64/128-row blocks.

        Inputs and the block map may include transport-only blocks after
        ``logical_blocks``. On SM120/SM121, FlashInfer executes the equivalent
        64-token decomposition and the wrapper restores transport rows as zeros.
        Pooling and routing retain the checkpoint's logical block size.
        """
        if block_size == 128 and query.is_cuda:
            capability = current_omni_platform.get_device_capability(query.device.index or 0)
            if capability is not None and capability.major == 12:
                logger.info_once("FastH3 VSA: FlashInfer SM120 block64 compute with logical block128 routing.")
                return _flashinfer_sm120_tile128(query, key, value, block_map, variable_block_sizes, logical_blocks)

        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # Prefer the explicitly selected native provider when it supports
        # these tensors; retain the existing Triton provider otherwise.
        if block_size not in (64, 128):
            raise ValueError(f"unsupported FastVideo block size: {block_size}")
        if block_size == 128 or os.environ.get("FASTVIDEO_VSA_SM100A", "0") == "1":
            try:
                from fastvideo_kernel import block_sparse_attn_sm100a
                from fastvideo_kernel.triton_kernels.index import map_to_index

                if block_sparse_attn_sm100a.is_supported(q, variable_block_sizes):
                    q2k_idx, q2k_num = map_to_index(block_map)
                    out, _ = block_sparse_attn_sm100a.block_sparse_attn_sm100a(
                        q,
                        k,
                        v,
                        q2k_idx.to(torch.int32).contiguous(),
                        q2k_num.to(torch.int32).contiguous(),
                        variable_block_sizes.to(torch.int32).contiguous(),
                        need_lse=False,
                    )
                    return out.transpose(1, 2).contiguous()
                if block_size == 128:
                    raise RuntimeError(
                        "VSA tile 128 requires FastVideo's sm_100/sm_103 kernel or FlashInfer's SM120 kernel"
                    )
            except torch.AcceleratorError:
                # A device fault can poison the context; do not launch another kernel.
                raise
            except (ImportError, RuntimeError) as exc:
                if block_size == 128:
                    raise RuntimeError("FastH3 Omni requires a supported Blackwell sparse attention kernel") from exc
                logger.warning_once(
                    "FASTVIDEO_VSA_SM100A=1 requested but the native Blackwell forward is "
                    "unavailable (%s); using the Triton block-sparse route instead.",
                    exc,
                )

        from fastvideo_kernel.block_sparse_attn import block_sparse_attn

        logical_len = logical_blocks * 64
        out, _ = block_sparse_attn(
            q[:, :, :logical_len].contiguous(),
            k[:, :, :logical_len].contiguous(),
            v[:, :, :logical_len].contiguous(),
            block_map[..., :logical_blocks, :logical_blocks].contiguous(),
            variable_block_sizes[:logical_blocks].to(torch.int32).contiguous(),
        )
        out = out.transpose(1, 2).contiguous()
        if out.shape[1] != query.shape[1]:
            out = torch.nn.functional.pad(out, (0, 0, 0, 0, 0, query.shape[1] - out.shape[1]))
        return out

    @fastvideo_block_sparse_attn_bshd.register_fake
    def _(query, key, value, block_map, variable_block_sizes, logical_blocks, block_size=64):
        del key, value, block_map, variable_block_sizes, logical_blocks, block_size
        return torch.empty_like(query)


fastvideo_block_sparse_attn_bshd = torch.ops.vllm_omni.fastvideo_block_sparse_attn_bshd


def mean_pool_tiles(x: torch.Tensor, sizes: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool zero-padded BSHD tiles with FP32 accumulation.

    ``sizes`` gives the number of valid rows in each tile.
    """
    batch, seq_len, heads, dim = x.shape
    blocks = seq_len // block_size
    pooled = x.view(batch, blocks, block_size, heads, dim).sum(dim=2, dtype=torch.float32)
    pooled = pooled / sizes.view(1, -1, 1, 1).clamp_min(1)
    return pooled.permute(0, 2, 1, 3)


def build_prefix_dense_block_map(
    scores: torch.Tensor,
    num_prefix_blocks: int,
    num_sparse_blocks: int,
    topk: int,
) -> torch.Tensor:
    """Keep prefix queries/keys dense and select top-k non-prefix keys per row."""
    keep_sparse = min(topk, num_sparse_blocks)
    if keep_sparse == num_sparse_blocks:
        return torch.ones_like(scores, dtype=torch.bool)
    block_map = torch.zeros_like(scores, dtype=torch.bool)
    indices = scores[..., num_prefix_blocks:].topk(keep_sparse, dim=-1).indices + num_prefix_blocks
    block_map.scatter_(-1, indices, True)
    block_map[..., :num_prefix_blocks] = True
    block_map[:, :, :num_prefix_blocks, :] = True
    return block_map
