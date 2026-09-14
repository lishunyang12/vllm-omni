# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3 VSA prefix routing, tile layout, and learned compression gate."""

from __future__ import annotations

import functools
import math
import os

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (
    FastVideoVSABackend,
    FastVideoVSAImpl,
    _construct_variable_block_sizes,
    _get_gate_compress,
    _get_non_pad_index,
    _get_tile_partition_indices,
)

logger = init_logger(__name__)


# H3 needs an explicit per-query block map: prefix queries are dense, while
# video queries select prefix + top-k video tiles. The generic
# video_sparse_attn() entry point cannot express that contract.
if not hasattr(torch.ops.vllm_omni, "fastvideo_h3_vsa_bhsd"):

    @torch.library.custom_op("vllm_omni::fastvideo_h3_vsa_bhsd", mutates_args=())
    def _fastvideo_h3_vsa_bhsd_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        logical_blocks: int,
    ) -> torch.Tensor:
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # Official FastVideo's measured FastH3 route opts into this native
        # Blackwell forward. Wheels without the extension (including SM103
        # builds today) retain the corrected explicit-mask Triton route.
        if os.environ.get("FASTVIDEO_VSA_SM100A", "0") == "1":
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
            except (ImportError, RuntimeError) as exc:
                # Opting in explicitly and then silently getting a different
                # numeric path is worse than the slower route it lands on.
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

    @_fastvideo_h3_vsa_bhsd_op.register_fake
    def _(query, key, value, block_map, variable_block_sizes, logical_blocks):
        del key, value, block_map, variable_block_sizes, logical_blocks
        return torch.empty_like(query)


_fastvideo_h3_vsa_bhsd_op = torch.ops.vllm_omni.fastvideo_h3_vsa_bhsd


@functools.lru_cache(maxsize=32)
def _get_h3_tile_metadata(
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Official FastVideo H3 geometry: pure prefix chunks + 3-D video tiles."""
    block_size = (4, 4, 4)
    block_elements = 64
    prefix_len = sum(prefix_segments)
    prefix_sizes: list[int] = []
    for segment in prefix_segments:
        full, remainder = divmod(segment, block_elements)
        prefix_sizes.extend([block_elements] * full)
        if remainder:
            prefix_sizes.append(remainder)

    video_indices = _get_tile_partition_indices(video_shape, block_size, device) + prefix_len
    video_sizes = _construct_variable_block_sizes(video_shape, block_size, device)
    partition = torch.cat([torch.arange(prefix_len, device=device, dtype=torch.long), video_indices])
    sizes = torch.cat([torch.tensor(prefix_sizes, device=device, dtype=torch.int32), video_sizes.to(torch.int32)])
    non_pad = _get_non_pad_index(sizes, block_elements)
    untile = non_pad[torch.argsort(partition)]
    total = prefix_len + math.prod(video_shape)
    if int(sizes.sum()) != total or untile.numel() != total:
        raise ValueError(
            f"invalid H3 VSA geometry: prefix={prefix_segments}, video={video_shape}, "
            f"sizes_sum={int(sizes.sum())}, total={total}"
        )
    return partition, sizes, non_pad, untile, len(prefix_sizes), int(video_sizes.numel())


def _get_h3_layout(
    attn_metadata: AttentionMetadata | None,
) -> tuple[tuple[int, ...], tuple[int, int, int], int] | None:
    if attn_metadata is None or attn_metadata.video_layout is None:
        return None
    prefix = attn_metadata.extra.get("vsa_h3_prefix_segments")
    if not isinstance(prefix, (tuple, list)):
        return None
    target = next(
        (span for span in reversed(attn_metadata.video_layout.video_spans) if span.role == "target"),
        None,
    )
    if target is None:
        return None
    return tuple(int(x) for x in prefix if int(x) > 0), target.latent_grid, target.start


def _pool_h3_tiles(x: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    batch, seq_len, heads, dim = x.shape
    blocks = seq_len // 64
    pooled = x.view(batch, blocks, 64, heads, dim).sum(dim=2, dtype=torch.float32)
    pooled = pooled / sizes.view(1, -1, 1, 1).clamp_min(1)
    return pooled.permute(0, 2, 1, 3)


def _build_h3_block_map(
    scores: torch.Tensor,
    num_prefix_blocks: int,
    num_video_blocks: int,
    topk: int,
) -> torch.Tensor:
    """Prefix K/V are exempt and prefix queries stay dense, as in FastVideo."""
    keep_video = min(topk, num_video_blocks)
    if keep_video == num_video_blocks:
        return torch.ones_like(scores, dtype=torch.bool)
    block_map = torch.zeros_like(scores, dtype=torch.bool)
    indices = scores[..., num_prefix_blocks:].topk(keep_video, dim=-1).indices + num_prefix_blocks
    block_map.scatter_(-1, indices, True)
    block_map[..., :num_prefix_blocks] = True
    block_map[:, :, :num_prefix_blocks, :] = True
    return block_map


class MiniMaxH3VSAImpl(FastVideoVSAImpl):
    """Apply H3 tile64 routing after shared parallel attention dispatch."""

    def _forward_h3(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        layout = _get_h3_layout(attn_metadata)
        if layout is None:
            raise ValueError("incomplete VSA-H3 layout metadata")
        prefix_segments, video_shape, target_start = layout
        if sum(prefix_segments) != target_start:
            raise ValueError(f"VSA-H3 prefix segments sum to {sum(prefix_segments)}, target starts at {target_start}")
        expected = target_start + math.prod(video_shape)
        if query.shape[1] != expected:
            raise ValueError(f"VSA-H3 layout has {expected} rows but attention received {query.shape[1]}")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"VSA-H3 requires fp16/bf16 tensors, got {query.dtype}")
        gate = _get_gate_compress(attn_metadata)
        if gate is not None:
            # H3 pads the packed document to 64 rows, while VSA operates on
            # the valid prefix. Validate/slice before launching attention so a
            # metadata error can never trigger an unsafe dense fallback after
            # an asynchronous custom kernel.
            if gate.shape[0] != query.shape[0] or gate.shape[2:] != query.shape[2:] or gate.shape[1] < query.shape[1]:
                raise ValueError(f"gate_compress shape {gate.shape} cannot cover query shape {query.shape}")
            gate = gate[:, : query.shape[1]]

        partition, sizes, non_pad, untile, prefix_blocks, video_blocks = _get_h3_tile_metadata(
            prefix_segments, video_shape, query.device
        )
        logical_blocks = int(sizes.numel())
        # The native sm100a kernel assigns pairs of query blocks to CTAs. Its
        # contract requires an even block count; the synthetic partner is
        # transport-only and is removed before returning.
        pair_pad = logical_blocks % 2
        kernel_blocks = logical_blocks + pair_pad
        target_shape = (query.shape[0], kernel_blocks * 64, query.shape[2], query.shape[3])
        q_tiled = torch.zeros(target_shape, device=query.device, dtype=query.dtype)
        k_tiled = torch.zeros_like(q_tiled)
        v_tiled = torch.zeros_like(q_tiled)
        q_tiled[:, non_pad] = query[:, partition]
        k_tiled[:, non_pad] = key[:, partition]
        v_tiled[:, non_pad] = value[:, partition]

        q_pool = _pool_h3_tiles(q_tiled[:, : logical_blocks * 64], sizes)
        k_pool = _pool_h3_tiles(k_tiled[:, : logical_blocks * 64], sizes)
        scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) * self.softmax_scale
        block_map = _build_h3_block_map(scores, prefix_blocks, video_blocks, self.topk)
        kernel_sizes = sizes
        if pair_pad:
            block_map = torch.nn.functional.pad(block_map, (0, 1, 0, 1), value=False)
            kernel_sizes = torch.nn.functional.pad(sizes, (0, 1), value=0)

        logger.info_once(
            "FASTVIDEO_VSA H3 routing: seq_len=%d, prefix_segments=%s, video_shape=%s, "
            "prefix_blocks=%d, video_blocks=%d, topk=%d, kernel_blocks=%d",
            query.shape[1],
            prefix_segments,
            video_shape,
            prefix_blocks,
            video_blocks,
            min(self.topk, video_blocks),
            kernel_blocks,
        )
        output = _fastvideo_h3_vsa_bhsd_op(
            q_tiled.contiguous(),
            k_tiled.contiguous(),
            v_tiled.contiguous(),
            block_map.contiguous(),
            kernel_sizes.contiguous(),
            logical_blocks,
        )[:, : logical_blocks * 64]

        if gate is not None:
            gate_tiled = torch.zeros_like(q_tiled[:, : logical_blocks * 64])
            gate_tiled[:, non_pad] = gate[:, partition]
            v_pool = _pool_h3_tiles(v_tiled[:, : logical_blocks * 64], sizes)
            compressed = torch.matmul(torch.softmax(scores, dim=-1), v_pool)
            compressed = compressed.permute(0, 2, 1, 3).to(output.dtype)
            output = (
                output.view(output.shape[0], logical_blocks, 64, output.shape[2], output.shape[3])
                + compressed.unsqueeze(2)
                * gate_tiled.view(gate_tiled.shape[0], logical_blocks, 64, gate_tiled.shape[2], gate_tiled.shape[3])
            ).view_as(output)
        return output[:, untile].contiguous()

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if _get_h3_layout(attn_metadata) is None:
            return super().forward_cuda(query, key, value, attn_metadata)

        original_query, original_key, original_value = query, key, value
        original_seq_len = query.shape[1]
        valid_seq_len = original_seq_len
        if attn_metadata is not None and attn_metadata.packed_padding is not None:
            valid_seq_len = attn_metadata.packed_padding.q_length
            if attn_metadata.packed_padding.kv_length != valid_seq_len:
                return self._fallback(
                    original_query, original_key, original_value, attn_metadata, "packed Q/KV lengths must match"
                )
            query = query[:, :valid_seq_len]
            key = key[:, :valid_seq_len]
            value = value[:, :valid_seq_len]

        try:
            output = self._forward_h3(query, key, value, attn_metadata)
            if valid_seq_len == original_seq_len:
                return output
            restored = torch.zeros_like(original_query)
            restored[:, :valid_seq_len] = output
            return restored
        except Exception as exc:
            # A CUDA fault poisons the process context; attempting SDPA
            # afterwards obscures the original kernel failure and cannot
            # recover the request.
            if isinstance(exc, torch.AcceleratorError):
                raise
            if not self.fallback_on_error:
                raise
            return self._fallback(
                original_query, original_key, original_value, attn_metadata, f"VSA-H3 kernel failed: {exc}"
            )


class MiniMaxH3VSABackend(FastVideoVSABackend):
    @staticmethod
    def get_impl_cls() -> type[MiniMaxH3VSAImpl]:
        return MiniMaxH3VSAImpl
