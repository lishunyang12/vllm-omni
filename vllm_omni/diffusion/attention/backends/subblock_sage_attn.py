# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Portions derived from the SGLang project (https://github.com/sgl-project/sglang, Apache-2.0),
# Copyright contributors to the SGLang project.
"""Opt-in SubBlock sparse + Sage attention for MiniMax-H3 on SM120 (RTX PRO 6000 Blackwell).

Training-free block sparsity: ``SubBlockRouter`` pools Q/K into sub-blocks, scores 64-token block pairs with a
log-sum-exp, keeps the top ``1 - sparsity`` key blocks per query block, and FlashInfer's CAKE Sage kernel
(INT8 Q/K, FP8 V, 64x64 blocks) executes the selected blocks. Query blocks that hold non-video rows (text, visual
conditions, audio) stay dense. Everything else (cross attention, short sequences, the first ``start_step`` denoise
steps, layers in ``skip_layers``) runs the platform's exact BF16 attention.

The output is **lossy**: sparsity changes the attention result and Sage quantization adds a little more. The backend
changes nothing unless it is selected explicitly, and only SM120 is supported.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

BLOCK = 64
# Routing metadata (heads_per_call * query_blocks * key_blocks) above this is split over head groups.
MAX_ROUTED_BLOCK_PAIRS = 2**29


def build_mixed_plan(
    index: torch.Tensor, topk: int, num_k_blocks: int, sparse_query_blocks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query-block key-block lists for CAKE: sparse rows keep their routed ``topk``, the others all blocks.

    ``index`` is ``[B, H, Gq, topk]``; ``sparse_query_blocks`` is a ``[Gq]`` bool. Returns the ``[B, H, Gq, Gk]``
    index tensor (rows padded with the remaining block ids, which the per-row count masks off) and the
    ``[B, H, Gq]`` int32 counts.
    """
    full = torch.arange(num_k_blocks, device=index.device, dtype=index.dtype).view(1, 1, 1, num_k_blocks)
    mixed = full.expand(*index.shape[:-1], num_k_blocks).clone()
    mixed[..., :topk] = torch.where(sparse_query_blocks.view(1, 1, -1, 1), index, mixed[..., :topk])
    counts = (
        torch.where(sparse_query_blocks.view(1, 1, -1), topk, num_k_blocks).expand(*index.shape[:-1]).to(torch.int32)
    )
    return mixed, counts


def first_sparse_query_block(dense_prefix_rows: int) -> int:
    """First 64-row query block that holds no prefix (text/condition/audio) row; earlier blocks stay dense."""
    return -(-int(dense_prefix_rows) // BLOCK)


def sparse_query_block_mask(live_rows: int, dense_prefix_rows: int, device: torch.device) -> torch.Tensor:
    blocks = -(-int(live_rows) // BLOCK)
    return torch.arange(blocks, device=device) >= first_sparse_query_block(dense_prefix_rows)


def _is_sm120() -> bool:
    return torch.cuda.is_available() and current_omni_platform.get_device_capability() == (12, 0)


class SubBlockSageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = False

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        # Runs on the valid prefix of a [real, pad] packing and restores zero pad rows, like FASTVIDEO_VSA.
        return current_omni_platform.is_cuda()

    @classmethod
    def validate_available(cls) -> None:
        if importlib.util.find_spec("flashinfer") is None:
            raise ImportError("SUBBLOCK_SAGE_ATTN requires FlashInfer with the SM120 CAKE Sage kernels")
        if torch.cuda.is_available() and not _is_sm120():
            raise RuntimeError(
                "SUBBLOCK_SAGE_ATTN supports SM120 (RTX PRO 6000 Blackwell) only; this GPU is "
                f"{current_omni_platform.get_device_capability()}. Remove it from --diffusion-attention-config."
            )
        from vllm_omni.diffusion.attention.ops.flashinfer_block_sparse import require_flashinfer_sparse

        require_flashinfer_sparse("sage")

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_name() -> str:
        return "SUBBLOCK_SAGE_ATTN"

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        return None

    @staticmethod
    def get_impl_cls() -> type[SubBlockSageAttentionImpl]:
        return SubBlockSageAttentionImpl


class SubBlockSageAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        cfg = (backend_kwargs or {}).get("subblock") or {}
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.qkv_layout = qkv_layout
        self.sparsity = float(cfg.get("sparsity", 0.75))
        self.start_step = int(cfg.get("start_step", 10))
        self.end_step = int(cfg.get("end_step", 0))
        self.min_seq_len = int(cfg.get("min_seq_len", 4096))
        self.skip_layers = frozenset(cfg.get("skip_layers") or ())
        self.layer_idx = _try_extract_layer_index(prefix)
        from vllm_omni.diffusion.attention.ops.subblock_router import SubBlockRouter

        self.router = SubBlockRouter(n_k=int(cfg.get("n_k", 4)), n_q=int(cfg.get("n_q", 4)))
        self.dense = _make_dense_impl(num_heads, head_size, softmax_scale, causal, num_kv_heads, qkv_layout)

    # ------------------------------------------------------------------ admission
    def _step_enabled(self) -> bool:
        if not is_forward_context_available():
            return self.start_step == 0 and self.end_step == 0
        ctx = get_forward_context()
        step, total = ctx.denoise_step_idx, ctx.total_denoise_steps
        if step is None:
            return self.start_step == 0 and self.end_step == 0
        if step < self.start_step:
            return False
        return not (self.end_step > 0 and total is not None and step >= total - self.end_step)

    def _decline_reason(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_metadata: AttentionMetadata | None
    ) -> str | None:
        if self.layer_idx is not None and self.layer_idx in self.skip_layers:
            return "layer is in skip_layers"
        if not self._step_enabled():
            return "dense denoise step"
        if self.causal:
            return "causal attention"
        if query.dtype != torch.bfloat16 or self.head_size != 128 or query.dim() != 4:
            return "needs BF16 BSHD tensors with head size 128"
        if key.shape[1] != query.shape[1] or key.shape[2] != query.shape[2]:
            return "cross or grouped-query attention"
        if attn_metadata is None or attn_metadata.video_layout is None:
            return "no packed video layout (the dense prefix rows are unknown)"
        if attn_metadata.attn_mask is not None and attn_metadata.packed_padding is None:
            return "an arbitrary attention mask"
        if not _is_sm120():
            return "not an SM120 GPU"
        return None

    @staticmethod
    def _dense_prefix_rows(attn_metadata: AttentionMetadata) -> int | None:
        """Rows before the first target video row: text, visual conditions and audio, all kept dense."""
        layout = attn_metadata.video_layout
        target = next((s for s in layout.video_spans if s.role == "target"), None)
        if target is not None:
            return int(target.start)
        return None if layout.prefix_len is None else int(layout.prefix_len)

    @staticmethod
    def _live_rows(attn_metadata: AttentionMetadata, rows: int) -> int:
        """Used rows of the packing ([prefix | video | padding]): the packed-padding length when published."""
        packed = attn_metadata.packed_padding
        if packed is not None:
            return min(int(packed.q_length), rows)
        layout = attn_metadata.video_layout
        if layout.used_len is not None:
            return min(int(layout.used_len), rows)
        if layout.prefix_len is not None and layout.latent_grid is not None:
            grid = layout.latent_grid
            return min(int(layout.prefix_len) + grid[0] * grid[1] * grid[2], rows)
        return rows

    # ------------------------------------------------------------------ execution
    def _fallback(self, query, key, value, attn_metadata, reason: str) -> torch.Tensor:
        logger.debug_once("SUBBLOCK_SAGE_ATTN running dense BF16 attention: %s", reason)
        packed = attn_metadata.packed_padding if attn_metadata is not None else None
        if attn_metadata is None or packed is None or attn_metadata.attn_mask is not None:
            return self.dense.forward(query, key, value, attn_metadata)
        q_len, kv_len = min(int(packed.q_length), query.shape[1]), min(int(packed.kv_length), key.shape[1])
        out = self.dense.forward(query[:, :q_len], key[:, :kv_len], value[:, :kv_len], None)
        if q_len == query.shape[1]:
            return out
        restored = torch.zeros((out.shape[0], query.shape[1], *out.shape[2:]), device=out.device, dtype=out.dtype)
        restored[:, :q_len] = out
        return restored

    # Opaque to torch.compile: the routing reads per-step Python state (denoise step, layout ints) and launches
    # Triton kernels.
    @torch.compiler.disable
    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        reason = self._decline_reason(query, key, value, attn_metadata)
        if reason is not None:
            return self._fallback(query, key, value, attn_metadata, reason)
        assert attn_metadata is not None
        live = self._live_rows(attn_metadata, query.shape[1])
        prefix = self._dense_prefix_rows(attn_metadata)
        if prefix is None or live < max(self.min_seq_len, BLOCK) or prefix >= live:
            return self._fallback(query, key, value, attn_metadata, "sequence too short or no video rows")
        # The Triton router and the quantizer index BSHD tensors as contiguous; the model hands over strided views.
        out = self._sparse(
            query[:, :live].contiguous(), key[:, :live].contiguous(), value[:, :live].contiguous(), prefix
        )
        if live == query.shape[1]:
            return out
        restored = torch.zeros_like(query)
        restored[:, :live] = out
        return restored

    def _sparse(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, prefix_rows: int) -> torch.Tensor:
        from vllm_omni.diffusion.attention.ops.sage_block_sparse_attention import sage_block_sparse_attention
        from vllm_omni.diffusion.attention.ops.subblock_router import SubBlockRouter

        _, live, heads, _ = query.shape
        blocks = -(-live // BLOCK)
        heads_per_call = max(1, MAX_ROUTED_BLOCK_PAIRS // (blocks * blocks))
        mask = sparse_query_block_mask(live, prefix_rows, query.device)
        block_sizes = SubBlockRouter.block_sizes(live, query.device)
        outputs = []
        for start in range(0, heads, heads_per_call):
            hs = slice(start, start + heads_per_call)
            q, k, v = query[:, :, hs], key[:, :, hs], value[:, :, hs]
            plan = self.router.route(q, k, sparsity=self.sparsity, softmax_scale=self.softmax_scale)
            indices, counts = build_mixed_plan(plan.index, plan.topk, plan.num_blocks, mask)
            outputs.append(sage_block_sparse_attention(q, k, v, indices, counts, block_sizes, self.softmax_scale))
        logger.info_once(
            "SUBBLOCK_SAGE_ATTN active: S=%d dense prefix rows=%d, sparse query rows keep %d/%d key blocks",
            live,
            prefix_rows,
            plan.topk,
            plan.num_blocks,
        )
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)


def _make_dense_impl(num_heads, head_size, softmax_scale, causal, num_kv_heads, qkv_layout) -> AttentionImpl:
    """The exact BF16 attention for the steps and calls that stay dense: cuDNN where it imports, else SDPA."""
    try:
        from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl

        return CuDNNAttentionImpl(num_heads=num_heads, head_size=head_size, softmax_scale=softmax_scale, causal=causal)
    except Exception:
        return SDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            qkv_layout=qkv_layout,
        )


def _try_extract_layer_index(prefix: str) -> int | None:
    try:
        from vllm.model_executor.models.utils import extract_layer_index

        return extract_layer_index(prefix)
    except Exception:
        return None
