# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

try:
    from flashinfer.prefill import single_prefill_with_kv_cache

    HAS_FLASHINFER = True
except Exception as e:
    HAS_FLASHINFER = False
    logger.warning(
        "FlashInfer is unavailable; FLASHINFER_ATTN backend will not work. Reason: %s",
        e,
    )


class FlashInferAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        # Dense prefill path does not consume arbitrary 2D padding masks;
        # the Impl falls back to SDPA when a mask is present.
        return False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # FlashInfer dense prefill is well-tested for these head_dims on
        # Ampere/Hopper/Blackwell. Covers the dominant diffusion DiT shapes
        # (SD3 = 64, Flux/HV/Wan = 128, joint-attn = 256).
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashInferAttentionImpl"]:
        return FlashInferAttentionImpl


class FlashInferAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale

    def _sdpa_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

        impl = SDPAImpl(
            num_heads=query.shape[2],
            head_size=query.shape[3],
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return impl.forward_cuda(query, key, value, attn_metadata)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if not HAS_FLASHINFER:
            raise ImportError(
                "FLASHINFER_ATTN backend requires flashinfer. "
                "Install it or set DIFFUSION_ATTENTION_BACKEND to another backend."
            )

        # Custom / padding masks: dense prefill has no generic mask path, so
        # fall back to SDPA which handles them correctly.
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            if not torch.all(attn_metadata.attn_mask != 0):
                return self._sdpa_fallback(query, key, value, attn_metadata)

        # Input layout is (B, S, H, D); FlashInfer dense prefill takes (S, H, D).
        batch_size = query.shape[0]
        outputs = []
        for b in range(batch_size):
            out = single_prefill_with_kv_cache(
                query[b],
                key[b],
                value[b],
                sm_scale=self.softmax_scale,
                causal=self.causal,
                return_lse=False,
            )
            outputs.append(out)

        if batch_size == 1:
            return outputs[0].unsqueeze(0)
        return torch.stack(outputs, dim=0)
