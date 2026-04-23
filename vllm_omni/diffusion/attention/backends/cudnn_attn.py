# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.sdpa import _maybe_reshape_attn_mask

logger = init_logger(__name__)


class CuDNNAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # cuDNN 9.5+ FMHA on Blackwell supports any head_dim divisible by 8
        # up to 256 for BF16/FP16. Empty list = "accept any"; the sdpa_kernel
        # fallback chain handles shapes cuDNN can't take.
        return []

    @staticmethod
    def get_name() -> str:
        return "CUDNN_ATTN"

    @staticmethod
    def get_impl_cls() -> type["CuDNNAttentionImpl"]:
        return CuDNNAttentionImpl


class CuDNNAttentionImpl(AttentionImpl):
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

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        attention_mask = None
        if attn_metadata:
            attention_mask = _maybe_reshape_attn_mask(query, key, attn_metadata.attn_mask, mask_mode="broadcast_k")

        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
            output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=self.causal,
                scale=self.softmax_scale,
            )
        return output.permute(0, 2, 1, 3)
