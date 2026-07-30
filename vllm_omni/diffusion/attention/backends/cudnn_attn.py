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

        enable_gqa = query.shape[2] != key.shape[2]
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))

        def _run_sdpa(backends: list[SDPBackend]) -> torch.Tensor:
            with sdpa_kernel(backends):
                return torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=self.causal,
                    scale=self.softmax_scale,
                    enable_gqa=enable_gqa,
                )

        # cuDNN FMHA rejects K/V sequence length 1 (LTX-2 audio cross-attn /
        # dummy warmup). Prefer MATH up-front: under torch.compile fake-tensor
        # tracing, an unpinned SDPA fallback still aborts with
        # "No available kernel" once Flash/Efficient are runtime-disabled.
        if key.shape[2] == 1:
            return _run_sdpa([SDPBackend.MATH]).permute(0, 2, 1, 3)

        # Pin cuDNN exclusively. A priority list like [CUDNN, FLASH, MATH] hits a
        # PyTorch SDPA dispatch quirk: when FLASH rejects a non-None attn_mask,
        # cuDNN gets runtime-disabled in the same call and the dispatcher falls
        # through to MATH even though cuDNN alone handles the mask fine
        # (~11 ms vs ~215 ms for MATH on sm_120 HV-1.5 shapes).
        #
        # Fall back to MATH if cuDNN rejects the shape, e.g. under torch.compile
        # where Dynamo sees a symbolic head_dim and cuDNN's kernel selection
        # fails (observed in LTX-2 audio attention).
        try:
            output = _run_sdpa([SDPBackend.CUDNN_ATTENTION])
        except RuntimeError as e:
            if "No available kernel" not in str(e):
                raise
            logger.warning_once(
                "cuDNN SDPA rejected this shape; falling back to MATH SDPA. "
                "Set DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA for the full dispatcher path on every call."
            )
            output = _run_sdpa([SDPBackend.MATH])
        return output.permute(0, 2, 1, 3)
