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
        # FlashInfer single_prefill_with_kv_cache accepts a ``custom_mask`` kwarg
        # (2D boolean, ``True`` = keep) for non-causal attention. See reviewer
        # comment on #3079 and the flashinfer docs:
        #   https://docs.flashinfer.ai/generated/flashinfer.prefill.single_prefill_with_kv_cache.html
        return True

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

    @staticmethod
    def _pack_mask_for_flashinfer(attn_mask: torch.Tensor) -> torch.Tensor | None:
        """Convert a diffusion-style attn_mask into the 2D boolean form
        FlashInfer's ``custom_mask`` expects.

        Accepts:
          (S, S) — already 2D; just binarize
          (B, 1, S, S) / (1, 1, S, S) — take the first slice
          (B, H, S, S) — take the first slice (heads share the mask in the
                         diffusion paths we see; if that ever changes we'll
                         need a per-head pack)

        Returns ``None`` when the mask is all-ones and can be elided.
        """
        mask = attn_mask
        while mask.dim() > 2:
            mask = mask[0]
        # FlashInfer expects True = keep. Our masks come in as additive
        # float (0 = keep, -inf = drop). Anything non-zero and finite-negative
        # means "drop"; treat strictly negative as mask-out.
        if mask.dtype == torch.bool:
            bool_mask = mask
        else:
            bool_mask = mask > float("-inf") / 2
        if bool_mask.all():
            return None
        return bool_mask

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

        custom_mask = None
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            custom_mask = self._pack_mask_for_flashinfer(attn_metadata.attn_mask)

        # Input layout is (B, S, H, D); FlashInfer dense prefill takes (S, H, D).
        batch_size = query.shape[0]
        outputs = []
        for b in range(batch_size):
            kwargs: dict = {
                "sm_scale": self.softmax_scale,
                "causal": self.causal,
                "return_lse": False,
            }
            # ``custom_mask`` is only honored when causal=False per FlashInfer docs.
            if custom_mask is not None and not self.causal:
                kwargs["custom_mask"] = custom_mask
            out = single_prefill_with_kv_cache(query[b], key[b], value[b], **kwargs)
            outputs.append(out)

        if batch_size == 1:
            return outputs[0].unsqueeze(0)
        return torch.stack(outputs, dim=0)
