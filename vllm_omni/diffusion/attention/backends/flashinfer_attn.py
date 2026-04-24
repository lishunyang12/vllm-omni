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
    def _pack_mask_for_flashinfer(
        attn_mask: torch.Tensor, qo_len: int, kv_len: int
    ) -> torch.Tensor | None:
        """Convert a diffusion-style attn_mask into the 2D boolean form
        FlashInfer's ``custom_mask`` expects: ``(qo_len, kv_len)``, ``True``
        = keep.

        Returns ``None`` when the mask is all-ones (elide), or when the shape
        cannot be broadcast to ``(qo_len, kv_len)``. Caller falls back to
        SDPA in the latter case — FlashInfer crashes (illegal memory access)
        rather than raising if handed a wrong-shape mask, so we pre-check.
        """
        mask = attn_mask
        # Strip leading singleton dims: (B, H, S_q, S_k) -> (S_q, S_k) when
        # B == H == 1 or all heads share the mask. We only keep the final
        # 2D slice; if it's a broadcast-ready (1, S_k) or (S_k,) mask, fall
        # back to SDPA rather than guess-broadcast it.
        while mask.dim() > 2:
            mask = mask[0]
        if mask.dim() != 2 or mask.shape != (qo_len, kv_len):
            # Shape doesn't match what FlashInfer expects. Signal the caller
            # (via ValueError) to fall back to SDPA — safer than passing a
            # mismatched buffer to the kernel.
            raise ValueError(
                f"attn_mask shape {tuple(mask.shape)} is not (qo_len={qo_len}, "
                f"kv_len={kv_len}); falling back to SDPA"
            )
        if mask.dtype == torch.bool:
            bool_mask = mask
        else:
            bool_mask = mask != float("-inf")
        if bool_mask.all():
            return None
        return bool_mask.contiguous()

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

        # Try the custom_mask path; if the mask can't be packed into the
        # (qo_len, kv_len) layout FlashInfer expects, fall back to SDPA
        # instead of risking an illegal-memory-access crash in the kernel.
        custom_mask = None
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            try:
                custom_mask = self._pack_mask_for_flashinfer(
                    attn_metadata.attn_mask, qo_len=query.shape[1], kv_len=key.shape[1]
                )
            except ValueError as e:
                logger.debug("Falling back to SDPA for mask path: %s", e)
                return self._sdpa_fallback(query, key, value, attn_metadata)

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
