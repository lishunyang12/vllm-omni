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


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 192, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl


class FlashAttentionImpl(AttentionImpl):
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
        self.num_heads = num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """CUDA/ROCm flash attention implementation."""
        # Dispatch to FP8 path if K/V are quantized
        if key.dtype == torch.float8_e4m3fn:
            return self._forward_fp8(query, key, value, attn_metadata)
        # Import flash attention functions with fallback chain from utils/fa.py
        # FA3 (fa3_fwd_interface) -> FA3 (flash_attn_interface) -> FA2 (flash_attn)
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            HAS_FLASH_ATTN,
            _pad_input,
            _unpad_input,
            _upad_input,
            flash_attn_func,
            flash_attn_varlen_func,
        )

        if not HAS_FLASH_ATTN:
            raise ImportError(
                "FlashAttentionBackend requires Flash Attention. "
                "Please install one of: fa3-fwd, flash-attention, or flash-attn. "
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        query_length = query.size(1)
        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None
        #  Contains at least one padding token in the sequence
        if attention_mask is not None and torch.any(~attention_mask):
            assert attention_mask.ndim == 2, "attention_mask must be 2D, (batch_size, seq_len)"
            q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
                query, key, value, attention_mask, query_length, _unpad_input
            )

            out_unpad = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seq_lens_q,
                cu_seqlens_k=cu_seq_lens_k,
                max_seqlen_q=max_length_q,
                max_seqlen_k=max_length_k,
                **{
                    "causal": self.causal,
                    "softmax_scale": self.softmax_scale,
                },
            )
            if isinstance(out_unpad, tuple):
                out_unpad = out_unpad[0]

            out = _pad_input(out_unpad, indices_q, query.size(0), query_length)

        else:
            out = flash_attn_func(
                query,
                key,
                value,
                causal=self.causal,
                softmax_scale=self.softmax_scale,
            )
            # FA3 may return (out, lse) tuple, FA2 returns just out
            if isinstance(out, tuple):
                out = out[0]
        return out

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """NPU attention implementation using mindiesd."""
        try:
            from mindiesd import attention_forward
        except ImportError:
            raise ImportError(
                "FlashAttentionBackend NPU implementation requires MindIE-SD. "
                "Please install MindIE-SD to enable NPU attention support. "
                "For installation details, see https://gitcode.com/Ascend/MindIE-SD"
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        attention_mask = attn_metadata.attn_mask if attn_metadata else None
        output = attention_forward(
            query,
            key,
            value,
            attn_mask=attention_mask,
            opt_mode="manual",
            op_type="fused_attn_score",
            layout="BNSD",
        )
        return output

    def _forward_fp8(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """FP8 KV attention path: native FA3 or dequant fallback."""
        k_scale = attn_metadata.k_scale
        v_scale = attn_metadata.v_scale

        # Try FA3 native FP8 (Hopper / Ada / Ampere via fa3-fwd)
        from vllm_omni.diffusion.attention.backends.ring.ring_globals import (
            HAS_FA3,
            fa3_attn_func,
        )

        if HAS_FA3 and fa3_attn_func is not None:
            out = fa3_attn_func(
                query,
                key,
                value,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                descale_k=k_scale,
                descale_v=v_scale,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out

        # Fallback: dequantize to compute dtype and use standard path
        logger.warning_once(
            "FP8 KV quantization without FA3 provides no performance benefit. "
            "Install FA3 for optimal FP8 support on Hopper GPUs."
        )
        from vllm_omni.diffusion.quantization.kv_quant import dequantize_fp8

        key_bf16 = dequantize_fp8(key, k_scale, query.dtype)
        value_bf16 = dequantize_fp8(value, v_scale, query.dtype)
        # Clear scales to avoid re-detection on recursive call
        attn_metadata.k_scale = None
        attn_metadata.v_scale = None
        return self.forward_cuda(query, key_bf16, value_bf16, attn_metadata)
