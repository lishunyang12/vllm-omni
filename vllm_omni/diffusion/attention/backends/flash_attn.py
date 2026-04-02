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

    @staticmethod
    def _unwrap_flash_output(out: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        # FA3 may return (out, lse), FA2 returns out
        return out[0] if isinstance(out, tuple) else out

    def _forward_varlen_masked(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            _pad_input,
            _unpad_input,
            _upad_input,
            flash_attn_varlen_func,
        )

        assert attention_mask.ndim == 2, "attention_mask must be 2D, (batch_size, seq_len)"
        query_length = query.size(1)
        q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
            query, key, value, attention_mask, query_length, _unpad_input
        )

        varlen_kwargs: dict = {
            "causal": self.causal,
            "softmax_scale": self.softmax_scale,
        }
        if q_descale is not None:
            varlen_kwargs["q_descale"] = q_descale
            varlen_kwargs["k_descale"] = k_descale
            varlen_kwargs["v_descale"] = v_descale

        out_unpad = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            **varlen_kwargs,
        )
        out_unpad = self._unwrap_flash_output(out_unpad)
        return _pad_input(out_unpad, indices_q, query.size(0), query_length)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """CUDA/ROCm flash attention implementation."""
        # Dispatch to FP8 path if Q/K/V are quantized
        if key.dtype == torch.float8_e4m3fn:
            return self._forward_fp8(query, key, value, attn_metadata)
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            HAS_FLASH_ATTN,
            flash_attn_func,
        )

        if not HAS_FLASH_ATTN:
            raise ImportError(
                "FlashAttentionBackend requires Flash Attention. "
                "Please install one of: fa3-fwd, flash-attention, or flash-attn. "
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None

        if attention_mask is not None and torch.any(~attention_mask):
            return self._forward_varlen_masked(
                query,
                key,
                value,
                attention_mask,
            )

        out = flash_attn_func(
            query,
            key,
            value,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        return self._unwrap_flash_output(out)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """XPU flash attention implementation."""
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            HAS_FLASH_ATTN,
            flash_attn_varlen_func,
        )

        if not HAS_FLASH_ATTN:
            raise ImportError(
                "FlashAttentionBackend requires Flash Attention. "
                "Please assure vllm-xpu-kernels properly installed. "
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None

        if attention_mask is not None and torch.any(~attention_mask):
            return self._forward_varlen_masked(
                query,
                key,
                value,
                attention_mask,
            )

        batch_size, q_len = query.size()[:2]
        cu_seqlens = torch.arange(0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device)
        # b s ... -> (b s) ...
        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        out = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=q_len,
            max_seqlen_k=q_len,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        out = self._unwrap_flash_output(out)
        # (b s) h d -> b s h d
        return out.reshape(batch_size, q_len, *out.shape[1:])

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

    @staticmethod
    def _reshape_descale(scale: torch.Tensor, batch: int, num_heads_k: int) -> torch.Tensor:
        """Reshape per-tensor scale to FA3's expected (batch, num_heads_k) shape."""
        return scale.view(1, 1).expand(batch, num_heads_k).contiguous()

    def _forward_fp8(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """FP8 Q/K/V attention path.

        Uses vLLM's bundled FA3 backend (vllm_flash_attn) which has the
        two-level accumulation fix for FP8 on Hopper. Falls back to
        fa3-fwd or dequant if vLLM's FA3 is unavailable.
        """
        q_scale = attn_metadata.q_scale
        k_scale = attn_metadata.k_scale
        v_scale = attn_metadata.v_scale

        B, S, H, D = key.shape
        q_descale = self._reshape_descale(q_scale, B, H)
        k_descale = self._reshape_descale(k_scale, B, H)
        v_descale = self._reshape_descale(v_scale, B, H)

        # Try vLLM's bundled FA3 (has two-level accumulation fix for FP8)
        try:
            from vllm.vllm_flash_attn import flash_attn_varlen_func as vllm_varlen

            # varlen API needs (total_tokens, H, D) and cu_seqlens
            q_flat = query.reshape(B * S, H, D)
            k_flat = key.reshape(B * S, H, D)
            v_flat = value.reshape(B * S, H, D)
            cu_seqlens = torch.arange(
                0, (B + 1) * S, step=S, dtype=torch.int32, device=query.device
            )

            out = vllm_varlen(
                q_flat, k_flat, v_flat,
                max_seqlen_q=S,
                cu_seqlens_q=cu_seqlens,
                max_seqlen_k=S,
                cu_seqlens_k=cu_seqlens,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                fa_version=3,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out.reshape(B, S, H, D)
        except Exception as e:
            logger.warning_once(
                "vLLM FA3 FP8 failed (%s), trying fa3-fwd fallback.", e
            )

        # Fallback: fa3-fwd (may lack two-level accumulation fix)
        from vllm_omni.diffusion.attention.backends.ring.ring_globals import (
            HAS_FA3,
            fa3_attn_func,
        )

        if HAS_FA3 and fa3_attn_func is not None:
            out = fa3_attn_func(
                query, key, value,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out

        # Last resort: dequant to BF16
        from vllm_omni.quantization.kv_quant import dequantize_fp8

        logger.warning_once(
            "No FA3 available for FP8 attention. Dequantizing to BF16."
        )
        output_dtype = torch.bfloat16
        query = dequantize_fp8(query, q_scale, output_dtype)
        key = dequantize_fp8(key, k_scale, output_dtype)
        value = dequantize_fp8(value, v_scale, output_dtype)
        attn_metadata.q_scale = None
        attn_metadata.k_scale = None
        attn_metadata.v_scale = None
        return self.forward_cuda(query, key, value, attn_metadata)
