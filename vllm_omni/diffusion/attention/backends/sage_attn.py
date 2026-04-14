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
    from sageattention import sageattn_qk_int8_pv_fp16_cuda
except ImportError:
    logger.warning(
        "SageAttentionBackend is not available. You may install sage-attention"
        " by pip install git+https://github.com/thu-ml/SageAttention.git"
    )
    raise ImportError

# TODO add sage3 attention backend


@torch.compiler.disable
def _run_sageattn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    # SageAttention relies on custom pybind/CUDA entrypoints that torch.compile
    # does not trace correctly in Wan2.2 TP serving. Keeping the kernel eager
    # avoids the second-request corruption that can collapse outputs to black.
    # The upstream Wan integration uses HND layout. Match that path here
    # instead of SageAttention's NHD branch to minimize backend drift.
    output = sageattn_qk_int8_pv_fp16_cuda(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        tensor_layout="HND",
        is_causal=causal,
        sm_scale=softmax_scale,
        pv_accum_dtype="fp32",
    )
    return output.transpose(1, 2)


class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):
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
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        output = _run_sageattn(
            query=query,
            key=key,
            value=value,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        return output
