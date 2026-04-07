# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


class AttentionBackend(ABC):
    """Abstract class for diffusion attention backends."""

    accept_output_buffer: bool = False

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None) -> bool:
        """Whether this backend supports the given KV cache quantization dtype.

        Override in subclasses that support quantized KV (e.g. FP8).
        Default: only None (no quantization) is supported.
        """
        return kv_cache_dtype is None

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls():  # -> Type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_supported_head_sizes() -> list[int]:
        """Get the list of supported head sizes for this backend."""
        raise NotImplementedError

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        supported_head_sizes = cls.get_supported_head_sizes()
        return (not supported_head_sizes) or head_size in supported_head_sizes


@dataclass
class AttentionMetadata:
    attn_mask: torch.Tensor | None = None
    joint_attn_mask: torch.Tensor | None = None
    # a joint mask for the joint query, key, and value, depends the joint_strategy
    joint_query: torch.Tensor | None = None
    # a replicated tensor among processes appended to the front or rear of query, depends the joint_strategy
    joint_key: torch.Tensor | None = None
    # a replicated tensor among processes appended to the front or rear of key, depends the joint_strategy
    joint_value: torch.Tensor | None = None
    # a replicated tensor among processes appended to the front or rear of value, depends the joint_strategy
    joint_strategy: str = "front"
    # the strategy to joint the query, key, and value, can be "front" or "rear"

    # KV cache dtype for quantization (e.g. "fp8"). Each backend decides
    # whether and how to quantize Q/K/V based on this field.
    kv_cache_dtype: str | None = None


T = TypeVar("T", bound=AttentionMetadata)


class AttentionImpl(ABC, Generic[T]):

    # Per-platform kv_cache_dtype support. Maps OmniPlatformEnum value
    # (e.g. "cuda", "npu") to the set of quantized dtypes that platform
    # handles. The base forward() checks this before dispatching and
    # clears unsupported dtypes with a warning.
    #
    # To add FP8 support for a new platform in a subclass:
    #   _supported_kv_cache_dtypes = {"cuda": {"fp8"}, "npu": {"fp8"}}
    _supported_kv_cache_dtypes: dict[str, set[str]] = {}

    @abstractmethod
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
        raise NotImplementedError

    # Platform enum value → forward method name. New platforms only need
    # to implement forward_{name}() and add an entry here.
    _PLATFORM_DISPATCH: dict[str, str] = {
        "cuda": "forward_cuda",
        "rocm": "forward_hip",
        "npu": "forward_npu",
        "xpu": "forward_xpu",
        "musa": "forward_musa",
    }

    def _handle_kv_cache_dtype(
        self,
        attn_metadata: T | None,
        platform_key: str,
    ) -> None:
        """Check kv_cache_dtype compatibility for this platform.

        If the requested kv_cache_dtype is not in _supported_kv_cache_dtypes
        for the current platform, it is cleared to None with a warning.
        """
        if attn_metadata is None:
            return
        kv_cache_dtype = attn_metadata.kv_cache_dtype
        if kv_cache_dtype is None:
            return
        supported = self._supported_kv_cache_dtypes.get(platform_key, set())
        if kv_cache_dtype not in supported:
            logger.warning_once(
                "kv_cache_dtype='%s' requested but %s on %s does not support "
                "it. Running in native dtype.",
                kv_cache_dtype,
                type(self).__name__,
                platform_key,
            )
            attn_metadata.kv_cache_dtype = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        """Dispatch to platform-specific forward implementation."""
        platform_key = current_omni_platform._omni_enum.value
        method_name = self._PLATFORM_DISPATCH.get(platform_key)
        if method_name is None:
            raise NotImplementedError(
                f"No forward implementation for platform: {platform_key}. "
                f"Register it in AttentionImpl._PLATFORM_DISPATCH."
            )
        self._handle_kv_cache_dtype(attn_metadata, platform_key)
        method = getattr(self, method_name)
        return method(query, key, value, attn_metadata)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        # By default, HIP ops are compatible with CUDA ops.
        return self.forward_cuda(query, key, value, attn_metadata)

    def forward_musa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        # By default, MUSA ops are compatible with CUDA ops.
        return self.forward_cuda(query, key, value, attn_metadata)
