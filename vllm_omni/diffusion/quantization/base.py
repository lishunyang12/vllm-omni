# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for diffusion model quantization configurations."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )


class DiffusionQuantizationConfig(ABC):
    """Base class for diffusion model quantization configurations.

    This provides a thin wrapper over vLLM's quantization configs,
    allowing diffusion-model-specific defaults and future extensibility.

    Subclasses should implement:
        - get_name(): Return the quantization method name
        - get_vllm_quant_config(): Return the underlying vLLM config
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the quantization method name (e.g., 'fp8', 'int8')."""
        raise NotImplementedError

    @abstractmethod
    def get_vllm_quant_config(self) -> "QuantizationConfig | None":
        """Return the underlying vLLM QuantizationConfig for linear layers."""
        raise NotImplementedError

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        """Return supported activation dtypes."""
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU compute capability required.

        Override in subclasses for method-specific requirements.
        """
        return 80  # Ampere default
