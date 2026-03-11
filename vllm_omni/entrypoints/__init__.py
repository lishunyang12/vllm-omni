# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
vLLM-Omni entrypoints module.
"""

from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.entrypoints.async_omni_v1 import AsyncOmniV1
from vllm_omni.entrypoints.omni_v1 import OmniV1

AsyncOmni = AsyncOmniV1
Omni = OmniV1

__all__ = [
    "AsyncOmni",
    "AsyncOmniDiffusion",
    "Omni",
]
