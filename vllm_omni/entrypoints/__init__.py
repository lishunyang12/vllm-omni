# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
vLLM-Omni entrypoints module.

Provides high-level interfaces for running omni models including:
- AsyncOmni: Async orchestrator for multi-stage LLM pipelines
- AsyncOmniDiffusion: Async interface for diffusion model inference
- Omni: Unified entrypoint that auto-selects between LLM and Diffusion
"""

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.entrypoints.async_omni_v1 import AsyncOmniV1
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.omni_v1 import OmniV1

__all__ = [
    "AsyncOmni",
    "AsyncOmniDiffusion",
    "Omni",
    "AsyncOmniV1",
    "OmniV1",
]
