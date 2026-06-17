# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-agnostic full-duplex realtime core.

A full-duplex session runs input and output concurrently: input keeps arriving
while the assistant is producing output, and a barge-in can interrupt in-flight
output. This package owns that *control plane* — session lifecycle, epoch-based
barge-in, and the realtime event loop — behind a single :class:`DuplexAdapter`
seam. Model/pipeline specifics (a fused audio model like MiniCPM-o, or JoyVL's
proactive video loop) live in adapters under ``fullduplex.adapters``; they never
reimplement the lifecycle."""

from vllm_omni.fullduplex.adapter import DuplexAdapter, DuplexCapability, OutputChunk
from vllm_omni.fullduplex.runtime import DuplexRuntime
from vllm_omni.fullduplex.session import DuplexSession, DuplexSessionConfig, DuplexState

__all__ = [
    "DuplexAdapter",
    "DuplexCapability",
    "DuplexRuntime",
    "DuplexSession",
    "DuplexSessionConfig",
    "DuplexState",
    "OutputChunk",
]
