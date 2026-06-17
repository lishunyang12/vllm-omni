# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The single seam between the duplex control plane and a model/pipeline.

A ``DuplexAdapter`` says what the pipeline can do, accepts input, decides when to
respond, and produces output chunks. The runtime owns the session lifecycle and
barge-in; the adapter owns only model policy. Two very different pipelines plug
in the same way: a fused audio model (MiniCPM-o: listen/speak + native TTS) and
JoyVL (proactive video + external speech)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from vllm_omni.fullduplex.session import DuplexSession


@dataclass
class DuplexCapability:
    input_modalities: frozenset[str]
    output_modalities: frozenset[str]
    #: Whether the adapter can start a response without an explicit commit.
    proactive: bool = False


@dataclass
class OutputChunk:
    """One piece of assistant output (a text delta, an audio delta, …)."""

    modality: str  # "text" | "audio" | ...
    data: Any
    final: bool = False


@runtime_checkable
class DuplexAdapter(Protocol):
    def capabilities(self) -> DuplexCapability: ...

    async def on_input(self, session: DuplexSession, modality: str, data: Any) -> None:
        """Buffer/ingest one input chunk (frame, audio, or text)."""
        ...

    def should_respond(self, session: DuplexSession) -> bool:
        """Decide whether to start a response now (after input / per tick)."""
        ...

    async def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]:
        """Produce output chunks for one response. The runtime stops consuming
        if the session epoch changes (barge-in), so long generations stay
        interruptible."""
        ...

    async def on_barge_in(self, session: DuplexSession) -> None:
        """React to an interruption (drop in-flight work, etc.)."""
        ...

    async def on_playback_ack(self, session: DuplexSession, cursor: int) -> None:
        """The client acked playback up to ``cursor``; commit played output."""
        ...
