# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The single seam between the duplex control plane and a model/pipeline.

A ``DuplexAdapter`` says what the pipeline can do, accepts input, and produces
output; the runtime owns the session lifecycle and barge-in. A minimal adapter
implements only ``capabilities`` / ``on_input`` / ``respond`` — ``should_respond``
and the barge-in / playback hooks have sensible defaults. Two very different
pipelines plug in the same way: a fused audio model (MiniCPM-o: listen/speak +
native TTS) and JoyVL (proactive video + external speech)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

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


class DuplexAdapter(ABC):
    # ----- required ------------------------------------------------------- #

    @abstractmethod
    def capabilities(self) -> DuplexCapability: ...

    @abstractmethod
    async def on_input(self, session: DuplexSession, modality: str, data: Any) -> None:
        """Buffer/ingest one input chunk (frame, audio, or text)."""

    @abstractmethod
    def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]:
        """Produce output chunks for one response (an async generator). The
        runtime stops consuming if the session epoch changes (barge-in), so long
        generations stay interruptible."""

    # ----- optional (defaults) -------------------------------------------- #

    def should_respond(self, session: DuplexSession) -> bool:
        """Whether to start a response now (after input / per tick)."""
        return True

    async def on_barge_in(self, session: DuplexSession) -> None:
        """React to an interruption (drop in-flight work, etc.)."""

    async def on_playback_ack(self, session: DuplexSession, cursor: int) -> None:
        """The client acked playback up to ``cursor``; commit played output."""
