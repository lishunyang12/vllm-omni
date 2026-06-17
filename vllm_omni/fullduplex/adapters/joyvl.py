# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JoyVL full-duplex adapter — a thin transport over :class:`JoyVLPolicy`.

Video frames stream in, the model decides speak/silence/delegate, and text comes
out (speech stays external — ASR/TTS bridges). All JoyVL logic lives in the
policy; this adapter only buffers frames and runs inference via an injected async
``generate(messages) -> str``."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from vllm_omni.fullduplex.adapter import DuplexAdapter, DuplexCapability, OutputChunk
from vllm_omni.fullduplex.session import DuplexSession
from vllm_omni.interaction.delegation import DelegationBridge, StubDelegationBridge
from vllm_omni.interaction.output_parser import Action
from vllm_omni.interaction.policy import JoyVLPolicy, sample_frames

GenerateFn = Callable[[list[dict[str, Any]]], Awaitable[str]]


class JoyVLDuplexAdapter(DuplexAdapter):
    def __init__(
        self,
        generate: GenerateFn,
        *,
        persona: str = "default",
        num_frames: int = 4,
        chunk_frames: int = 200,
        frame_seconds: float = 1.0,
        delegation: DelegationBridge | None = None,
    ) -> None:
        self._generate = generate
        self._policy = JoyVLPolicy(
            persona=persona,
            num_frames=num_frames,
            chunk_frames=chunk_frames,
            frame_seconds=frame_seconds,
            delegation=delegation or StubDelegationBridge(),
        )
        self._frames: list[str] = []
        self._pending_query: str | None = None

    def capabilities(self) -> DuplexCapability:
        return DuplexCapability(frozenset({"video", "text"}), frozenset({"text"}), proactive=True)

    async def on_input(self, session: DuplexSession, modality: str, data: Any) -> None:
        if modality == "video":
            self._policy.tick()
            self._frames.append(data)
        elif modality == "text":
            self._pending_query = data

    def should_respond(self, session: DuplexSession) -> bool:
        return bool(self._frames)

    async def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]:
        policy = self._policy
        await policy.fold_delegations()
        policy.set_query(self._pending_query)
        self._pending_query = None

        parts = [{"type": "image_url", "image_url": {"url": f}} for f in sample_frames(self._frames, policy.num_frames)]
        messages, _ = policy.build_messages(parts)
        action = policy.commit(await self._generate(messages))

        frames = [(str(i), f) for i, f in enumerate(self._frames)]
        await policy.submit_if_delegate(action, frames)
        if action.action is not Action.SILENCE and action.text:
            yield OutputChunk("text", action.text)
        if policy.needs_flush():
            await policy.flush(frames)
            self._frames.clear()

    async def on_barge_in(self, session: DuplexSession) -> None:
        self._pending_query = None
