# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JoyVL full-duplex adapter.

Plugs JoyVL into the duplex runtime: video frames stream in, the model decides
speak/silence/delegate, and text comes out (speech stays external — ASR/TTS
bridges). It reuses :class:`InteractionBrain` for memory/Q&A and an injected
async ``generate(messages) -> str`` for inference, so it carries no transport or
engine specifics. Output is text deltas; barge-in is handled by the runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from vllm_omni.fullduplex.adapter import DuplexCapability, OutputChunk
from vllm_omni.fullduplex.session import DuplexSession
from vllm_omni.interaction.output_parser import Action, parse_action
from vllm_omni.interaction.prompts import SYSTEM_PROMPTS, USER_QUERY_HEADER
from vllm_omni.interaction.state import InteractionBrain

GenerateFn = Callable[[list[dict[str, Any]]], Awaitable[str]]


def _sample(frames: list[str], num: int) -> list[str]:
    n = len(frames)
    if n <= num:
        return list(frames)
    stride = max(1, n // num)
    idx = [i * stride for i in range(num - 1)] + [n - 1]
    return [frames[i] for i in idx]


class JoyVLDuplexAdapter:
    def __init__(
        self,
        generate: GenerateFn,
        *,
        persona: str = "default",
        num_frames: int = 4,
        chunk_frames: int = 200,
        frame_seconds: float = 1.0,
    ) -> None:
        self._generate = generate
        self._system_prompt = SYSTEM_PROMPTS.get(persona, SYSTEM_PROMPTS["default"])
        self._num_frames = num_frames
        self._brain = InteractionBrain(chunk_frames=chunk_frames, frame_seconds=frame_seconds)
        self._frames: list[str] = []
        self._pending_query: str | None = None

    def capabilities(self) -> DuplexCapability:
        return DuplexCapability(
            input_modalities=frozenset({"video", "text"}),
            output_modalities=frozenset({"text"}),
            proactive=True,
        )

    async def on_input(self, session: DuplexSession, modality: str, data: Any) -> None:
        if modality == "video":
            self._brain.tick()
            self._frames.append(data)
        elif modality == "text":
            self._pending_query = data

    def should_respond(self, session: DuplexSession) -> bool:
        return bool(self._frames)

    async def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]:
        brain = self._brain
        brain.update_query(self._pending_query)
        self._pending_query = None

        content: list[dict[str, Any]] = []
        prefix = brain.build_prefix()
        if prefix:
            content.append({"type": "text", "text": prefix})
        for frame in _sample(self._frames, self._num_frames):
            content.append({"type": "image_url", "image_url": {"url": frame}})
        if brain.current_query and brain.query_in_current_chunk:
            content.append({"type": "text", "text": f"{USER_QUERY_HEADER}\n{brain.current_query}"})
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": content},
        ]

        action = parse_action(await self._generate(messages))
        if action.action is not Action.SILENCE and brain.current_query:
            brain.record_response(action.text)
        if action.action is not Action.SILENCE and action.text:
            yield OutputChunk("text", action.text, final=True)

        if brain.should_flush():
            await brain.flush([(str(i), f) for i, f in enumerate(self._frames)])
            self._frames.clear()

    async def on_barge_in(self, session: DuplexSession) -> None:
        self._pending_query = None

    async def on_playback_ack(self, session: DuplexSession, cursor: int) -> None:
        return
