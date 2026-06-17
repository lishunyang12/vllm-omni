# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The JoyVL per-tick policy — one place for the decision logic.

Like a Tianshou ``Policy``, this owns the decision (and the session memory via
``InteractionBrain``); transports are thin "collectors" that feed frames/query,
call :meth:`build_messages` + :meth:`commit`, and run the async lifecycle
(:meth:`fold_delegations` / :meth:`submit_if_delegate` / :meth:`flush`).
Inference itself is the transport's job (an OpenAI backend, the WS engine, …),
so the JoyVL logic lives here once instead of in each transport."""

from __future__ import annotations

from typing import Any

from vllm_omni.interaction.delegation import DelegationBridge
from vllm_omni.interaction.memory import Summarizer
from vllm_omni.interaction.output_parser import Action, ParsedAction, parse_action
from vllm_omni.interaction.prompts import SYSTEM_PROMPTS, USER_QUERY_HEADER
from vllm_omni.interaction.state import InteractionBrain


def sample_frames(frames: list[str], num_frames: int) -> list[str]:
    """Uniformly sample up to ``num_frames`` frames (keeping the most recent)."""
    n = len(frames)
    if n <= num_frames:
        return list(frames)
    stride = max(1, n // num_frames)
    idx = [i * stride for i in range(num_frames - 1)] + [n - 1]
    return [frames[i] for i in idx]


class JoyVLPolicy:
    def __init__(
        self,
        *,
        persona: str = "default",
        system_prompt: str | None = None,
        num_frames: int = 4,
        summarizer: Summarizer | None = None,
        delegation: DelegationBridge | None = None,
        chunk_frames: int = 200,
        long_term_every_n_chunks: int = 5,
        keep_qa_history: bool = True,
        frame_seconds: float = 1.0,
        enable_delegation: bool = True,
    ) -> None:
        self.brain = InteractionBrain(
            summarizer=summarizer,
            delegation=delegation,
            chunk_frames=chunk_frames,
            long_term_every_n_chunks=long_term_every_n_chunks,
            keep_qa_history=keep_qa_history,
            frame_seconds=frame_seconds,
            enable_delegation=enable_delegation,
        )
        self.system_prompt = system_prompt or SYSTEM_PROMPTS.get(persona, SYSTEM_PROMPTS["default"])
        self.num_frames = num_frames

    # ----- ingest --------------------------------------------------------- #

    def tick(self, n: int = 1) -> None:
        self.brain.tick(n)

    def set_query(self, query: str | None) -> bool:
        return self.brain.update_query(query)

    def should_respond(self) -> bool:
        # Per-tick proactivity is the default; a transport may gate further
        # (e.g. only when the engine is free). The model decides silence itself.
        return True

    # ----- prompt --------------------------------------------------------- #

    def user_content(self, frame_parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stable head (memory prefix) + frames + the active query (append-only)."""
        content: list[dict[str, Any]] = []
        prefix = self.brain.build_prefix()
        if prefix:
            content.append({"type": "text", "text": prefix})
        content.extend(frame_parts)
        if self.brain.current_query and self.brain.query_in_current_chunk:
            content.append({"type": "text", "text": f"{USER_QUERY_HEADER}\n{self.brain.current_query}"})
        return content

    def build_messages(self, frame_parts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        user = {"role": "user", "content": self.user_content(frame_parts)}
        return [{"role": "system", "content": self.system_prompt}, user], user

    # ----- commit + async lifecycle --------------------------------------- #

    def commit(self, response_text: str) -> ParsedAction:
        action = parse_action(response_text)
        if action.action is not Action.SILENCE and self.brain.current_query:
            self.brain.record_response(action.text)
        return action

    async def fold_delegations(self) -> dict[str, Any] | None:
        return await self.brain.fold_delegations()

    async def submit_if_delegate(
        self, action: ParsedAction, frames: list[tuple[str, str]] | None = None
    ) -> dict | None:
        return await self.brain.submit_delegation(action, frames or [])

    def needs_flush(self) -> bool:
        return self.brain.should_flush()

    async def flush(self, frames: list[tuple[str, str]] | None = None) -> None:
        await self.brain.flush(frames or [])
