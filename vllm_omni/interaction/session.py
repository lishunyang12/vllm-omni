# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP per-session interaction loop.

This is the HTTP transport over the shared :class:`InteractionBrain`: it owns the
multi-turn chunk message buffer and frame list and runs inference via a
``ModelBackend``, but delegates all query / Q&A / memory / chunk-flush /
delegation state to the brain so that logic lives in one place (shared with the
streaming-video handler).

Message layout is stable-head / append-only for prefix-cache reuse: the head
(system prompt + memory prefix) only changes at chunk-flush boundaries; a newly
issued query rides in that tick's appended message, and moves into the head once
its chunk is evicted."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.interaction.backend import ModelBackend
from vllm_omni.interaction.config import InteractionConfig
from vllm_omni.interaction.delegation import DelegationBridge
from vllm_omni.interaction.memory import Summarizer, WorkingChunk
from vllm_omni.interaction.output_parser import Action, ParsedAction, parse_action
from vllm_omni.interaction.prompts import SYSTEM_PROMPTS, USER_QUERY_HEADER
from vllm_omni.interaction.state import InteractionBrain


@dataclass
class StepResult:
    action: ParsedAction
    chunk_index: int
    frame_index: int
    inference_skipped: bool
    latency_ms: float
    long_term_memory: str
    mid_term_summaries: list[dict[str, Any]] = field(default_factory=list)
    delegation: dict[str, Any] | None = None


class InteractionSession:
    def __init__(
        self,
        session_id: str,
        config: InteractionConfig,
        backend: ModelBackend,
        summarizer: Summarizer | None = None,
        delegation: DelegationBridge | None = None,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self._backend = backend
        self._brain = InteractionBrain(
            summarizer=summarizer,
            delegation=delegation,
            chunk_frames=config.chunk_frames,
            long_term_every_n_chunks=config.long_term_every_n_chunks,
            keep_qa_history=config.keep_qa_history,
            frame_seconds=config.frame_seconds,
            enable_delegation=config.enable_delegation,
        )
        self.chunk = WorkingChunk()  # transport-local: multi-turn messages + frames
        self._system_prompt = config.system_prompt
        self.last_access = time.monotonic()

    def set_persona(self, persona: str) -> bool:
        prompt = SYSTEM_PROMPTS.get(persona)
        if prompt is None:
            return False
        self._system_prompt = prompt
        return True

    async def step(self, frames: list[str], query: str | None = None, t: float | None = None) -> StepResult:
        self.last_access = time.monotonic()
        started = time.perf_counter()
        brain = self._brain

        base = t if t is not None else brain.frame_index * self.config.frame_seconds
        time_ranges = [f"{base + i * self.config.frame_seconds:.1f}s" for i in range(len(frames))]

        query_is_fresh = brain.update_query(query)
        delegation_info = await brain.fold_delegations()

        if brain.should_flush():
            await brain.flush(self.chunk.frames)
            self.chunk = WorkingChunk()

        for tr, url in zip(time_ranges, frames):
            brain.tick()
            self.chunk.frames.append((tr, url))
        self.chunk.messages.append(self._frame_message(time_ranges, frames, include_query=query_is_fresh))

        if self.config.force_silence_before_query and not brain.current_query:
            action = ParsedAction(Action.SILENCE, raw="</silence>")
            skipped = True
        else:
            action = await self._infer()
            skipped = False
        self.chunk.messages.append({"role": "assistant", "content": action.raw or "</silence>"})

        if action.spoke:
            brain.record_response(action.text)
        submitted = await brain.submit_delegation(action, list(self.chunk.frames))
        if submitted:
            delegation_info = submitted

        return StepResult(
            action=action,
            chunk_index=brain.chunk_index,
            frame_index=brain.frame_index,
            inference_skipped=skipped,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            long_term_memory=brain.memory.long_term_memory,
            mid_term_summaries=[
                {"chunk_index": m.chunk_index, "frame_range": m.frame_range, "summary_text": m.summary_text}
                for m in brain.memory.mid_term_summaries
            ],
            delegation=delegation_info,
        )

    def reset(self) -> None:
        self._brain.reset()
        self.chunk = WorkingChunk()

    async def _infer(self) -> ParsedAction:
        s = self.config.sampling
        extra_body = {
            "skip_special_tokens": False,
            "top_k": s.top_k,
            "repetition_penalty": s.repetition_penalty,
            "presence_penalty": s.presence_penalty,
        }
        raw, _ = await self._backend.generate(
            self._build_api_messages(),
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
            extra_body=extra_body,
        )
        return parse_action(raw)

    def _build_api_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt}]
        chunk_messages = [dict(m) for m in self.chunk.messages]
        prefix = self._brain.build_prefix()
        if prefix and chunk_messages:
            head = chunk_messages[0]
            head["content"] = [{"type": "text", "text": prefix}] + list(head["content"])
        messages.extend(chunk_messages)
        return messages

    def _frame_message(self, time_ranges: list[str], frames: list[str], include_query: bool) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if include_query and self._brain.current_query:
            content.append({"type": "text", "text": f"{USER_QUERY_HEADER}\n{self._brain.current_query.strip()}"})
        for tr, url in zip(time_ranges, frames):
            content.append({"type": "text", "text": f"<{tr}>"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": "user", "content": content}
