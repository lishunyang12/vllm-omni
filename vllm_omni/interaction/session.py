# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-session interaction loop. Each tick ingests one or more frames plus an
optional user query, decides speak/silence/delegate, and maintains the three-tier
memory and pending delegations.

Message layout is stable-head / append-only for prefix-cache reuse: the head
(system prompt + memory prefix) only changes at chunk-flush boundaries; a newly
issued query rides in that tick's appended message rather than the head, and
moves into the head once its chunk is evicted."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.interaction.backend import ModelBackend
from vllm_omni.interaction.config import InteractionConfig
from vllm_omni.interaction.delegation import DelegationBridge
from vllm_omni.interaction.memory import (
    MidTermSummary,
    QAEntry,
    SessionMemory,
    Summarizer,
    WorkingChunk,
    build_memory_prefix,
)
from vllm_omni.interaction.output_parser import Action, ParsedAction, parse_action
from vllm_omni.interaction.prompts import SYSTEM_PROMPTS, USER_QUERY_HEADER


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
        self._summarizer = summarizer
        self._delegation = delegation

        self.memory = SessionMemory()
        self.chunk = WorkingChunk()
        self.chunk_index = 1
        self.frame_index = 0

        self.current_query: str | None = None
        self.query_time: str | None = None
        self.query_in_current_chunk = False
        self._pending_delegations: list[dict[str, str]] = []
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

        base = t if t is not None else self.frame_index * self.config.frame_seconds
        time_ranges = [f"{base + i * self.config.frame_seconds:.1f}s" for i in range(len(frames))]
        time_range = time_ranges[-1] if time_ranges else f"{base:.1f}s"

        query_is_fresh = self._update_query(query, time_ranges[0] if time_ranges else time_range)
        delegation_info = await self._fold_delegations()

        if self.config.chunk_frames > 0 and self.chunk.frame_count >= self.config.chunk_frames:
            await self._flush_chunk()

        for tr, url in zip(time_ranges, frames):
            self.frame_index += 1
            self.chunk.frames.append((tr, url))
        self.chunk.messages.append(self._frame_message(time_ranges, frames, include_query=query_is_fresh))

        if self.config.force_silence_before_query and not self.current_query:
            action = ParsedAction(Action.SILENCE, raw="</silence>")
            skipped = True
        else:
            action = await self._infer()
            skipped = False
        self.chunk.messages.append({"role": "assistant", "content": action.raw or "</silence>"})

        if action.spoke and self.current_query:
            self.chunk.response_records.append((time_range, action.text))
        if action.action is Action.DELEGATE and self._delegation and self.config.enable_delegation:
            task_id = await self._delegation.submit(
                action.delegated_question or "", action.text, list(self.chunk.frames)
            )
            self._pending_delegations.append({"task_id": task_id, "question": action.delegated_question or ""})
            delegation_info = {"task_id": task_id, "status": "submitted", "question": action.delegated_question}

        return StepResult(
            action=action,
            chunk_index=self.chunk_index,
            frame_index=self.frame_index,
            inference_skipped=skipped,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            long_term_memory=self.memory.long_term_memory,
            mid_term_summaries=[
                {"chunk_index": m.chunk_index, "frame_range": m.frame_range, "summary_text": m.summary_text}
                for m in self.memory.mid_term_summaries
            ],
            delegation=delegation_info,
        )

    def reset(self) -> None:
        self.memory = SessionMemory()
        self.chunk = WorkingChunk()
        self.chunk_index = 1
        self.frame_index = 0
        self.current_query = None
        self.query_time = None
        self.query_in_current_chunk = False
        self._pending_delegations.clear()

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

        prefix = build_memory_prefix(
            self.memory,
            current_query=self.current_query,
            include_query=self.current_query is not None and not self.query_in_current_chunk,
            keep_qa_history=self.config.keep_qa_history,
        )
        if prefix and chunk_messages:
            head = chunk_messages[0]
            head["content"] = [{"type": "text", "text": prefix}] + list(head["content"])
        messages.extend(chunk_messages)
        return messages

    def _frame_message(self, time_ranges: list[str], frames: list[str], include_query: bool) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if include_query and self.current_query:
            content.append({"type": "text", "text": f"{USER_QUERY_HEADER}\n{self.current_query.strip()}"})
        for tr, url in zip(time_ranges, frames):
            content.append({"type": "text", "text": f"<{tr}>"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": "user", "content": content}

    def _update_query(self, query: str | None, time_range: str) -> bool:
        q = (query or "").strip()
        if not q:
            return False
        if self.current_query is None:
            self.current_query, self.query_time, self.query_in_current_chunk = q, time_range, True
            return True
        if q != self.current_query:
            self._archive_query()
            self.current_query, self.query_time, self.query_in_current_chunk = q, time_range, True
            return True
        return False

    def _archive_query(self) -> None:
        if self.current_query and self.chunk.response_records:
            self.memory.qa_history.append(
                QAEntry(self.current_query, self.query_time or "", list(self.chunk.response_records))
            )
        self.chunk.response_records = []

    async def _flush_chunk(self) -> None:
        self._archive_query()
        if self._summarizer is not None and self.chunk.frames:
            frame_range = f"{self.chunk.frames[0][0]}-{self.chunk.frames[-1][0]}"
            summary = await self._summarizer.summarize_chunk(self.chunk_index, frame_range, self.chunk.frames)
            self.memory.mid_term_summaries.append(MidTermSummary(self.chunk_index, frame_range, summary))
            if len(self.memory.mid_term_summaries) >= self.config.long_term_every_n_chunks:
                self.memory.long_term_memory = await self._summarizer.compress_to_long_term(
                    self.memory.long_term_memory, self.memory.mid_term_summaries
                )
                self.memory.mid_term_summaries.clear()

        self.chunk = WorkingChunk()
        self.chunk_index += 1
        self.query_in_current_chunk = False

    async def _fold_delegations(self) -> dict[str, Any] | None:
        if not self._pending_delegations or self._delegation is None:
            return None
        folded: dict[str, Any] | None = None
        still_pending: list[dict[str, str]] = []
        for task in self._pending_delegations:
            result = await self._delegation.poll(task["task_id"])
            if result.is_ready:
                self.memory.qa_history.append(
                    QAEntry(f"[delegated] {task['question']}", self.query_time or "", [("", result.digest)])
                )
                folded = {
                    "task_id": result.task_id,
                    "status": "ready",
                    "question": task["question"],
                    "digest": result.digest,
                }
            elif result.status == "error":
                folded = {"task_id": result.task_id, "status": "error", "question": task["question"]}
            else:
                still_pending.append(task)
        self._pending_delegations = still_pending
        return folded
