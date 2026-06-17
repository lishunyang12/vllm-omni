# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport-agnostic interaction core.

``InteractionBrain`` owns everything that is *not* prompt assembly or inference:
the three-tier memory, the current-query / Q&A bookkeeping, chunk-flush +
summarization, and the delegation bridge. Both the HTTP session and the
streaming-video handler delegate to it, so the per-tick state machine (and the
sliding QA history the prefix cache reuses) lives in exactly one place.

It is frame-agnostic: each transport manages its own frame buffer and passes the
frames to summarize into :meth:`flush`. ``flush`` is async (summarization calls a
model); the HTTP path ``await``s it, the sync WS hooks schedule it as a task.
"""

from __future__ import annotations

from typing import Any

from vllm_omni.interaction.delegation import DelegationBridge
from vllm_omni.interaction.memory import (
    MidTermSummary,
    QAEntry,
    SessionMemory,
    Summarizer,
    build_memory_prefix,
)
from vllm_omni.interaction.output_parser import Action, ParsedAction


class InteractionBrain:
    def __init__(
        self,
        *,
        summarizer: Summarizer | None = None,
        delegation: DelegationBridge | None = None,
        chunk_frames: int = 200,
        long_term_every_n_chunks: int = 5,
        keep_qa_history: bool = True,
        frame_seconds: float = 1.0,
        enable_delegation: bool = True,
    ) -> None:
        self._summarizer = summarizer
        self._delegation = delegation
        self._chunk_frames = chunk_frames
        self._long_term_every_n = long_term_every_n_chunks
        self._keep_qa_history = keep_qa_history
        self._frame_seconds = frame_seconds
        self._enable_delegation = enable_delegation
        self.reset()

    def reset(self) -> None:
        self.memory = SessionMemory()
        self.current_query: str | None = None
        self.query_time: str | None = None
        self.query_in_current_chunk = False
        self.frame_index = 0
        self.chunk_index = 1
        self._chunk_frame_count = 0
        self.response_records: list[tuple[str, str]] = []
        self._pending_delegations: list[dict[str, str]] = []

    # ----- time / frames -------------------------------------------------- #

    def now(self) -> str:
        """Timestamp of the next frame (used for a query as it arrives)."""
        return f"{self.frame_index * self._frame_seconds:.1f}s"

    def last_frame_time(self) -> str:
        """Timestamp of the most recently seen frame (used for a response)."""
        return f"{max(0, self.frame_index - 1) * self._frame_seconds:.1f}s"

    def tick(self, n: int = 1) -> None:
        self.frame_index += n
        self._chunk_frame_count += n

    def should_flush(self) -> bool:
        return self._chunk_frames > 0 and self._chunk_frame_count >= self._chunk_frames

    # ----- query / Q&A ---------------------------------------------------- #

    def update_query(self, query: str | None) -> bool:
        """Adopt a new/changed query; archive the previous one. Returns True if fresh."""
        q = (query or "").strip()
        if not q or q == self.current_query:
            return False
        if self.current_query is not None:
            self.archive_query()
        self.current_query = q
        self.query_time = self.now()
        self.query_in_current_chunk = True
        return True

    def archive_query(self) -> None:
        if self.current_query and self.response_records:
            self.memory.qa_history.append(
                QAEntry(self.current_query, self.query_time or "", list(self.response_records))
            )
        self.response_records = []

    def record_response(self, text: str) -> None:
        if text and self.current_query:
            self.response_records.append((self.last_frame_time(), text))

    def build_prefix(self) -> str:
        return build_memory_prefix(
            self.memory,
            current_query=self.current_query,
            include_query=self.current_query is not None and not self.query_in_current_chunk,
            keep_qa_history=self._keep_qa_history,
        )

    # ----- chunk flush / summarization ------------------------------------ #

    async def flush(self, frames: list[tuple[str, str]]) -> None:
        """Evict the current chunk: archive Q&A and (if a summarizer is wired)
        fold the chunk's frames into mid-/long-term memory."""
        self.archive_query()
        if self._summarizer is not None and frames:
            frame_range = f"{frames[0][0]}-{frames[-1][0]}"
            summary = await self._summarizer.summarize_chunk(self.chunk_index, frame_range, frames)
            self.memory.mid_term_summaries.append(MidTermSummary(self.chunk_index, frame_range, summary))
            if len(self.memory.mid_term_summaries) >= self._long_term_every_n:
                self.memory.long_term_memory = await self._summarizer.compress_to_long_term(
                    self.memory.long_term_memory, self.memory.mid_term_summaries
                )
                self.memory.mid_term_summaries.clear()
        self.chunk_index += 1
        self._chunk_frame_count = 0
        self.query_in_current_chunk = False

    # ----- delegation ----------------------------------------------------- #

    async def submit_delegation(self, action: ParsedAction, frames: list[tuple[str, str]]) -> dict[str, Any] | None:
        if action.action is not Action.DELEGATE or self._delegation is None or not self._enable_delegation:
            return None
        task_id = await self._delegation.submit(action.delegated_question or "", action.text, frames)
        self._pending_delegations.append({"task_id": task_id, "question": action.delegated_question or ""})
        return {"task_id": task_id, "status": "submitted", "question": action.delegated_question}

    async def fold_delegations(self) -> dict[str, Any] | None:
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
