# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared per-session interaction state.

``SessionBrain`` holds the three-tier memory plus the current-query / Q&A
bookkeeping, and produces the stable-head prefix (system memory + Q&A history +
carried-over query) used for prefix-cache reuse. Both the streaming-video
handler and the standalone HTTP session build on it so the per-tick logic lives
in one place."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.interaction.memory import QAEntry, SessionMemory, build_memory_prefix


@dataclass
class SessionBrain:
    memory: SessionMemory = field(default_factory=SessionMemory)
    current_query: str | None = None
    query_time: str | None = None
    #: True while the active query was issued in the current (not-yet-evicted)
    #: chunk; such a query rides in the turn message, otherwise in the head.
    query_in_current_chunk: bool = False
    frame_index: int = 0
    frame_seconds: float = 1.0

    def tick(self) -> None:
        self.frame_index += 1

    def now(self) -> str:
        return f"{self.frame_index * self.frame_seconds:.1f}s"

    def update_query(self, query: str | None) -> bool:
        """Adopt a new/changed query. Returns True if it is fresh this turn."""
        q = (query or "").strip()
        if not q or q == self.current_query:
            return False
        self.current_query = q
        self.query_time = self.now()
        self.query_in_current_chunk = True
        return True

    def build_prefix(self, keep_qa_history: bool = True) -> str:
        return build_memory_prefix(
            self.memory,
            current_query=self.current_query,
            include_query=bool(self.current_query) and not self.query_in_current_chunk,
            keep_qa_history=keep_qa_history,
        )

    def record_response(self, text: str) -> None:
        """Append a spoken response to the active query's Q&A entry."""
        if not text:
            return
        now = self.now()
        history = self.memory.qa_history
        if history and history[-1].query == self.current_query:
            history[-1].responses.append((now, text))
        else:
            history.append(QAEntry(self.current_query or "", self.query_time or now, [(now, text)]))

    def end_turn(self) -> None:
        """A query becomes part of the stable head once its turn is done."""
        self.query_in_current_chunk = False
