# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Three-tier memory: working chunk (raw frames) -> mid-term (one text summary
per evicted chunk) -> long-term (mid-terms periodically compressed). Tiers above
the working chunk are text, assembled by ``build_memory_prefix`` into the stable
head so the engine's prefix cache is reused across ticks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_omni.interaction import prompts
from vllm_omni.interaction.backend import ModelBackend

# --------------------------------------------------------------------------- #
# State containers
# --------------------------------------------------------------------------- #


@dataclass
class WorkingChunk:
    """The HTTP transport's live chunk: frames + the multi-turn message buffer."""

    #: ``(time_range, image_data_url)`` for every frame in the chunk.
    frames: list[tuple[str, str]] = field(default_factory=list)
    #: Internal chat messages (user frame turns + assistant replies).
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MidTermSummary:
    chunk_index: int
    frame_range: str
    summary_text: str


@dataclass
class QAEntry:
    query: str
    query_time: str
    responses: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class SessionMemory:
    long_term_memory: str = ""
    mid_term_summaries: list[MidTermSummary] = field(default_factory=list)
    qa_history: list[QAEntry] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Prefix assembly (prefix-cache friendly)
# --------------------------------------------------------------------------- #


def build_memory_prefix(
    memory: SessionMemory,
    *,
    current_query: str | None,
    include_query: bool,
    keep_qa_history: bool,
) -> str:
    """Build the text prefix (video history + Q&A history + current query).

    Returned text is injected ahead of the first frame in the chunk.
    """
    sections: list[str] = []

    history_parts: list[str] = []
    if memory.long_term_memory:
        history_parts.append(memory.long_term_memory)
    for entry in memory.mid_term_summaries:
        history_parts.append(f"<{entry.frame_range}>\n{entry.summary_text}")
    if history_parts:
        sections.append(prompts.VIDEO_HISTORY_HEADER + "\n\n".join(history_parts))

    if keep_qa_history and memory.qa_history:
        lines = []
        for idx, entry in enumerate(memory.qa_history, 1):
            parts = [f"#{idx} [{prompts.QA_QUERY_LABEL}@{entry.query_time or 'N/A'}] {entry.query}"]
            for resp_time, payload in entry.responses:
                parts.append(f"[{prompts.QA_RESPONSE_LABEL}@{resp_time}] {payload}")
            lines.append("\n".join(parts))
        sections.append(prompts.QA_HISTORY_HEADER + "\n".join(lines))

    if include_query and current_query:
        sections.append(prompts.USER_QUERY_HEADER + "\n" + current_query.strip())

    return "\n\n".join(sections)


# --------------------------------------------------------------------------- #
# Summarizer (mid-term + long-term consolidation)
# --------------------------------------------------------------------------- #


def _sample_indices(n: int, budget: int) -> list[int]:
    """Uniformly sample up to ``budget`` indices from ``range(n)``."""
    if n == 0 or budget <= 0:
        return []
    if n <= budget:
        return list(range(n))
    if budget == 1:
        return [n // 2]
    return [round(i * (n - 1) / (budget - 1)) for i in range(budget)]


class Summarizer:
    """Builds mid-term chunk summaries and compresses them into long-term memory.

    Runs against any chat backend (often a small Qwen3-VL served alongside the
    main model, or the main model itself).
    """

    def __init__(
        self,
        backend: ModelBackend,
        *,
        key_frames_per_chunk: int = 8,
        mid_term_max_tokens: int = 1024,
        long_term_max_tokens: int = 1024,
        preferred_time_span: float = 10.0,
    ) -> None:
        self._backend = backend
        self._key_frames = key_frames_per_chunk
        self._mid_max_tokens = mid_term_max_tokens
        self._long_max_tokens = long_term_max_tokens
        self._preferred_time_span = preferred_time_span

    async def summarize_chunk(
        self,
        chunk_index: int,
        frame_range: str,
        frames: list[tuple[str, str]],
    ) -> str:
        """Produce a mid-term text summary for one evicted chunk."""
        if not frames:
            return prompts.EMPTY_CHUNK_SUMMARY.format(frame_range=frame_range)

        picked = [frames[i] for i in _sample_indices(len(frames), self._key_frames)]
        prompt = prompts.MID_TERM_SUMMARY_PROMPT.format(
            chunk_index=chunk_index,
            frame_range=frame_range,
            length_instruction="",
            preferred_time_span=f"{self._preferred_time_span:g} seconds",
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for time_range, data_url in picked:
            content.append({"type": "text", "text": f"<{time_range}>"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        text, _ = await self._backend.generate(
            [{"role": "user", "content": content}],
            max_tokens=self._mid_max_tokens,
            temperature=0.3,
            top_p=0.9,
        )
        return text.strip()

    async def compress_to_long_term(
        self,
        existing_long_term: str,
        mid_terms: list[MidTermSummary],
    ) -> str:
        """Compress new mid-term summaries and append to long-term memory."""
        if not mid_terms:
            return existing_long_term

        merged_range = f"{_range_start(mid_terms[0].frame_range)}-{_range_end(mid_terms[-1].frame_range)}"
        summaries_text = "\n\n".join(f"<{m.frame_range}>\n{m.summary_text}" for m in mid_terms)
        prompt = prompts.LONG_TERM_COMPRESS_PROMPT.format(
            merged_range=merged_range,
            summaries_text=summaries_text,
            length_instruction="",
        )
        compressed, _ = await self._backend.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=self._long_max_tokens,
            temperature=0.3,
            top_p=0.9,
        )
        block = f"<{merged_range}>\n{compressed.strip()}"
        return f"{existing_long_term.rstrip()}\n\n{block}" if existing_long_term else block


def _range_start(frame_range: str) -> str:
    for sep in (" ~ ", "-"):
        if sep in frame_range:
            return frame_range.split(sep, 1)[0].strip()
    return frame_range.strip()


def _range_end(frame_range: str) -> str:
    for sep in (" ~ ", "-"):
        if sep in frame_range:
            return frame_range.split(sep, 1)[-1].strip()
    return frame_range.strip()
