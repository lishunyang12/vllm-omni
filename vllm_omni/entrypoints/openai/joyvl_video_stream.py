# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JoyVL streaming-video handler on the shared OmniStreamingVideoHandler base.

JoyVL is proactive: every free tick it decides speak / silence / delegate from
the control tokens, instead of waiting for a ``video.query``. The decision lives
in the model; this handler only triggers per tick, assembles a stable-head /
append-only prompt (system + memory prefix, then the changing frames) for
prefix-cache reuse, and folds spoken turns into a Q&A memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_omni.entrypoints.openai.video_stream_base import (
    _BAD_FRAME,
    OmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
    VideoStreamTurnTrigger,
)
from vllm_omni.interaction.memory import QAEntry, SessionMemory, build_memory_prefix
from vllm_omni.interaction.output_parser import Action, parse_action
from vllm_omni.interaction.prompts import SYSTEM_PROMPTS, USER_QUERY_HEADER

_DEFAULT_PERSONA = "default"
_FRAME_SECONDS = 1.0


@dataclass
class JoyVLSessionState:
    """Per-session JoyVL state (returned by ``create_message_history``)."""

    memory: SessionMemory = field(default_factory=SessionMemory)
    current_query: str | None = None
    query_time: str | None = None
    query_in_current_chunk: bool = False
    frame_index: int = 0


class JoyVLStreamingVideoHandler(OmniStreamingVideoHandler):
    """Proactive JoyVL pipeline on the shared streaming-video endpoint."""

    persona: str = _DEFAULT_PERSONA

    # ----- pipeline hooks ------------------------------------------------- #

    def create_message_history(self, config: StreamingVideoSessionConfig) -> Any:
        return JoyVLSessionState()

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        # Proactive: run a turn whenever the model is free and a frame exists.
        # The model itself emits </silence> when there is nothing to say, so we
        # do not gate on a pending user query.
        return not trigger.is_generating and trigger.frame_count >= 1

    def on_frame_buffered(self, raw_bytes: bytes, frame_b64: str, message_history: Any, config) -> None:
        if isinstance(message_history, JoyVLSessionState):
            message_history.frame_index += 1

    def build_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: Any,
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        state: JoyVLSessionState = message_history
        self._update_query(state, query_text)

        system_prompt = config.system_prompt or SYSTEM_PROMPTS.get(self.persona, SYSTEM_PROMPTS["default"])
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        user_content: list[dict[str, Any]] = []
        # Stable head: long-term + mid-term + Q&A history (+ carried-over query).
        prefix = build_memory_prefix(
            state.memory,
            current_query=state.current_query,
            include_query=bool(state.current_query) and not state.query_in_current_chunk,
            keep_qa_history=True,
        )
        if prefix:
            user_content.append({"type": "text", "text": prefix})

        for part in self._frame_parts(frame_buffer, config, prewarmed_frames):
            user_content.append(part)

        # A freshly-issued query rides in this turn's message (append-only),
        # moving into the stable head once its chunk is evicted.
        if state.current_query and state.query_in_current_chunk:
            user_content.append({"type": "text", "text": f"{USER_QUERY_HEADER}\n{state.current_query}"})

        user_message = {"role": "user", "content": user_content}
        messages.append(user_message)
        return messages, user_message

    def on_turn_complete(self, message_history: Any, user_message: dict[str, Any], response_text: str) -> None:
        state: JoyVLSessionState = message_history
        action = parse_action(response_text)
        if action.action is Action.SILENCE:
            return
        # Record what was spoken against the active query for long-horizon recall.
        if state.current_query:
            self._record_response(state, action.text)
        state.query_in_current_chunk = False

    # ----- helpers -------------------------------------------------------- #

    def _frame_parts(
        self,
        frame_buffer: list[str],
        config: StreamingVideoSessionConfig,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> list[dict[str, Any]]:
        n = len(frame_buffer)
        if n <= config.num_frames:
            frames = list(frame_buffer)
        else:
            stride = max(1, n // config.num_frames)
            idx = [i * stride for i in range(config.num_frames - 1)] + [n - 1]
            frames = [frame_buffer[i] for i in idx]

        prewarmed = prewarmed_frames or {}
        parts: list[dict[str, Any]] = []
        for frame_b64 in frames:
            cached = prewarmed.get(frame_b64)
            if cached is _BAD_FRAME:
                continue
            if cached is not None:
                pil, pil_uuid = cached
                parts.append({"type": "image_pil", "image_pil": pil, "uuid": pil_uuid})
            else:
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})
        return parts

    def _update_query(self, state: JoyVLSessionState, query_text: str) -> None:
        query = (query_text or "").strip()
        if not query or query == state.current_query:
            return
        time_range = f"{state.frame_index * _FRAME_SECONDS:.1f}s"
        state.current_query = query
        state.query_time = time_range
        state.query_in_current_chunk = True

    def _record_response(self, state: JoyVLSessionState, text: str) -> None:
        if not text:
            return
        time_range = f"{state.frame_index * _FRAME_SECONDS:.1f}s"
        history = state.memory.qa_history
        if history and history[-1].query == state.current_query:
            history[-1].responses.append((time_range, text))
        else:
            history.append(QAEntry(state.current_query or "", state.query_time or time_range, [(time_range, text)]))
