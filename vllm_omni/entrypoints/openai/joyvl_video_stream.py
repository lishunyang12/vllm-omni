# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JoyVL streaming-video handler on the shared OmniStreamingVideoHandler base.

JoyVL is proactive: every free tick it decides speak / silence / delegate from
the control tokens, instead of waiting for a ``video.query``. The decision lives
in the model; this handler only triggers per tick, assembles a stable-head /
append-only prompt (system + memory prefix, then the changing frames) for
prefix-cache reuse, and folds spoken turns into Q&A memory via SessionBrain."""

from __future__ import annotations

from typing import Any

from vllm_omni.entrypoints.openai.video_stream_base import (
    _BAD_FRAME,
    OmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
    VideoStreamTurnTrigger,
)
from vllm_omni.interaction.output_parser import Action, parse_action
from vllm_omni.interaction.prompts import SYSTEM_PROMPTS, USER_QUERY_HEADER
from vllm_omni.interaction.state import SessionBrain

_DEFAULT_PERSONA = "default"
_FRAME_SECONDS = 1.0


class JoyVLStreamingVideoHandler(OmniStreamingVideoHandler):
    """Proactive JoyVL pipeline on the shared streaming-video endpoint."""

    persona: str = _DEFAULT_PERSONA

    # ----- pipeline hooks ------------------------------------------------- #

    def create_message_history(self, config: StreamingVideoSessionConfig) -> Any:
        return SessionBrain(frame_seconds=_FRAME_SECONDS)

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        # Proactive: run a turn whenever the model is free and a frame exists.
        # The model itself emits </silence> when there is nothing to say, so we
        # do not gate on a pending user query.
        return not trigger.is_generating and trigger.frame_count >= 1

    def on_frame_buffered(self, raw_bytes: bytes, frame_b64: str, message_history: Any, config) -> None:
        if isinstance(message_history, SessionBrain):
            message_history.tick()

    def build_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: Any,
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        brain: SessionBrain = message_history
        brain.update_query(query_text)

        system_prompt = config.system_prompt or SYSTEM_PROMPTS.get(self.persona, SYSTEM_PROMPTS["default"])
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        user_content: list[dict[str, Any]] = []
        prefix = brain.build_prefix()
        if prefix:
            user_content.append({"type": "text", "text": prefix})

        user_content.extend(self._frame_parts(frame_buffer, config, prewarmed_frames))

        # A freshly-issued query rides in this turn's message (append-only),
        # moving into the stable head once its chunk is evicted.
        if brain.current_query and brain.query_in_current_chunk:
            user_content.append({"type": "text", "text": f"{USER_QUERY_HEADER}\n{brain.current_query}"})

        user_message = {"role": "user", "content": user_content}
        messages.append(user_message)
        return messages, user_message

    def on_turn_complete(self, message_history: Any, user_message: dict[str, Any], response_text: str) -> None:
        brain: SessionBrain = message_history
        action = parse_action(response_text)
        if action.action is not Action.SILENCE and brain.current_query:
            brain.record_response(action.text)
        brain.end_turn()

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
