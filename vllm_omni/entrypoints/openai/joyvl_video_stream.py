# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JoyVL streaming-video handler on the shared OmniStreamingVideoHandler base.

JoyVL is proactive: every free tick it decides speak / silence / delegate from
the control tokens, instead of waiting for a ``video.query``. The decision lives
in the model; this handler only triggers per tick, assembles a stable-head /
append-only prompt (system + memory prefix, then the changing frames) for
prefix-cache reuse, and folds spoken turns into Q&A memory via InteractionBrain."""

from __future__ import annotations

import asyncio
from typing import Any

from vllm_omni.entrypoints.openai.video_stream_base import (
    _BAD_FRAME,
    OmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
    VideoStreamTurnTrigger,
)
from vllm_omni.interaction.prompts import SYSTEM_PROMPTS
from vllm_omni.interaction.state import InteractionBrain
from vllm_omni.interaction.turn import build_user_content, commit_turn, sample_frames

_DEFAULT_PERSONA = "default"
_FRAME_SECONDS = 1.0
_CHUNK_FRAMES = 200


class JoyVLStreamingVideoHandler(OmniStreamingVideoHandler):
    """Proactive JoyVL pipeline on the shared streaming-video endpoint."""

    persona: str = _DEFAULT_PERSONA
    chunk_frames: int = _CHUNK_FRAMES

    # ----- pipeline hooks ------------------------------------------------- #

    def create_message_history(self, config: StreamingVideoSessionConfig) -> Any:
        # No summarizer wired on the WS base yet, so flush archives Q&A only
        # (mid/long-term activate once a summarizer is injected here).
        return InteractionBrain(chunk_frames=self.chunk_frames, frame_seconds=_FRAME_SECONDS)

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        # Proactive: run a turn whenever the model is free and a frame exists.
        # The model itself emits </silence> when there is nothing to say, so we
        # do not gate on a pending user query.
        return not trigger.is_generating and trigger.frame_count >= 1

    def on_frame_buffered(self, raw_bytes: bytes, frame_b64: str, message_history: Any, config) -> None:
        if isinstance(message_history, InteractionBrain):
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
        brain: InteractionBrain = message_history
        brain.update_query(query_text)
        system_prompt = config.system_prompt or SYSTEM_PROMPTS.get(self.persona, SYSTEM_PROMPTS["default"])
        parts = self._frame_parts(frame_buffer, config.num_frames, prewarmed_frames)
        user_message = {"role": "user", "content": build_user_content(brain, parts)}
        return [{"role": "system", "content": system_prompt}, user_message], user_message

    def on_turn_complete(self, message_history: Any, user_message: dict[str, Any], response_text: str) -> None:
        brain: InteractionBrain = message_history
        commit_turn(brain, response_text)
        if brain.should_flush():
            # Async so chunk summarization (when a summarizer is wired) hides
            # behind the next turn's inference rather than blocking it.
            asyncio.create_task(brain.flush([]))

    # ----- helpers -------------------------------------------------------- #

    def _frame_parts(
        self,
        frame_buffer: list[str],
        num_frames: int,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> list[dict[str, Any]]:
        # Frame parts are transport-specific: this path can reuse prewarmed PIL
        # decodes (image_pil) to skip re-decoding base64 at query time.
        prewarmed = prewarmed_frames or {}
        parts: list[dict[str, Any]] = []
        for frame_b64 in sample_frames(frame_buffer, num_frames):
            cached = prewarmed.get(frame_b64)
            if cached is _BAD_FRAME:
                continue
            if cached is not None:
                pil, pil_uuid = cached
                parts.append({"type": "image_pil", "image_pil": pil, "uuid": pil_uuid})
            else:
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})
        return parts
