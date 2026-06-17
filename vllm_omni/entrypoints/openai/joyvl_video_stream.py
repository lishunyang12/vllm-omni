# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import os
from typing import Any

from vllm_omni.entrypoints.openai.video_stream_base import (
    _BAD_FRAME,
    OmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
    VideoStreamTurnTrigger,
)
from vllm_omni.interaction.backend import OpenAIBackend
from vllm_omni.interaction.delegation import StubDelegationBridge
from vllm_omni.interaction.memory import Summarizer
from vllm_omni.interaction.output_parser import ParsedAction
from vllm_omni.interaction.policy import JoyVLPolicy, sample_frames

_DEFAULT_PERSONA = "default"
_FRAME_SECONDS = 1.0
_CHUNK_FRAMES = 16


def _summarizer_from_env() -> Summarizer | None:
    url = os.environ.get("JOYVL_SUMMARIZER_URL")
    if not url:
        return None
    model = os.environ.get("JOYVL_SUMMARIZER_MODEL", "JoyAI-VL-Interaction-Preview")
    return Summarizer(OpenAIBackend(url, model))


class JoyVLStreamingVideoHandler(OmniStreamingVideoHandler):
    persona: str = _DEFAULT_PERSONA
    chunk_frames: int = _CHUNK_FRAMES

    def create_message_history(self, config: StreamingVideoSessionConfig) -> Any:
        return JoyVLPolicy(
            persona=self.persona,
            system_prompt=config.system_prompt,
            num_frames=config.num_frames,
            chunk_frames=self.chunk_frames,
            frame_seconds=_FRAME_SECONDS,
            summarizer=_summarizer_from_env(),
            delegation=StubDelegationBridge(),
        )

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        return not trigger.is_generating and trigger.frame_count >= 1

    def on_frame_buffered(self, raw_bytes: bytes, frame_b64: str, message_history: Any, config) -> None:
        if isinstance(message_history, JoyVLPolicy):
            time_range = message_history.brain.now()
            message_history.observe(time_range, f"data:image/jpeg;base64,{frame_b64}")

    def build_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: Any,
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        policy: JoyVLPolicy = message_history
        policy.set_query(query_text)
        return policy.build_messages(self._frame_parts(frame_buffer, policy.num_frames, prewarmed_frames))

    def on_turn_complete(self, message_history: Any, user_message: dict[str, Any], response_text: str) -> None:
        policy: JoyVLPolicy = message_history
        action = policy.commit(response_text)

        asyncio.create_task(self._post_turn(policy, action))

    async def _post_turn(self, policy: JoyVLPolicy, action: ParsedAction) -> None:
        await policy.submit_if_delegate(action, list(policy.working_frames))
        await policy.fold_delegations()
        if policy.needs_flush():
            await policy.consolidate(policy.close_chunk(), policy.take_working_frames())

    def _frame_parts(
        self,
        frame_buffer: list[str],
        num_frames: int,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> list[dict[str, Any]]:
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
