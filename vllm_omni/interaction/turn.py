# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared JoyVL turn construction.

Every JoyVL transport (HTTP session, streaming-video handler, full-duplex
adapter) builds the same per-tick prompt — stable head (memory prefix) + frames
+ the active query — and commits the model's output to memory the same way. That
policy lives here so the transports differ only in how they obtain frames and
run inference, not in the JoyVL logic itself."""

from __future__ import annotations

from typing import Any

from vllm_omni.interaction.output_parser import Action, ParsedAction, parse_action
from vllm_omni.interaction.prompts import USER_QUERY_HEADER
from vllm_omni.interaction.state import InteractionBrain


def sample_frames(frames: list[str], num_frames: int) -> list[str]:
    """Uniformly sample up to ``num_frames`` frames (keeping the most recent)."""
    n = len(frames)
    if n <= num_frames:
        return list(frames)
    stride = max(1, n // num_frames)
    idx = [i * stride for i in range(num_frames - 1)] + [n - 1]
    return [frames[i] for i in idx]


def build_user_content(brain: InteractionBrain, frame_parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable head (memory prefix) + frame parts + the active query.

    The query rides here while its chunk is live (append-only) and moves into the
    head prefix once the chunk is evicted — the layout that keeps the prefix cache
    warm across ticks.
    """
    content: list[dict[str, Any]] = []
    prefix = brain.build_prefix()
    if prefix:
        content.append({"type": "text", "text": prefix})
    content.extend(frame_parts)
    if brain.current_query and brain.query_in_current_chunk:
        content.append({"type": "text", "text": f"{USER_QUERY_HEADER}\n{brain.current_query}"})
    return content


def commit_turn(brain: InteractionBrain, response_text: str) -> ParsedAction:
    """Parse the model output and fold a spoken response into Q&A memory."""
    action = parse_action(response_text)
    if action.action is not Action.SILENCE and brain.current_query:
        brain.record_response(action.text)
    return action
