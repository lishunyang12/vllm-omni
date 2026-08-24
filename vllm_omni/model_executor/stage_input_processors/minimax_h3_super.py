# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax H3 draft to LTX-2.5 refiner transition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _original_prompt(prompt: Any) -> dict[str, Any]:
    prompt = _first(prompt)
    if isinstance(prompt, dict):
        return prompt
    if isinstance(prompt, str):
        return {"prompt": prompt}
    if hasattr(prompt, "_asdict"):
        return prompt._asdict()
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    return {}


def h3_to_ltx25_refiner(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> dict[str, Any] | None:
    """Pass decoded H3 tensors to LTX without an intermediate MP4."""
    del requires_multimodal_data
    if not source_outputs:
        return None

    output = source_outputs[0]
    videos = getattr(output, "images", None) or []
    video = _first(videos)
    multimodal_output = getattr(output, "multimodal_output", None)
    multimodal_output = multimodal_output if isinstance(multimodal_output, Mapping) else {}
    audio = multimodal_output.get("audio")
    if video is None or audio is None:
        raise ValueError("MiniMax H3 Super handoff requires both decoded video and audio")

    original = _original_prompt(prompt)
    prompt_text = str(original.get("prompt") or "")
    if not prompt_text:
        raise ValueError("MiniMax H3 Super requires a non-empty prompt")
    multi_modal_data = original.get("multi_modal_data")
    multi_modal_data = multi_modal_data if isinstance(multi_modal_data, Mapping) else {}
    first_frame = _first(multi_modal_data.get("image"))

    return {
        "prompt": prompt_text,
        "additional_information": {
            "h3_video": video,
            "h3_audio": audio,
            "h3_first_frame": first_frame,
            "h3_fps": multimodal_output.get("fps", 24),
            "h3_audio_sample_rate": multimodal_output.get("audio_sample_rate", 32_000),
        },
    }


__all__ = ["h3_to_ltx25_refiner"]
