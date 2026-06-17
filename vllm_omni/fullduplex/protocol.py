# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Realtime event vocabulary for the duplex runtime.

Aligned with the OpenAI Realtime-style events the demos use; transports
(``/v1/duplex``, ``/v1/realtime?duplex=1``) translate their wire JSON to/from
these. Kept as plain string types + dict builders so any transport can map them."""

from __future__ import annotations

from typing import Any

# client -> server
INPUT_APPEND = "input.append"  # {"modality","data"}
INPUT_COMMIT = "input.commit"
RESPONSE_CREATE = "response.create"
RESPONSE_CANCEL = "response.cancel"  # barge-in
PLAYBACK_ACK = "playback.ack"  # {"cursor"}
CLOSE = "close"

# server -> client
RESPONSE_CREATED = "response.created"
RESPONSE_DELTA = "response.delta"  # {"modality","data"}; audio uses response.audio.delta downstream
RESPONSE_DONE = "response.done"
RESPONSE_CANCELLED = "response.cancelled"
ERROR = "error"


def created(response_index: int) -> dict[str, Any]:
    return {"type": RESPONSE_CREATED, "response_index": response_index}


def delta(response_index: int, modality: str, data: Any) -> dict[str, Any]:
    return {"type": RESPONSE_DELTA, "response_index": response_index, "modality": modality, "data": data}


def done(response_index: int) -> dict[str, Any]:
    return {"type": RESPONSE_DONE, "response_index": response_index}


def cancelled(response_index: int) -> dict[str, Any]:
    return {"type": RESPONSE_CANCELLED, "response_index": response_index}


def error(message: str) -> dict[str, Any]:
    return {"type": ERROR, "message": message}
