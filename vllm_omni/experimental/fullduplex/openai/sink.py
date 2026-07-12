# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from vllm_omni.experimental.fullduplex.core.events import EmitProtocolEvent
from vllm_omni.experimental.fullduplex.openai.history import ResponseLifecycleLedger
from vllm_omni.experimental.fullduplex.openai.realtime import RealtimeEventProjector

SendJson = Callable[[dict[str, object]], Awaitable[None]]


class _Projector(Protocol):
    def project(self, effect: EmitProtocolEvent) -> tuple[dict[str, object], ...]: ...


class WebSocketRealtimeSink:
    """Project fenced core effects to OpenAI-Realtime frames over a WebSocket.

    Implements :class:`~vllm_omni.experimental.fullduplex.core.ports.DuplexEventSink`.
    The runtime hands every :class:`EmitProtocolEvent` to :meth:`emit`; the
    projector turns it into zero or more wire frames (``response.created``,
    ``response.output_text.delta``, ``response.output_audio.delta``,
    ``response.output_audio.done``, ``response.done``, ``response.listen``,
    ``error``) which are sent in order. Response-lifecycle bookkeeping lives in
    the shared :class:`ResponseLifecycleLedger`, so identity never leaks into the
    domain state.
    """

    def __init__(
        self,
        send_json: SendJson,
        *,
        lifecycle: ResponseLifecycleLedger | None = None,
        projector: _Projector | None = None,
    ) -> None:
        self._send_json = send_json
        self._projector = projector or RealtimeEventProjector(lifecycle=lifecycle)

    async def emit(self, event: EmitProtocolEvent) -> None:
        for frame in self._projector.project(event):
            await self._send_json(frame)


__all__ = ["SendJson", "WebSocketRealtimeSink"]
