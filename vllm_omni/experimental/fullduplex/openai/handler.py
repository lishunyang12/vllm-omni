# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from vllm_omni.experimental.fullduplex.core.continuous import (
    ContinuousDuplexState,
    reduce_continuous_event,
)
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime
from vllm_omni.experimental.fullduplex.engine.omni import (
    DuplexAdapterPattern,
    DuplexInputMode,
    DuplexRuntimeCapabilities,
    DuplexSignalSource,
    OmniDuplexEnginePort,
)
from vllm_omni.experimental.fullduplex.minicpmo45.adapter import (
    MiniCPMO45ModelEventAdapter,
    MiniCPMO45NativeDuplexServingAdapter,
)
from vllm_omni.experimental.fullduplex.openai.inbound import (
    DEFAULT_CHUNK_PERIOD_MS,
    RealtimeInboundAdapter,
)
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexSessionConfig
from vllm_omni.experimental.fullduplex.openai.realtime import DemoRealtimeProjector
from vllm_omni.experimental.fullduplex.openai.sink import WebSocketRealtimeSink

logger = logging.getLogger(__name__)

SendJson = Callable[[dict[str, object]], Awaitable[None]]


class RuntimeDuplexHandler:
    """Drive one MiniCPM-o full-duplex session through the Track A FSM runtime.

    This is the continuous full-duplex "front door": it wires the inbound
    WebSocket frame stream into typed ``DomainEvent``s, runs them through
    :class:`DuplexRuntime` with :func:`reduce_continuous_event` over
    :class:`OmniDuplexEnginePort`, and projects the resulting effects back to the
    compact realtime dialect the bundled ``realtime_web`` demo consumes.

    ``run`` is transport-agnostic (async frame iterator + ``send_json`` callable)
    so it can be exercised without a live socket; :meth:`handle_realtime_session`
    is the thin FastAPI WebSocket adapter that also resolves the opening
    ``session.update`` (voice / reference audio) into the engine session config.
    """

    def __init__(
        self,
        engine: object,
        *,
        model: str | None = None,
        model_config: Any = None,
        chunk_period_ms: int = DEFAULT_CHUNK_PERIOD_MS,
        engine_timeout: float | None = 10.0,
    ) -> None:
        self._engine = engine
        self._model = model
        self._model_config = model_config
        self._chunk_period_ms = chunk_period_ms
        self._engine_timeout = engine_timeout

    def _new_session_id(self) -> str:
        return f"sess_{uuid.uuid4().hex[:12]}"

    async def run(
        self,
        recv_frames: AsyncIterator[dict],
        send_json: SendJson,
        *,
        session_id: str | None = None,
        session_config: dict[str, object] | None = None,
        input_audio_format: str = "pcm_f32le",
    ) -> str:
        session_id = session_id or self._new_session_id()
        await send_json({"type": "session.created", "session": {"id": session_id}})
        # The worker only attaches the native Stage0/Stage1 duplex runtimes when
        # the session opens with implementation_level="model_native_duplex"
        # (worker.py:maybe_load_minicpmo_native_duplex_target). Without it the
        # model runs an uninitialized path and produces NaN logits.
        native_caps = DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
            adapter_patterns={DuplexAdapterPattern.SCHEDULER_DATA_PLANE},
            signal_sources={
                DuplexSignalSource.MODEL_NATIVE,
                DuplexSignalSource.CLIENT_EVENT,
                DuplexSignalSource.SERVER_POLICY,
            },
            supports_model_internal_state=True,
            supports_stage_resumption=True,
            supports_scheduler_native_append=True,
            supports_core_resumable_request=True,
            supports_stage_connector_handoff=True,
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            implementation_level="model_native_duplex",
            stage_handoff_transport="scheduler_data_plane",
            chunk_period_ms=self._chunk_period_ms,
            target_barge_in_latency_ms=1000,
        )
        port = OmniDuplexEnginePort(
            self._engine,
            capabilities=native_caps,
            session_config=dict(session_config or {}),
            input_mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            output_mapper=MiniCPMO45ModelEventAdapter().map_output,
            timeout=self._engine_timeout,
        )
        # The bundled realtime_web demo (app.js) speaks the compact dialect.
        sink = WebSocketRealtimeSink(send_json, projector=DemoRealtimeProjector())
        runtime = DuplexRuntime(
            session_id,
            port,
            sink,
            reduce=reduce_continuous_event,
            initial_state=ContinuousDuplexState.open(session_id),
        )
        inbound = RealtimeInboundAdapter(
            chunk_period_ms=self._chunk_period_ms,
            input_audio_format=input_audio_format,
        )
        await runtime.run(inbound.events(recv_frames))
        return session_id

    async def _prepare_session_config(self, session: dict) -> dict[str, object]:
        """Resolve a client ``session.update`` into an engine session config.

        Mirrors the Track B serving path: build a :class:`DuplexSessionConfig`
        and run the MiniCPM-o native adapter, which folds voice / reference-audio
        and scheduler policy into ``extra_body``. Best-effort: on any failure the
        raw ``extra_body`` (plus voice) is used so the session can still open.
        """
        extra_body = dict(session.get("extra_body") or {})
        voice = session.get("voice")
        # Mirror Track B (DuplexSessionConfig.from_event): the reference-audio
        # default is resolved from `config.model`/`model_config.model`, so the
        # session-body model path must win when the client sends one; only fall
        # back to the server launch model when it does not.
        body_model = session.get("model")
        config = DuplexSessionConfig(
            model=body_model if isinstance(body_model, str) and body_model else self._model,
            instructions=session.get("instructions"),
            voice=voice if isinstance(voice, str) else None,
            extra_body=extra_body,
        )
        try:
            await MiniCPMO45NativeDuplexServingAdapter.prepare_session_config(config, model_config=self._model_config)
        except Exception as exc:  # noqa: BLE001 - degrade to a raw config, never fail the demo open
            logger.warning("native duplex session prep failed; using raw config: %s", exc)
        # The engine expects the full DuplexSessionConfig shape: the prepared
        # values (ref_audio_data, context tokens, scheduler/sampling policy) live
        # NESTED under `extra_body`, not flattened at the top level. Track B opens
        # with `config.as_dict()`; matching it is required or the model can't find
        # the ref-audio context and produces NaN logits.
        return config.as_dict()

    async def handle_realtime_session(self, websocket: object) -> None:
        """FastAPI WebSocket adapter: accept, resolve session.update, run."""
        await websocket.accept()  # type: ignore[attr-defined]

        from starlette.websockets import WebSocketDisconnect

        # Peek the opening frame: app.js sends session.update first. Resolve it
        # into the engine session config; any other leading frame is replayed
        # into the stream unchanged.
        session_config: dict[str, object] = {}
        # OpenAI realtime declares the input encoding once on session.update; the
        # OpenBMB demo/bridge send pcm16 appends with no per-frame format, so this
        # session-level value drives the pcm16->f32le conversion in the inbound
        # adapter. Missing it makes pcm16 decode as float32 -> NaN mel -> dead engine.
        input_audio_format = "pcm_f32le"
        replay: dict | None = None
        try:
            first = await websocket.receive_json()  # type: ignore[attr-defined]
        except WebSocketDisconnect:
            return
        if isinstance(first, dict) and (first.get("type") or first.get("event")) == "session.update":
            session = first.get("session") if isinstance(first.get("session"), dict) else {}
            fmt = session.get("input_audio_format")
            if isinstance(fmt, str) and fmt:
                input_audio_format = fmt
            session_config = await self._prepare_session_config(session)
        elif isinstance(first, dict):
            replay = first

        async def recv_frames() -> AsyncIterator[dict]:
            if replay is not None:
                yield replay
            while True:
                try:
                    frame = await websocket.receive_json()  # type: ignore[attr-defined]
                except WebSocketDisconnect:
                    return
                if isinstance(frame, dict):
                    yield frame

        async def send_json(frame: dict[str, object]) -> None:
            await websocket.send_json(frame)  # type: ignore[attr-defined]

        await self.run(
            recv_frames(),
            send_json,
            session_config=session_config,
            input_audio_format=input_audio_format,
        )


__all__ = ["RuntimeDuplexHandler", "SendJson"]
