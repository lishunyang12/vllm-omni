# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end drive of DuplexRuntime with the continuous full-duplex reducer."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from vllm_omni.experimental.fullduplex.core.continuous import (
    ContinuousDuplexState,
    reduce_continuous_event,
)
from vllm_omni.experimental.fullduplex.core.events import (
    AppendToEngine,
    DomainEvent,
    EmitProtocolEvent,
    InputChunk,
    ModelAudioDelta,
    ModelSpeaking,
    ModelTurnEnded,
    ProtocolEventKind,
    RebuildStage0Context,
    ReserveResponse,
    ResetStage1,
    SessionCloseRequested,
)
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Sink:
    def __init__(self) -> None:
        self.events: list[EmitProtocolEvent] = []
        self._seen: dict[ProtocolEventKind, asyncio.Event] = {}

    async def emit(self, event: EmitProtocolEvent) -> None:
        self.events.append(event)
        self._seen.setdefault(event.kind, asyncio.Event()).set()

    async def wait_for(self, kind: ProtocolEventKind) -> None:
        await asyncio.wait_for(self._seen.setdefault(kind, asyncio.Event()).wait(), timeout=1)

    def kinds(self) -> list[ProtocolEventKind]:
        return [e.kind for e in self.events]


class _Engine:
    def __init__(self) -> None:
        self.commands: list[object] = []
        self._events: asyncio.Queue[DomainEvent | None] = asyncio.Queue()

    async def reserve(self, command: ReserveResponse) -> None:
        self.commands.append(command)

    async def append(self, command: AppendToEngine) -> None:
        self.commands.append(command)

    async def cancel(self, fence: DuplexFence) -> None:
        self.commands.append(("cancel", fence))

    async def reset(self, command: ResetStage1) -> None:
        self.commands.append(command)

    async def rebuild(self, command: RebuildStage0Context) -> None:
        self.commands.append(command)

    async def close(self, fence: DuplexFence) -> None:
        self.commands.append(("close", fence))
        await self._events.put(None)

    async def emit(self, event: DomainEvent) -> None:
        await self._events.put(event)

    async def events(self) -> AsyncIterator[DomainEvent]:
        while (event := await self._events.get()) is not None:
            yield event


def _make_runtime(engine: _Engine, sink: _Sink) -> DuplexRuntime:
    session_id = "sess-demo"
    return DuplexRuntime(
        session_id,
        engine,
        sink,
        reduce=reduce_continuous_event,
        initial_state=ContinuousDuplexState.open(session_id),
    )


@pytest.mark.asyncio
async def test_continuous_runtime_streams_a_model_owned_response_with_input_overlap():
    engine = _Engine()
    sink = _Sink()
    runtime = _make_runtime(engine, sink)
    ef = DuplexFence("sess-demo")  # epoch fence

    async def session() -> AsyncIterator[DomainEvent]:
        # Stream input; no client commit — the model owns the turn.
        yield InputChunk(data=b"\x00\x00\x00\x00", modality="audio")
        await engine.emit(ModelSpeaking(fence=ef))
        await sink.wait_for(ProtocolEventKind.RESPONSE_STARTED)
        # Full-duplex overlap: more input arrives WHILE the model is speaking.
        yield InputChunk(data=b"\x11\x11\x11\x11", modality="audio")
        await engine.emit(ModelAudioDelta(data=b"\x00" * 8, generated_cursor=0, sent_cursor=0, fence=ef))
        await sink.wait_for(ProtocolEventKind.AUDIO_DELTA)
        await engine.emit(ModelTurnEnded(fence=ef))
        await sink.wait_for(ProtocolEventKind.RESPONSE_COMPLETED)
        yield SessionCloseRequested()

    await asyncio.wait_for(runtime.run(session()), timeout=3)

    kinds = sink.kinds()
    assert ProtocolEventKind.RESPONSE_STARTED in kinds
    assert ProtocolEventKind.AUDIO_DELTA in kinds
    assert ProtocolEventKind.RESPONSE_COMPLETED in kinds

    # Both input chunks reached the engine — including the one sent mid-response.
    appends = [c for c in engine.commands if isinstance(c, AppendToEngine) and c.chunk is not None]
    assert len(appends) == 2, "input during model speech must still be forwarded (full duplex)"
