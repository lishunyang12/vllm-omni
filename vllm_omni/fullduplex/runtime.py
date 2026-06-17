# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

from vllm_omni.fullduplex import protocol as ev
from vllm_omni.fullduplex.adapter import DuplexAdapter
from vllm_omni.fullduplex.session import DuplexSession, DuplexState

Emit = Callable[[dict], Awaitable[None]]


class DuplexRuntime:
    def __init__(self, session: DuplexSession, adapter: DuplexAdapter) -> None:
        self.session = session
        self.adapter = adapter
        self._capabilities = adapter.capabilities()

    async def run(self, inputs: AsyncIterator[dict], emit: Emit) -> None:
        async for event in inputs:
            etype = event.get("type")
            if etype == ev.INPUT_APPEND:
                modality = event.get("modality", "")
                if modality not in self._capabilities.input_modalities:
                    await emit(ev.error(f"unsupported input modality: {modality}"))
                    continue
                await self.adapter.on_input(self.session, modality, event.get("data"))
                self.session.state = DuplexState.LISTENING
                if self.session.config.proactive and self.adapter.should_respond(self.session):
                    await self._respond(emit)
            elif etype in (ev.INPUT_COMMIT, ev.RESPONSE_CREATE):
                if self.adapter.should_respond(self.session):
                    await self._respond(emit)
            elif etype == ev.RESPONSE_CANCEL:
                await self._barge_in()
                await emit(ev.cancelled(self.session.response_index))
            elif etype == ev.PLAYBACK_ACK:
                await self.adapter.on_playback_ack(self.session, int(event.get("cursor", 0)))
            elif etype == ev.CLOSE:
                break
        self.session.state = DuplexState.CLOSED

    async def _respond(self, emit: Emit) -> None:
        response_index, epoch = self.session.begin_response()
        await emit(ev.created(response_index))
        try:
            async for chunk in self.adapter.respond(self.session):
                if self.session.is_stale(epoch):
                    return
                await emit(ev.delta(response_index, chunk.modality, chunk.data))
        except Exception as err:
            await emit(ev.error(f"response failed: {err}"))
        finally:
            if not self.session.is_stale(epoch):
                await emit(ev.done(response_index))
                self.session.end_response()

    async def _barge_in(self) -> None:
        self.session.barge_in()
        await self.adapter.on_barge_in(self.session)
