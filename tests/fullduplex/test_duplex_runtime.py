# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the model-agnostic duplex runtime + the JoyVL adapter."""

from collections.abc import AsyncIterator

import pytest

from vllm_omni.fullduplex import protocol as ev
from vllm_omni.fullduplex.adapter import DuplexCapability, OutputChunk
from vllm_omni.fullduplex.adapters.joyvl import JoyVLDuplexAdapter
from vllm_omni.fullduplex.runtime import DuplexRuntime
from vllm_omni.fullduplex.session import DuplexSession, DuplexSessionConfig


async def _feed(events):
    for e in events:
        yield e


def _collector():
    out: list[dict] = []

    async def emit(event: dict) -> None:
        out.append(event)

    return out, emit


class _FakeAdapter:
    def __init__(self, chunks, barge_after=None):
        self._chunks = chunks
        self._barge_after = barge_after  # index after which to self-interrupt

    def capabilities(self):
        return DuplexCapability(frozenset({"text"}), frozenset({"text"}), proactive=False)

    async def on_input(self, session, modality, data):
        pass

    def should_respond(self, session):
        return True

    async def respond(self, session) -> AsyncIterator[OutputChunk]:
        for i, c in enumerate(self._chunks):
            if self._barge_after is not None and i == self._barge_after:
                session.barge_in()  # simulate an interruption arriving mid-response
            yield OutputChunk("text", c)

    async def on_barge_in(self, session):
        pass

    async def on_playback_ack(self, session, cursor):
        pass


@pytest.mark.asyncio
async def test_runtime_basic_response():
    session = DuplexSession("s", DuplexSessionConfig(output_modalities=("text",)))
    rt = DuplexRuntime(session, _FakeAdapter(["a", "b"]))
    out, emit = _collector()
    await rt.run(_feed([{"type": ev.INPUT_COMMIT}, {"type": ev.CLOSE}]), emit)
    types = [e["type"] for e in out]
    assert types == [ev.RESPONSE_CREATED, ev.RESPONSE_DELTA, ev.RESPONSE_DELTA, ev.RESPONSE_DONE]
    assert [e["data"] for e in out if e["type"] == ev.RESPONSE_DELTA] == ["a", "b"]


@pytest.mark.asyncio
async def test_runtime_barge_in_drops_stale_output():
    session = DuplexSession("s")
    rt = DuplexRuntime(session, _FakeAdapter(["a", "b", "c"], barge_after=1))
    out, emit = _collector()
    await rt.run(_feed([{"type": ev.INPUT_COMMIT}, {"type": ev.CLOSE}]), emit)
    data = [e["data"] for e in out if e["type"] == ev.RESPONSE_DELTA]
    assert data == ["a"]  # "b"/"c" produced under a stale epoch are dropped
    assert ev.RESPONSE_DONE not in [e["type"] for e in out]  # interrupted -> no done


@pytest.mark.asyncio
async def test_joyvl_adapter_speaks_then_silences():
    replies = iter(["</response> a fire is breaking out", "</silence>"])

    async def fake_generate(messages):
        return next(replies)

    adapter = JoyVLDuplexAdapter(fake_generate, num_frames=4)
    cfg = DuplexSessionConfig(input_modalities=("video", "text"), output_modalities=("text",), proactive=True)
    rt = DuplexRuntime(DuplexSession("s", cfg), adapter)
    out, emit = _collector()
    await rt.run(
        _feed(
            [
                {"type": ev.INPUT_APPEND, "modality": "text", "data": "alert me if a fire breaks out"},
                {"type": ev.INPUT_APPEND, "modality": "video", "data": "data:image/jpeg;base64,AAA"},
                {"type": ev.INPUT_APPEND, "modality": "video", "data": "data:image/jpeg;base64,BBB"},
                {"type": ev.CLOSE},
            ]
        ),
        emit,
    )
    deltas = [e["data"] for e in out if e["type"] == ev.RESPONSE_DELTA]
    assert "a fire is breaking out" in deltas  # first tick spoke
    # second video tick -> model returned </silence> -> a response opened but no text delta
    assert deltas.count("a fire is breaking out") == 1
