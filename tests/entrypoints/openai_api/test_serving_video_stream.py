"""Unit tests for streaming video input WebSocket handler.

Uses Starlette's TestClient with a mocked chat service — runs on CPU
without model weights or CUDA.

The handler module is loaded directly via importlib to bypass
vllm_omni.__init__ which triggers CUDA imports.  All vllm dependencies
are mocked in sys.modules before the module is executed.
"""

import base64
import importlib.util
import io
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI, WebSocket
from PIL import Image
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Mock vllm modules to avoid CUDA dependency, then load the handler module
# directly (bypassing vllm_omni.__init__ which also triggers CUDA).
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Stand-in for ChatCompletionRequest — just stores kwargs as attrs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


_MOCK_MODULES = {
    "vllm": MagicMock(),
    "vllm.logger": MagicMock(init_logger=MagicMock(return_value=MagicMock())),
    "vllm.entrypoints": MagicMock(),
    "vllm.entrypoints.chat_utils": MagicMock(ChatCompletionMessageParam=dict),
    "vllm.entrypoints.openai": MagicMock(),
    "vllm.entrypoints.openai.chat_completion": MagicMock(),
    "vllm.entrypoints.openai.chat_completion.protocol": MagicMock(ChatCompletionRequest=_FakeRequest),
}

sys.modules.update(_MOCK_MODULES)

_handler_path = Path(__file__).resolve().parents[3] / "vllm_omni" / "entrypoints" / "openai" / "serving_video_stream.py"
_spec = importlib.util.spec_from_file_location("serving_video_stream", str(_handler_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

OmniStreamingVideoHandler = _mod.OmniStreamingVideoHandler
_MAX_BUFFER_FRAMES = _mod._MAX_BUFFER_FRAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame_b64(width: int = 64, height: int = 48, color: tuple = (255, 0, 0)) -> str:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _text_chunks(*texts: str) -> list[str]:
    chunks = [f'data: {{"choices":[{{"delta":{{"content":"{t}"}}}}]}}\n\n' for t in texts]
    chunks.append("data: [DONE]\n\n")
    return chunks


def _audio_chunks(text: str, audio_b64: str) -> list[str]:
    return [
        f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}\n\n',
        f'data: {{"choices":[{{"delta":{{"content":"{audio_b64}"}},"finish_reason":"stop"}}]}}\n\n',
        "data: [DONE]\n\n",
    ]


def _make_service(chunks: list[str]):
    svc = MagicMock()

    async def _gen(*_a, **_kw):
        for c in chunks:
            yield c

    svc.create_chat_completion = AsyncMock(side_effect=_gen)
    return svc


def _app(chat_service=None, **kw):
    if chat_service is None:
        chat_service = _make_service(_text_chunks("Hello", " world"))
    handler = OmniStreamingVideoHandler(chat_service=chat_service, **kw)
    app = FastAPI()

    @app.websocket("/ws")
    async def ep(websocket: WebSocket):
        await handler.handle_session(websocket)

    return app, chat_service


def _drain(ws, stop_type: str) -> list[dict]:
    msgs = []
    while True:
        m = ws.receive_json()
        msgs.append(m)
        if m["type"] == stop_type:
            break
    return msgs


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_config_then_done(self):
        a, _ = _app()
        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.done"})
            assert ws.receive_json()["type"] == "session.done"

    def test_missing_config(self):
        a, _ = _app()
        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "video.frame", "data": "abc"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "session.config" in msg["message"]

    def test_config_timeout(self):
        a, _ = _app(config_timeout=0.1)
        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Timeout" in msg["message"]

    def test_unknown_type_non_fatal(self):
        a, _ = _app()
        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "bogus"})
            assert ws.receive_json()["type"] == "error"
            ws.send_json({"type": "video.done"})
            assert ws.receive_json()["type"] == "session.done"

    def test_invalid_config_rejected(self):
        a, _ = _app()
        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "num_frames": 999})
            assert ws.receive_json()["type"] == "error"


# ---------------------------------------------------------------------------
# Frame buffering
# ---------------------------------------------------------------------------


class TestFrameBuffering:
    def test_valid_frame(self):
        a, _ = _app()
        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.frame", "data": _make_frame_b64()})
            ws.send_json({"type": "video.done"})
            assert ws.receive_json()["type"] == "session.done"

    def test_empty_frame_rejected(self):
        a, _ = _app()
        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.frame", "data": ""})
            assert ws.receive_json()["type"] == "error"

    def test_invalid_image_rejected(self):
        a, _ = _app()
        garbage = base64.b64encode(b"not an image").decode()
        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.frame", "data": garbage})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Invalid" in msg["message"]

    def test_query_without_frames(self):
        a, _ = _app()
        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.query", "text": "What?"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "No frames" in msg["message"]


# ---------------------------------------------------------------------------
# Query + streaming response
# ---------------------------------------------------------------------------


class TestQuery:
    def test_text_streaming(self):
        svc = _make_service(_text_chunks("A person", " is walking."))
        a, _ = _app(chat_service=svc)
        frame = _make_frame_b64()

        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.frame", "data": frame})
            ws.send_json({"type": "video.query", "text": "Describe."})

            msgs = _drain(ws, "response.text.done")
            types = [m["type"] for m in msgs]
            assert "response.start" in types
            assert "response.text.delta" in types
            done = next(m for m in msgs if m["type"] == "response.text.done")
            assert done["text"] == "A person is walking."

            ws.send_json({"type": "video.done"})
            assert ws.receive_json()["type"] == "session.done"

    def test_multiple_queries(self):
        call_count = 0

        def _chunks():
            nonlocal call_count
            call_count += 1
            return _text_chunks(f"Response {call_count}")

        svc = MagicMock()

        async def _gen(*_a, **_kw):
            for ch in _chunks():
                yield ch

        svc.create_chat_completion = AsyncMock(side_effect=_gen)
        a, _ = _app(chat_service=svc)
        frame = _make_frame_b64()

        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.frame", "data": frame})

            ws.send_json({"type": "video.query", "text": "Q1"})
            _drain(ws, "response.text.done")

            ws.send_json({"type": "video.query", "text": "Q2"})
            _drain(ws, "response.text.done")

            ws.send_json({"type": "video.done"})
            ws.receive_json()

        assert call_count == 2

    def test_audio_modality(self):
        audio_bytes = b"\x00\x01\x02\x03"
        audio_b64 = base64.b64encode(audio_bytes).decode()
        svc = _make_service(_audio_chunks("Hello", audio_b64))
        a, _ = _app(chat_service=svc)
        frame = _make_frame_b64()

        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "model": "test", "modalities": ["text", "audio"]})
            ws.send_json({"type": "video.frame", "data": frame})
            ws.send_json({"type": "video.query", "text": "Speak."})

            msgs = _drain(ws, "response.text.done")
            types = [m["type"] for m in msgs]
            assert "response.text.delta" in types
            assert "response.audio.start" in types
            assert "response.audio.done" in types

            ws.send_json({"type": "video.done"})
            ws.receive_json()

    def test_generation_failure(self):
        svc = MagicMock()
        svc.create_chat_completion = AsyncMock(side_effect=RuntimeError("boom"))
        a, _ = _app(chat_service=svc)
        frame = _make_frame_b64()

        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.frame", "data": frame})
            ws.send_json({"type": "video.query", "text": "What?"})

            msgs = _drain(ws, "error")
            assert "boom" in msgs[-1]["message"]


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------


class TestFrameSampling:
    def test_num_frames_limits_images(self):
        svc = _make_service(_text_chunks("ok"))
        a, _ = _app(chat_service=svc)
        frame = _make_frame_b64()

        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "model": "test", "num_frames": 2})
            for _ in range(10):
                ws.send_json({"type": "video.frame", "data": frame})
            ws.send_json({"type": "video.query", "text": "What?"})
            _drain(ws, "response.text.done")
            ws.send_json({"type": "video.done"})
            ws.receive_json()

        req = svc.create_chat_completion.call_args[0][0]
        user_content = req.messages[-1]["content"]
        image_count = sum(1 for item in user_content if isinstance(item, dict) and item.get("type") == "image_url")
        assert image_count == 2

    def test_sliding_window(self):
        svc = _make_service(_text_chunks("ok"))
        a, _ = _app(chat_service=svc)

        with TestClient(a) as c, c.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.config", "model": "test", "num_frames": 128})
            for _ in range(_MAX_BUFFER_FRAMES + 5):
                ws.send_json({"type": "video.frame", "data": _make_frame_b64()})
            ws.send_json({"type": "video.query", "text": "count"})
            _drain(ws, "response.text.done")
            ws.send_json({"type": "video.done"})
            ws.receive_json()

        req = svc.create_chat_completion.call_args[0][0]
        user_content = req.messages[-1]["content"]
        image_count = sum(1 for item in user_content if isinstance(item, dict) and item.get("type") == "image_url")
        assert image_count == _MAX_BUFFER_FRAMES
