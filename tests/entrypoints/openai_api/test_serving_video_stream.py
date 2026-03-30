"""Unit tests for streaming video input WebSocket handler.

Uses Starlette's TestClient with a mocked chat service — runs on CPU
without model weights or CUDA.
"""

import base64
import io
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, WebSocket
from PIL import Image
from starlette.testclient import TestClient

# Mock vllm.logger before importing the handler to avoid CUDA dependency
_mock_logger = MagicMock()
with patch.dict(sys.modules, {"vllm.logger": MagicMock(init_logger=MagicMock(return_value=_mock_logger))}):
    from vllm_omni.entrypoints.openai.serving_video_stream import (  # noqa: E402
        OmniStreamingVideoHandler,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame_b64(width: int = 64, height: int = 48, color: tuple = (255, 0, 0)) -> str:
    """Create a small JPEG frame encoded as base64."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_text_chunks(*texts: str) -> list[str]:
    """Build SSE chunks that stream text deltas."""
    chunks = []
    for t in texts:
        chunks.append(f'data: {{"choices":[{{"delta":{{"content":"{t}"}}}}]}}\n\n')
    chunks.append("data: [DONE]\n\n")
    return chunks


def _make_audio_chunks(text: str, audio_b64: str) -> list[str]:
    """Build SSE chunks that stream text then audio (finish_reason=stop)."""
    return [
        f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}\n\n',
        f'data: {{"choices":[{{"delta":{{"content":"{audio_b64}"}},"finish_reason":"stop"}}]}}\n\n',
        "data: [DONE]\n\n",
    ]


def _build_app(chat_service=None, **handler_kwargs):
    """Build a FastAPI app wired to a mocked chat service."""
    if chat_service is None:
        chat_service = _make_chat_service(_make_text_chunks("Hello", " world"))

    handler = OmniStreamingVideoHandler(chat_service=chat_service, **handler_kwargs)
    app = FastAPI()

    @app.websocket("/v1/video/chat/stream")
    async def ws_endpoint(websocket: WebSocket):
        await handler.handle_session(websocket)

    return app, chat_service


def _make_chat_service(chunks: list[str]):
    """Create a mock chat service that yields the given SSE chunks.

    Uses side_effect so each call gets a fresh generator.
    """
    svc = MagicMock()

    async def _gen(*_args, **_kwargs):
        for c in chunks:
            yield c

    svc.create_chat_completion = AsyncMock(side_effect=_gen)
    return svc


def _collect_until(ws, stop_type: str) -> list[dict]:
    """Receive JSON messages until one matches stop_type."""
    msgs = []
    while True:
        m = ws.receive_json()
        msgs.append(m)
        if m["type"] == stop_type:
            break
    return msgs


# ---------------------------------------------------------------------------
# Protocol basics
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_config_then_done(self):
        app, _ = _build_app()
        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.done"})
            assert ws.receive_json()["type"] == "session.done"

    def test_missing_config(self):
        app, _ = _build_app()
        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "video.frame", "data": "abc"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "session.config" in msg["message"]

    def test_config_timeout(self):
        app, _ = _build_app(config_timeout=0.1)
        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Timeout" in msg["message"]

    def test_unknown_type_non_fatal(self):
        app, _ = _build_app()
        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "bogus"})
            assert ws.receive_json()["type"] == "error"
            # session still alive
            ws.send_json({"type": "video.done"})
            assert ws.receive_json()["type"] == "session.done"

    def test_invalid_config_rejected(self):
        app, _ = _build_app()
        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "session.config", "num_frames": 999})
            msg = ws.receive_json()
            assert msg["type"] == "error"


# ---------------------------------------------------------------------------
# Frame buffering
# ---------------------------------------------------------------------------


class TestFrameBuffering:
    def test_valid_frame_accepted(self):
        app, _ = _build_app()
        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.frame", "data": _make_frame_b64()})
            ws.send_json({"type": "video.done"})
            assert ws.receive_json()["type"] == "session.done"

    def test_empty_frame_rejected(self):
        app, _ = _build_app()
        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.frame", "data": ""})
            assert ws.receive_json()["type"] == "error"

    def test_invalid_image_rejected(self):
        app, _ = _build_app()
        garbage = base64.b64encode(b"not an image").decode()
        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.frame", "data": garbage})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Invalid" in msg["message"]

    def test_query_without_frames(self):
        app, _ = _build_app()
        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
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
        svc = _make_chat_service(_make_text_chunks("A person", " is walking."))
        app, _ = _build_app(chat_service=svc)
        frame = _make_frame_b64()

        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.frame", "data": frame})
            ws.send_json({"type": "video.query", "text": "Describe."})

            msgs = _collect_until(ws, "response.text.done")
            types = [m["type"] for m in msgs]
            assert "response.start" in types
            assert "response.text.delta" in types
            assert "response.text.done" in types

            done = next(m for m in msgs if m["type"] == "response.text.done")
            assert done["text"] == "A person is walking."

            ws.send_json({"type": "video.done"})
            assert ws.receive_json()["type"] == "session.done"

    def test_multiple_queries(self):
        """Second query in same session reuses buffered frames."""
        call_count = 0

        def _chunks():
            nonlocal call_count
            call_count += 1
            return _make_text_chunks(f"Response {call_count}")

        svc = MagicMock()

        async def _gen(*_a, **_kw):
            for c in _chunks():
                yield c

        svc.create_chat_completion = AsyncMock(side_effect=_gen)
        app, _ = _build_app(chat_service=svc)
        frame = _make_frame_b64()

        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.frame", "data": frame})

            ws.send_json({"type": "video.query", "text": "Q1"})
            _collect_until(ws, "response.text.done")

            ws.send_json({"type": "video.query", "text": "Q2"})
            _collect_until(ws, "response.text.done")

            ws.send_json({"type": "video.done"})
            ws.receive_json()

        assert call_count == 2

    def test_audio_modality(self):
        """When modalities includes audio, the final chunk is routed as audio."""
        audio_bytes = b"\x00\x01\x02\x03"
        audio_b64 = base64.b64encode(audio_bytes).decode()
        svc = _make_chat_service(_make_audio_chunks("Hello", audio_b64))
        app, _ = _build_app(chat_service=svc)
        frame = _make_frame_b64()

        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "session.config", "model": "test", "modalities": ["text", "audio"]})
            ws.send_json({"type": "video.frame", "data": frame})
            ws.send_json({"type": "video.query", "text": "Speak."})

            # Collect all messages until text.done
            msgs = _collect_until(ws, "response.text.done")
            types = [m["type"] for m in msgs]

            # Should have text delta, audio events, then text.done
            assert "response.text.delta" in types
            assert "response.audio.start" in types
            assert "response.audio.done" in types

            ws.send_json({"type": "video.done"})
            ws.receive_json()

    def test_generation_failure(self):
        """Chat service exception is reported as error."""
        svc = MagicMock()
        svc.create_chat_completion = AsyncMock(side_effect=RuntimeError("boom"))
        app, _ = _build_app(chat_service=svc)
        frame = _make_frame_b64()

        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "session.config", "model": "test"})
            ws.send_json({"type": "video.frame", "data": frame})
            ws.send_json({"type": "video.query", "text": "What?"})

            msgs = []
            while True:
                m = ws.receive_json()
                msgs.append(m)
                if m["type"] == "error":
                    break
            assert "boom" in msgs[-1]["message"]


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------


class TestFrameSampling:
    def test_num_frames_limits_images(self):
        """When buffer has more frames than num_frames, sample uniformly."""
        svc = _make_chat_service(_make_text_chunks("ok"))
        app, _ = _build_app(chat_service=svc)
        frame = _make_frame_b64()

        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "session.config", "model": "test", "num_frames": 2})
            for _ in range(10):
                ws.send_json({"type": "video.frame", "data": frame})
            ws.send_json({"type": "video.query", "text": "What?"})
            _collect_until(ws, "response.text.done")
            ws.send_json({"type": "video.done"})
            ws.receive_json()

        # Inspect the request passed to chat service
        req = svc.create_chat_completion.call_args[0][0]
        user_content = req.messages[-1]["content"]
        image_count = sum(1 for item in user_content if isinstance(item, dict) and item.get("type") == "image_url")
        assert image_count == 2

    def test_sliding_window_drops_oldest(self):
        """Buffer caps at _MAX_BUFFER_FRAMES and drops oldest."""
        from vllm_omni.entrypoints.openai.serving_video_stream import _MAX_BUFFER_FRAMES

        svc = _make_chat_service(_make_text_chunks("ok"))
        app, _ = _build_app(chat_service=svc)

        with TestClient(app) as c, c.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json({"type": "session.config", "model": "test", "num_frames": 128})
            # Send more frames than the buffer limit
            for i in range(_MAX_BUFFER_FRAMES + 5):
                ws.send_json({"type": "video.frame", "data": _make_frame_b64()})
            ws.send_json({"type": "video.query", "text": "count"})
            _collect_until(ws, "response.text.done")
            ws.send_json({"type": "video.done"})
            ws.receive_json()

        req = svc.create_chat_completion.call_args[0][0]
        user_content = req.messages[-1]["content"]
        image_count = sum(1 for item in user_content if isinstance(item, dict) and item.get("type") == "image_url")
        assert image_count == _MAX_BUFFER_FRAMES
