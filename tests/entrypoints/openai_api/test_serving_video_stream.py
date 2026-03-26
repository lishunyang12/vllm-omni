"""Unit tests for streaming video input WebSocket handler.

These tests use Starlette's TestClient with a mocked chat service,
so they run on CPU without model weights. They validate the full
WebSocket protocol: session config, frame buffering, query submission,
response streaming, error handling, and session teardown.
"""

import base64
import io
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, WebSocket
from PIL import Image
from starlette.testclient import TestClient

from vllm_omni.entrypoints.openai.serving_video_stream import (
    OmniStreamingVideoHandler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_frame_b64(width: int = 64, height: int = 48,
                    color: tuple = (255, 0, 0)) -> str:
    """Create a small JPEG frame encoded as base64."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_test_app(chat_service=None, **handler_kwargs):
    """Build a FastAPI app with a mocked chat service for testing."""
    if chat_service is None:
        chat_service = MagicMock()

        # Mock create_chat_completion to return a streaming generator
        async def mock_streaming_response(*args, **kwargs):
            chunks = [
                'data: {"choices":[{"delta":{"content":"A person"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":" is walking."}}]}\n\n',
                'data: [DONE]\n\n',
            ]
            for c in chunks:
                yield c

        chat_service.create_chat_completion = AsyncMock(
            return_value=mock_streaming_response()
        )

    handler = OmniStreamingVideoHandler(
        chat_service=chat_service, **handler_kwargs
    )
    app = FastAPI()

    @app.websocket("/v1/video/chat/stream")
    async def ws_endpoint(websocket: WebSocket):
        await handler.handle_session(websocket)

    return app, chat_service


class TestVideoStreamProtocol:
    """Test the WebSocket protocol handshake and message routing."""

    def test_session_config_then_done(self):
        """Minimal session: config -> done."""
        app, _ = _build_test_app()
        with TestClient(app) as client:
            with client.websocket_connect("/v1/video/chat/stream") as ws:
                ws.send_json({"type": "session.config", "model": "test"})
                ws.send_json({"type": "video.done"})
                msg = ws.receive_json()
                assert msg["type"] == "session.done"

    def test_missing_config_type(self):
        """First message must be session.config."""
        app, _ = _build_test_app()
        with TestClient(app) as client:
            with client.websocket_connect("/v1/video/chat/stream") as ws:
                ws.send_json({"type": "video.frame", "data": "abc"})
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "session.config" in msg["message"]

    def test_config_timeout(self):
        """Handler returns error if config not sent within timeout."""
        app, _ = _build_test_app(config_timeout=0.1)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/video/chat/stream") as ws:
                # Don't send config — wait for timeout
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "Timeout" in msg["message"]

    def test_unknown_message_type(self):
        """Unknown message types produce an error but don't close."""
        app, _ = _build_test_app()
        with TestClient(app) as client:
            with client.websocket_connect("/v1/video/chat/stream") as ws:
                ws.send_json({"type": "session.config", "model": "test"})
                ws.send_json({"type": "bogus.event"})
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "Unknown" in msg["message"]

                # Session still alive — can send done
                ws.send_json({"type": "video.done"})
                msg = ws.receive_json()
                assert msg["type"] == "session.done"


class TestFrameBuffering:
    """Test video frame buffering and validation."""

    def test_buffer_valid_frame(self):
        """Valid JPEG frames are accepted without error."""
        app, _ = _build_test_app()
        frame = _make_frame_b64()
        with TestClient(app) as client:
            with client.websocket_connect("/v1/video/chat/stream") as ws:
                ws.send_json({"type": "session.config", "model": "test"})
                ws.send_json({"type": "video.frame", "data": frame})
                # No error — send done
                ws.send_json({"type": "video.done"})
                msg = ws.receive_json()
                assert msg["type"] == "session.done"

    def test_empty_frame_rejected(self):
        """Empty frame data produces an error."""
        app, _ = _build_test_app()
        with TestClient(app) as client:
            with client.websocket_connect("/v1/video/chat/stream") as ws:
                ws.send_json({"type": "session.config", "model": "test"})
                ws.send_json({"type": "video.frame", "data": ""})
                msg = ws.receive_json()
                assert msg["type"] == "error"

    def test_invalid_image_rejected(self):
        """Non-image base64 data produces an error."""
        app, _ = _build_test_app()
        garbage = base64.b64encode(b"not an image").decode()
        with TestClient(app) as client:
            with client.websocket_connect("/v1/video/chat/stream") as ws:
                ws.send_json({"type": "session.config", "model": "test"})
                ws.send_json({"type": "video.frame", "data": garbage})
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "Invalid" in msg["message"]

    def test_query_without_frames_rejected(self):
        """Query before any frames produces an error."""
        app, _ = _build_test_app()
        with TestClient(app) as client:
            with client.websocket_connect("/v1/video/chat/stream") as ws:
                ws.send_json({"type": "session.config", "model": "test"})
                ws.send_json({"type": "video.query", "text": "What?"})
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "No frames" in msg["message"]


class TestQueryAndResponse:
    """Test frame buffering -> query -> streaming response."""

    def test_single_query_streaming_response(self):
        """Frames + query -> streaming text deltas + done."""
        app, chat_service = _build_test_app()
        frame = _make_frame_b64()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/video/chat/stream") as ws:
                ws.send_json({"type": "session.config", "model": "test"})

                # Send 3 frames
                for _ in range(3):
                    ws.send_json({"type": "video.frame", "data": frame})

                # Submit query
                ws.send_json({
                    "type": "video.query",
                    "text": "What do you see?",
                })

                # Collect responses
                messages = []
                while True:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg["type"] in (
                        "response.text.done",
                        "error",
                    ):
                        break

                types = [m["type"] for m in messages]
                assert "response.start" in types
                assert "response.text.delta" in types
                assert "response.text.done" in types

                # Final text should contain the streamed content
                done_msg = next(
                    m for m in messages if m["type"] == "response.text.done"
                )
                assert "person" in done_msg["text"].lower() or len(done_msg["text"]) > 0

                # Close session
                ws.send_json({"type": "video.done"})
                msg = ws.receive_json()
                assert msg["type"] == "session.done"

        # Verify chat_service was called
        chat_service.create_chat_completion.assert_awaited_once()

    def test_multiple_queries_same_session(self):
        """Multiple queries reuse buffered frames."""
        call_count = 0

        async def mock_gen(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            chunks = [
                f'data: {{"choices":[{{"delta":{{"content":"Response {call_count}"}}}}]}}\n\n',
                'data: [DONE]\n\n',
            ]
            for c in chunks:
                yield c

        chat_service = MagicMock()
        chat_service.create_chat_completion = AsyncMock(side_effect=mock_gen)

        app, _ = _build_test_app(chat_service=chat_service)
        frame = _make_frame_b64()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/video/chat/stream") as ws:
                ws.send_json({"type": "session.config", "model": "test"})
                ws.send_json({"type": "video.frame", "data": frame})

                # First query
                ws.send_json({"type": "video.query", "text": "Q1"})
                msgs = []
                while True:
                    m = ws.receive_json()
                    msgs.append(m)
                    if m["type"] == "response.text.done":
                        break

                # Second query (frames still buffered)
                ws.send_json({"type": "video.query", "text": "Q2"})
                msgs2 = []
                while True:
                    m = ws.receive_json()
                    msgs2.append(m)
                    if m["type"] == "response.text.done":
                        break

                ws.send_json({"type": "video.done"})
                ws.receive_json()  # session.done

        assert call_count == 2


class TestSessionConfig:
    """Test session configuration validation."""

    def test_custom_num_frames(self):
        """num_frames parameter is respected."""
        app, chat_service = _build_test_app()
        frame = _make_frame_b64()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/video/chat/stream") as ws:
                ws.send_json({
                    "type": "session.config",
                    "model": "test",
                    "num_frames": 2,
                })

                # Send 10 frames
                for _ in range(10):
                    ws.send_json({"type": "video.frame", "data": frame})

                ws.send_json({"type": "video.query", "text": "What?"})

                # Drain response
                while True:
                    m = ws.receive_json()
                    if m["type"] == "response.text.done":
                        break

                ws.send_json({"type": "video.done"})
                ws.receive_json()

        # Verify the request was built with sampled frames (not all 10)
        call_args = chat_service.create_chat_completion.call_args
        request = call_args[0][0]
        # Count image_url items in the first user message content
        user_msg = request.messages[-1]
        image_count = sum(
            1 for item in user_msg["content"]
            if isinstance(item, dict) and item.get("type") == "image_url"
        )
        assert image_count == 2  # num_frames=2

    def test_invalid_config_rejected(self):
        """Invalid config values produce an error."""
        app, _ = _build_test_app()
        with TestClient(app) as client:
            with client.websocket_connect("/v1/video/chat/stream") as ws:
                ws.send_json({
                    "type": "session.config",
                    "num_frames": 999,  # exceeds max 128
                })
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "Invalid" in msg["message"]
