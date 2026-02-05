"""Integration tests for the streaming speech WebSocket endpoint."""

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from starlette.testclient import TestClient

from vllm_omni.entrypoints.openai.protocol.audio import (
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_speech_stream import (
    OmniStreamingSpeechHandler,
)


def _create_mock_speech_service():
    """Create a mock OmniOpenAIServingSpeech for testing."""
    service = MagicMock(spec=OmniOpenAIServingSpeech)

    # Mock _generate_audio_bytes to return fake WAV data
    async def mock_generate(request):
        # Return minimal WAV-like bytes and media type
        fake_audio = b"RIFF" + b"\x00" * 100  # Fake WAV header + data
        return fake_audio, "audio/wav"

    service._generate_audio_bytes = AsyncMock(side_effect=mock_generate)
    return service


def _build_test_app(
    speech_service=None,
    idle_timeout=5.0,
    config_timeout=5.0,
):
    """Build a FastAPI app with the streaming speech WebSocket route."""
    if speech_service is None:
        speech_service = _create_mock_speech_service()

    handler = OmniStreamingSpeechHandler(
        speech_service=speech_service,
        idle_timeout=idle_timeout,
        config_timeout=config_timeout,
    )

    app = FastAPI()

    @app.websocket("/v1/audio/speech/stream")
    async def ws_endpoint(websocket):
        await handler.handle_session(websocket)

    return app, handler, speech_service


class TestStreamingSpeechBasicLifecycle:
    """Test basic session lifecycle: config -> text -> done."""

    def test_single_sentence(self):
        app, _, service = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                # Send config
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "response_format": "wav",
                    }
                )

                # Send text with sentence boundary
                ws.send_json(
                    {
                        "type": "input.text",
                        "text": "Hello world. ",
                    }
                )

                # Receive audio.start
                msg = ws.receive_json()
                assert msg["type"] == "audio.start"
                assert msg["sentence_index"] == 0
                assert msg["sentence_text"] == "Hello world."
                assert msg["format"] == "wav"

                # Receive binary audio
                audio_data = ws.receive_bytes()
                assert len(audio_data) > 0

                # Receive audio.done
                msg = ws.receive_json()
                assert msg["type"] == "audio.done"
                assert msg["sentence_index"] == 0

                # Send done
                ws.send_json({"type": "input.done"})

                # Receive session.done
                msg = ws.receive_json()
                assert msg["type"] == "session.done"
                assert msg["total_sentences"] == 1

    def test_multiple_sentences(self):
        app, _, service = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                    }
                )

                ws.send_json(
                    {
                        "type": "input.text",
                        "text": "Hello world. How are you? ",
                    }
                )

                # First sentence
                msg = ws.receive_json()
                assert msg["type"] == "audio.start"
                assert msg["sentence_index"] == 0
                ws.receive_bytes()  # audio
                msg = ws.receive_json()
                assert msg["type"] == "audio.done"

                # Second sentence
                msg = ws.receive_json()
                assert msg["type"] == "audio.start"
                assert msg["sentence_index"] == 1
                ws.receive_bytes()  # audio
                msg = ws.receive_json()
                assert msg["type"] == "audio.done"

                ws.send_json({"type": "input.done"})

                msg = ws.receive_json()
                assert msg["type"] == "session.done"
                assert msg["total_sentences"] == 2

    def test_incremental_text(self):
        """Text arrives word-by-word, sentence formed across chunks."""
        app, _, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                    }
                )

                # Send text incrementally
                ws.send_json({"type": "input.text", "text": "Hello "})
                ws.send_json({"type": "input.text", "text": "world. "})

                # Now a sentence boundary was hit
                msg = ws.receive_json()
                assert msg["type"] == "audio.start"
                assert msg["sentence_text"] == "Hello world."
                ws.receive_bytes()
                msg = ws.receive_json()
                assert msg["type"] == "audio.done"

                # Send done
                ws.send_json({"type": "input.done"})

                msg = ws.receive_json()
                assert msg["type"] == "session.done"
                assert msg["total_sentences"] == 1


class TestStreamingSpeechFlush:
    """Test flush behavior when text has no sentence boundary."""

    def test_flush_on_done(self):
        """Text without sentence boundary is flushed on input.done."""
        app, _, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                    }
                )

                ws.send_json(
                    {
                        "type": "input.text",
                        "text": "Hello world without punctuation",
                    }
                )

                # No sentence boundary, so nothing generated yet
                # Now send done to flush
                ws.send_json({"type": "input.done"})

                # Should get audio for the flushed text
                msg = ws.receive_json()
                assert msg["type"] == "audio.start"
                assert "Hello world without punctuation" in msg["sentence_text"]
                ws.receive_bytes()
                msg = ws.receive_json()
                assert msg["type"] == "audio.done"

                msg = ws.receive_json()
                assert msg["type"] == "session.done"
                assert msg["total_sentences"] == 1

    def test_empty_flush(self):
        """input.done with empty buffer produces no audio."""
        app, _, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                    }
                )

                # Send nothing, just done
                ws.send_json({"type": "input.done"})

                msg = ws.receive_json()
                assert msg["type"] == "session.done"
                assert msg["total_sentences"] == 0


class TestStreamingSpeechErrors:
    """Test error handling scenarios."""

    def test_missing_config(self):
        """Sending text before config should produce an error."""
        app, _, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                # Send text instead of config
                ws.send_json(
                    {
                        "type": "input.text",
                        "text": "Hello",
                    }
                )

                # Should get error about expecting session.config
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "session.config" in msg["message"]

    def test_invalid_json(self):
        """Invalid JSON should produce an error but not kill session."""
        app, _, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                    }
                )

                # Send invalid JSON
                ws.send_text("not json at all")

                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "Invalid JSON" in msg["message"]

                # Session should still be alive — send done
                ws.send_json({"type": "input.done"})

                msg = ws.receive_json()
                assert msg["type"] == "session.done"

    def test_unknown_message_type(self):
        """Unknown message type produces error but doesn't kill session."""
        app, _, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                    }
                )

                ws.send_json({"type": "unknown.type"})

                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "Unknown message type" in msg["message"]

                ws.send_json({"type": "input.done"})
                msg = ws.receive_json()
                assert msg["type"] == "session.done"

    def test_generation_failure_continues_session(self):
        """If generation fails for one sentence, session continues."""
        service = _create_mock_speech_service()

        call_count = 0

        async def mock_generate_with_failure(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Generation failed")
            return b"RIFF" + b"\x00" * 100, "audio/wav"

        service._generate_audio_bytes = AsyncMock(side_effect=mock_generate_with_failure)

        app, _, _ = _build_test_app(speech_service=service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                    }
                )

                # First sentence will fail
                ws.send_json(
                    {
                        "type": "input.text",
                        "text": "First sentence. Second sentence. ",
                    }
                )

                # First sentence: audio.start -> error -> audio.done
                msg = ws.receive_json()
                assert msg["type"] == "audio.start"
                assert msg["sentence_index"] == 0

                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "Generation failed" in msg["message"]

                msg = ws.receive_json()
                assert msg["type"] == "audio.done"
                assert msg["sentence_index"] == 0

                # Second sentence should succeed
                msg = ws.receive_json()
                assert msg["type"] == "audio.start"
                assert msg["sentence_index"] == 1
                ws.receive_bytes()  # audio data
                msg = ws.receive_json()
                assert msg["type"] == "audio.done"
                assert msg["sentence_index"] == 1

                ws.send_json({"type": "input.done"})
                msg = ws.receive_json()
                assert msg["type"] == "session.done"


class TestStreamingSpeechSessionConfig:
    """Test session config validation."""

    def test_valid_config_all_fields(self):
        app, _, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "task_type": "CustomVoice",
                        "language": "English",
                        "instructions": "Speak cheerfully",
                        "response_format": "mp3",
                        "speed": 1.5,
                        "max_new_tokens": 1024,
                    }
                )

                ws.send_json({"type": "input.done"})
                msg = ws.receive_json()
                assert msg["type"] == "session.done"

    def test_invalid_speed(self):
        app, _, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "speed": 10.0,  # invalid: > 4.0
                    }
                )

                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "Invalid session config" in msg["message"]

    def test_invalid_response_format(self):
        app, _, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "response_format": "invalid",
                    }
                )

                msg = ws.receive_json()
                assert msg["type"] == "error"


class TestStreamingSpeechConfigModel:
    """Unit tests for StreamingSpeechSessionConfig model."""

    def test_defaults(self):
        config = StreamingSpeechSessionConfig()
        assert config.response_format == "wav"
        assert config.speed == 1.0
        assert config.voice is None
        assert config.task_type is None

    def test_all_fields(self):
        config = StreamingSpeechSessionConfig(
            model="test",
            voice="Vivian",
            task_type="CustomVoice",
            language="English",
            instructions="test",
            response_format="mp3",
            speed=2.0,
            max_new_tokens=1024,
            ref_audio="http://example.com/audio.wav",
            ref_text="hello",
            x_vector_only_mode=True,
        )
        assert config.voice == "Vivian"
        assert config.speed == 2.0
        assert config.response_format == "mp3"
