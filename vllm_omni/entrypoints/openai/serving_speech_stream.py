"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers and splits at sentence
boundaries, and generates audio per sentence using the existing TTS pipeline.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of input

    Server -> Client:
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        {"type": "audio.done", "sentence_index": 0}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}
"""

import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.text_splitter import SentenceSplitter

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds


class OmniStreamingSpeechHandler:
    """Handles WebSocket sessions for streaming text-input TTS.

    Each WebSocket connection is an independent session. Text arrives
    incrementally, is split at sentence boundaries, and audio is generated
    per sentence using the existing OmniOpenAIServingSpeech pipeline.

    Args:
        speech_service: The existing TTS serving instance (reused for
            validation and audio generation).
        idle_timeout: Max seconds to wait for a message before closing.
        config_timeout: Max seconds to wait for the initial session.config.
    """

    def __init__(
        self,
        speech_service: OmniOpenAIServingSpeech,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    ) -> None:
        self._speech_service = speech_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            # 1. Wait for session.config
            config = await self._receive_config(websocket)
            if config is None:
                return  # Error already sent, connection closing

            splitter = SentenceSplitter()
            sentence_index = 0

            # 2. Receive text chunks until input.done
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self._idle_timeout,
                    )
                except asyncio.TimeoutError:
                    await self._send_error(websocket, "Idle timeout: no message received")
                    return

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message")
                    continue

                msg_type = msg.get("type")

                if msg_type == "input.text":
                    text = msg.get("text", "")
                    sentences = splitter.add_text(text)
                    for sentence in sentences:
                        await self._generate_and_send(
                            websocket, config, sentence, sentence_index
                        )
                        sentence_index += 1

                elif msg_type == "input.done":
                    # Flush remaining buffer
                    remaining = splitter.flush()
                    if remaining:
                        await self._generate_and_send(
                            websocket, config, remaining, sentence_index
                        )
                        sentence_index += 1

                    # Send session.done
                    await websocket.send_json({
                        "type": "session.done",
                        "total_sentences": sentence_index,
                    })
                    return

                else:
                    await self._send_error(
                        websocket,
                        f"Unknown message type: {msg_type}",
                    )

        except WebSocketDisconnect:
            logger.info("Streaming speech: client disconnected")
        except Exception as e:
            logger.exception("Streaming speech session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                pass

    async def _receive_config(
        self, websocket: WebSocket
    ) -> StreamingSpeechSessionConfig | None:
        """Wait for and validate the session.config message."""
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error(
                websocket, "Timeout waiting for session.config"
            )
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type')}",
            )
            return None

        try:
            config = StreamingSpeechSessionConfig(**{
                k: v for k, v in msg.items() if k != "type"
            })
        except ValidationError as e:
            await self._send_error(
                websocket, f"Invalid session config: {e}"
            )
            return None

        return config

    async def _generate_and_send(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_text: str,
        sentence_index: int,
    ) -> None:
        """Generate audio for a single sentence and send it over WebSocket."""
        response_format = config.response_format or "wav"

        # Send audio.start
        await websocket.send_json({
            "type": "audio.start",
            "sentence_index": sentence_index,
            "sentence_text": sentence_text,
            "format": response_format,
        })

        try:
            # Build a per-sentence request reusing session config
            request = OpenAICreateSpeechRequest(
                input=sentence_text,
                model=config.model,
                voice=config.voice,
                task_type=config.task_type,
                language=config.language,
                instructions=config.instructions,
                response_format=response_format,
                speed=config.speed,
                max_new_tokens=config.max_new_tokens,
                ref_audio=config.ref_audio,
                ref_text=config.ref_text,
                x_vector_only_mode=config.x_vector_only_mode,
            )

            audio_bytes, _ = await self._speech_service._generate_audio_bytes(
                request
            )

            # Send binary audio frame
            await websocket.send_bytes(audio_bytes)

        except Exception as e:
            logger.error(
                "Generation failed for sentence %d: %s", sentence_index, e
            )
            await self._send_error(
                websocket,
                f"Generation failed for sentence {sentence_index}: {e}",
            )

        # Send audio.done (even on error, so client can track progress)
        await websocket.send_json({
            "type": "audio.done",
            "sentence_index": sentence_index,
        })

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json({
                "type": "error",
                "message": message,
            })
        except Exception:
            pass  # Connection may already be closed
