"""WebSocket handler for streaming video input understanding.

Accepts video frames incrementally via WebSocket, buffers them, and
generates text + optional audio responses using the existing Qwen3-Omni
multi-stage pipeline (thinker -> talker -> code2wav).

Protocol:
    Client -> Server:
        {"type": "session.config", ...}         # Session config (sent once)
        {"type": "video.frame", "data": "..."}  # base64 JPEG/PNG frame
        {"type": "video.query", "text": "..."}  # Submit query about buffered frames
        {"type": "video.done"}                  # End of session

    Server -> Client:
        {"type": "response.start"}
        {"type": "response.text.delta", "delta": "..."}
        {"type": "response.text.done", "text": "..."}
        {"type": "response.audio.start", "format": "wav"}
        <binary frame: audio bytes>
        {"type": "response.audio.done"}
        {"type": "session.done"}
        {"type": "error", "message": "..."}
"""

import asyncio
import base64
import io
import json
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 60.0  # seconds (longer than speech — video takes time)
_DEFAULT_CONFIG_TIMEOUT = 10.0
_MAX_FRAME_SIZE = 10 * 1024 * 1024  # 10MB per frame
_MAX_BUFFER_FRAMES = 64


class StreamingVideoSessionConfig(BaseModel):
    """Configuration sent as the first WebSocket message."""

    model: str | None = None
    modalities: list[str] = Field(
        default=["text"],
        description="Output modalities: 'text', 'audio', or both.",
    )
    num_frames: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Max frames to sample from buffer for the model.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt. Uses Qwen3-Omni default if not set.",
    )
    use_audio_in_video: bool = Field(
        default=False,
        description="Extract and interleave audio from video frames.",
    )
    sampling_params_list: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-stage sampling params [thinker, talker, code2wav].",
    )


class OmniStreamingVideoHandler:
    """Handles WebSocket sessions for streaming video input.

    Each WebSocket connection is an independent session. Video frames
    arrive incrementally, are buffered, and when a query is submitted
    they are sent through the Qwen3-Omni pipeline for understanding.

    Args:
        chat_service: The chat serving instance (reused for generating
            chat completions with video input).
        idle_timeout: Max seconds to wait for a message before closing.
        config_timeout: Max seconds to wait for the initial session.config.
    """

    def __init__(
        self,
        chat_service: Any,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    ) -> None:
        self._chat_service = chat_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            # 1. Wait for session.config
            config = await self._receive_config(websocket)
            if config is None:
                return

            frame_buffer: list[str] = []  # base64-encoded frames

            # 2. Receive frames and queries
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self._idle_timeout,
                    )
                except asyncio.TimeoutError:
                    await self._send_error(
                        websocket, "Idle timeout: no message received"
                    )
                    return

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message")
                    continue

                if not isinstance(msg, dict):
                    await self._send_error(
                        websocket, "Messages must be JSON objects"
                    )
                    continue

                msg_type = msg.get("type")

                if msg_type == "video.frame":
                    frame_data = msg.get("data", "")
                    if not frame_data:
                        await self._send_error(
                            websocket, "video.frame requires 'data' field"
                        )
                        continue

                    # Validate frame size
                    if len(frame_data) > _MAX_FRAME_SIZE:
                        await self._send_error(
                            websocket, "Frame too large"
                        )
                        continue

                    # Validate it's a decodable image
                    try:
                        raw_bytes = base64.b64decode(frame_data)
                        Image.open(io.BytesIO(raw_bytes)).verify()
                    except Exception:
                        await self._send_error(
                            websocket, "Invalid image data"
                        )
                        continue

                    if len(frame_buffer) >= _MAX_BUFFER_FRAMES:
                        # Drop oldest frame (sliding window)
                        frame_buffer.pop(0)
                    frame_buffer.append(frame_data)

                elif msg_type == "video.query":
                    query_text = msg.get("text", "Describe what you see.")
                    if not frame_buffer:
                        await self._send_error(
                            websocket,
                            "No frames buffered. Send video.frame first.",
                        )
                        continue

                    await self._process_query(
                        websocket, config, frame_buffer, query_text
                    )
                    # Keep frames for follow-up queries (user can send
                    # more frames + another query in the same session)

                elif msg_type == "video.done":
                    await websocket.send_json({"type": "session.done"})
                    return

                else:
                    await self._send_error(
                        websocket,
                        f"Unknown message type: {msg_type}",
                    )

        except WebSocketDisconnect:
            logger.info("Streaming video: client disconnected")
        except Exception as e:
            logger.exception("Streaming video session error: %s", e)
            try:
                await self._send_error(
                    websocket, f"Internal error: {e}"
                )
            except Exception:
                logger.debug(
                    "Failed to send error to video client",
                    exc_info=True,
                )

    async def _receive_config(
        self, websocket: WebSocket
    ) -> StreamingVideoSessionConfig | None:
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
            await self._send_error(
                websocket, "Invalid JSON in session.config"
            )
            return None

        if not isinstance(msg, dict) or msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type') if isinstance(msg, dict) else type(msg).__name__}",
            )
            return None

        try:
            config = StreamingVideoSessionConfig(
                **{k: v for k, v in msg.items() if k != "type"}
            )
        except ValidationError as e:
            await self._send_error(
                websocket, f"Invalid session config: {e}"
            )
            return None

        return config

    async def _process_query(
        self,
        websocket: WebSocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        query_text: str,
    ) -> None:
        """Build a chat completion request from buffered frames and stream response."""
        from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

        from vllm_omni.entrypoints.openai.protocol.chat_completion import (
            ChatCompletionRequest,
        )

        # Sample frames if buffer exceeds num_frames
        frames = frame_buffer
        if len(frames) > config.num_frames:
            indices = np.linspace(
                0, len(frames) - 1, config.num_frames, dtype=int
            )
            frames = [frame_buffer[i] for i in indices]

        # Build video data URL from frames — encode as a video-like
        # sequence of images. The simplest approach: use image_url items
        # which Qwen3-Omni processes as video frames when passed as a list.
        # But the standard approach is to build a single video data URL.
        # For frames, we pass them as individual image_url items and
        # the multimodal processor handles them as video frames.
        video_content: list[dict] = []
        for frame_b64 in frames:
            video_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                    },
                }
            )

        video_content.append({"type": "text", "text": query_text})

        # Build messages
        messages: list[ChatCompletionMessageParam] = []
        if config.system_prompt:
            messages.append(
                {"role": "system", "content": config.system_prompt}
            )
        messages.append({"role": "user", "content": video_content})

        # Build request
        request_kwargs: dict[str, Any] = {
            "model": config.model or "default",
            "messages": messages,
            "stream": True,
            "modalities": config.modalities,
            "mm_processor_kwargs": {
                "use_audio_in_video": config.use_audio_in_video,
            },
        }
        if config.sampling_params_list:
            request_kwargs["sampling_params_list"] = (
                config.sampling_params_list
            )

        try:
            request = ChatCompletionRequest(**request_kwargs)
        except Exception as e:
            await self._send_error(
                websocket, f"Failed to build request: {e}"
            )
            return

        await websocket.send_json({"type": "response.start"})

        full_text = ""
        try:
            generator = await self._chat_service.create_chat_completion(
                request, raw_request=None
            )

            if hasattr(generator, "__aiter__"):
                # Streaming response
                async for chunk_str in generator:
                    if not isinstance(chunk_str, str):
                        continue
                    # SSE format: "data: {...}\n\n"
                    for line in chunk_str.strip().split("\n"):
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content and isinstance(content, str):
                            full_text += content
                            await websocket.send_json(
                                {
                                    "type": "response.text.delta",
                                    "delta": content,
                                }
                            )

                        # Check for audio content
                        audio = delta.get("audio")
                        if audio and isinstance(audio, dict):
                            audio_data = audio.get("data")
                            if audio_data:
                                await websocket.send_json(
                                    {
                                        "type": "response.audio.start",
                                        "format": "wav",
                                    }
                                )
                                await websocket.send_bytes(
                                    base64.b64decode(audio_data)
                                )
                                await websocket.send_json(
                                    {"type": "response.audio.done"}
                                )
            else:
                # Non-streaming response
                resp = generator
                if hasattr(resp, "choices") and resp.choices:
                    content = resp.choices[0].message.content
                    if content:
                        full_text = content if isinstance(content, str) else str(content)
                        await websocket.send_json(
                            {
                                "type": "response.text.delta",
                                "delta": full_text,
                            }
                        )

        except Exception as e:
            logger.error("Video query generation failed: %s", e)
            await self._send_error(
                websocket, f"Generation failed: {e}"
            )
            return

        await websocket.send_json(
            {"type": "response.text.done", "text": full_text}
        )

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass
