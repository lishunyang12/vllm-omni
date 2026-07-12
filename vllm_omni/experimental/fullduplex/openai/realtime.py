# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64

from vllm_omni.experimental.fullduplex.core.events import (
    EmitProtocolEvent,
    ModelAudioDelta,
    ModelTextDelta,
    ProtocolEventKind,
)
from vllm_omni.experimental.fullduplex.openai.history import ResponseLifecycleLedger


class RealtimeEventProjector:
    """Purely project fenced core effects into OpenAI Realtime events."""

    def __init__(self, lifecycle: ResponseLifecycleLedger | None = None) -> None:
        self.lifecycle = lifecycle or ResponseLifecycleLedger()

    def project(self, effect: EmitProtocolEvent) -> tuple[dict[str, object], ...]:
        fence = effect.fence
        if effect.kind is ProtocolEventKind.RESPONSE_STARTED:
            response_id = self.lifecycle.start(fence)
            return (
                {
                    "type": "response.created",
                    "response": {"id": response_id, "status": "in_progress"},
                },
            )
        response_id = self.lifecycle.response_id(fence)
        if effect.kind is ProtocolEventKind.TEXT_DELTA:
            # The reducer carries the ModelTextDelta event as the payload; read
            # its text rather than stringifying the dataclass repr.
            payload = effect.payload
            text = payload.text if isinstance(payload, ModelTextDelta) else str(payload or "")
            self.lifecycle.append_text(fence, text)
            return ({"type": "response.output_text.delta", "response_id": response_id, "delta": text},)
        if effect.kind is ProtocolEventKind.AUDIO_DELTA:
            payload = effect.payload
            if isinstance(payload, ModelAudioDelta):
                data = payload.data
            elif isinstance(payload, bytes):
                data = payload
            else:
                data = b""
            self.lifecycle.append_audio(fence, data)
            return (
                {
                    "type": "response.output_audio.delta",
                    "response_id": response_id,
                    "delta": base64.b64encode(data).decode("ascii"),
                },
            )
        if effect.kind is ProtocolEventKind.SEGMENT_ENDED:
            return ({"type": "response.output_audio.done", "response_id": response_id},)
        if effect.kind is ProtocolEventKind.RESPONSE_COMPLETED:
            transcript = self.lifecycle.finish(fence)
            return (
                {
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "status": "completed",
                        "transcript": transcript,
                    },
                },
            )
        if effect.kind is ProtocolEventKind.MODEL_LISTENING:
            return ({"type": "response.listen", "response_id": response_id},)
        if effect.kind is ProtocolEventKind.ENGINE_FAILED:
            return ({"type": "error", "error": str(effect.payload or "engine failure")},)
        return ()


class DemoRealtimeProjector:
    """Project fenced core effects into the compact event dialect the bundled
    ``realtime_web`` demo (``app.js``) consumes.

    ``app.js`` predates the verbose OpenAI-Realtime names and listens for
    ``response.audio.delta`` / ``response.audio_transcript.delta`` /
    ``response.audio.done`` instead of the ``response.output_*`` variants. Audio
    is the model's 24 kHz little-endian ``float32`` PCM, tagged so the client
    decoder (``decodeOutputAudioDelta``) picks the right path.
    """

    OUTPUT_AUDIO_FORMAT = "pcm_f32le"
    OUTPUT_SAMPLE_RATE_HZ = 24000

    def __init__(self, lifecycle: ResponseLifecycleLedger | None = None) -> None:
        self.lifecycle = lifecycle or ResponseLifecycleLedger()

    def project(self, effect: EmitProtocolEvent) -> tuple[dict[str, object], ...]:
        fence = effect.fence
        if effect.kind is ProtocolEventKind.RESPONSE_STARTED:
            response_id = self.lifecycle.start(fence)
            return ({"type": "response.created", "response": {"id": response_id, "status": "in_progress"}},)
        response_id = self.lifecycle.response_id(fence)
        if effect.kind is ProtocolEventKind.TEXT_DELTA:
            payload = effect.payload
            text = payload.text if isinstance(payload, ModelTextDelta) else str(payload or "")
            self.lifecycle.append_text(fence, text)
            return ({"type": "response.audio_transcript.delta", "response_id": response_id, "delta": text},)
        if effect.kind is ProtocolEventKind.AUDIO_DELTA:
            payload = effect.payload
            if isinstance(payload, ModelAudioDelta):
                data = payload.data
            elif isinstance(payload, bytes):
                data = payload
            else:
                data = b""
            self.lifecycle.append_audio(fence, data)
            return (
                {
                    "type": "response.audio.delta",
                    "response_id": response_id,
                    "delta": base64.b64encode(data).decode("ascii"),
                    "format": self.OUTPUT_AUDIO_FORMAT,
                    "sample_rate_hz": self.OUTPUT_SAMPLE_RATE_HZ,
                },
            )
        if effect.kind is ProtocolEventKind.SEGMENT_ENDED:
            return ({"type": "response.audio.done", "response_id": response_id},)
        if effect.kind is ProtocolEventKind.RESPONSE_COMPLETED:
            transcript = self.lifecycle.finish(fence)
            return (
                {"type": "response.audio_transcript.done", "response_id": response_id, "transcript": transcript},
                {"type": "response.done", "response": {"id": response_id, "status": "completed"}},
            )
        if effect.kind is ProtocolEventKind.MODEL_LISTENING:
            return ({"type": "response.listen", "response_id": response_id},)
        if effect.kind is ProtocolEventKind.ENGINE_FAILED:
            return ({"type": "error", "error": str(effect.payload or "engine failure")},)
        return ()
