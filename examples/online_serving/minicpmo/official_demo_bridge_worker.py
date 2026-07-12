"""Bridge worker: official MiniCPM-o-Demo frontend/gateway -> vLLM duplex.

Implements the worker surface the official OpenBMB/MiniCPM-o-Demo gateway
expects (``GET /health`` + ``WS /ws/duplex``) and translates the official
per-chunk duplex protocol onto vLLM-Omni's ``/v1/realtime?duplex=1`` endpoint
in full-duplex (``extra_body.auto_response``) mode, so the official prebuilt
web frontend can drive a vLLM-served MiniCPM-o 4.5 session unchanged.

Official worker protocol (client side of ``/ws/duplex``):
  {"type": "prepare", system_prompt, ref_audio_base64?, tts_ref_audio_base64?}
      -> {"type": "prepared", "prompt_length": int}
  {"type": "audio_chunk", "audio_base64": <f32le 16 kHz b64>, force_listen?}
      -> {"type": "result", is_listen, text, audio_data (f32le 24 kHz b64),
          end_of_turn, current_time, wall_clock_ms, kv_cache_length, ...}
  {"type": "pause"} -> {"type": "paused"}; {"type": "resume"} -> {"type": "resumed"}
  {"type": "stop"} -> {"type": "stopped"}

Run (after the vLLM duplex server is up on --vllm-ws):
  python official_demo_bridge_worker.py --port 22500 \
      --vllm-ws ws://localhost:8099/v1/realtime?duplex=1 \
      --model /models/MiniCPM-o-4_5

Then point the official gateway at it:
  python gateway.py --http --workers localhost:22500
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import time
import wave

import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger("official_demo_bridge")

ARGS: argparse.Namespace | None = None

app = FastAPI(title="MiniCPM-o vLLM duplex bridge worker")

_state = {"status": "idle", "session_id": None, "total": 0}


@app.get("/health")
async def health() -> dict:
    return {
        "worker_status": _state["status"],
        "current_session_id": _state["session_id"],
        "total_requests": _state["total"],
        "avg_inference_time_ms": 0.0,
    }


def _f32b64_to_pcm16_b64(audio_b64: str) -> str:
    f32 = np.frombuffer(base64.b64decode(audio_b64), dtype=np.float32)
    pcm16 = np.clip(f32 * 32767.0, -32768, 32767).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


def _pcm16_b64_to_f32_b64(audio_b64: str) -> str:
    pcm16 = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
    f32 = (pcm16.astype(np.float32) / 32768.0).astype(np.float32)
    return base64.b64encode(f32.tobytes()).decode("ascii")


def _f32b64_to_wav_data_uri(audio_b64: str, sample_rate: int = 16000) -> str:
    """Official prepare carries ref audio as f32le b64; vLLM's serving adapter
    resolves ``extra_body.ref_audio`` URIs, so wrap it as a WAV data URI."""
    f32 = np.frombuffer(base64.b64decode(audio_b64), dtype=np.float32)
    pcm16 = np.clip(f32 * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class _VllmSession:
    """One official duplex session mapped onto one vLLM realtime session."""

    # One second at 24 kHz: the official per-result audio rhythm the
    # prebuilt frontend schedules playback against.
    _UNIT_SAMPLES = 24000

    def __init__(self, system_prompt: str, ref_audio_b64: str | None) -> None:
        self.system_prompt = system_prompt
        self.ref_audio_b64 = ref_audio_b64
        self.ws: websockets.WebSocketClientProtocol | None = None
        # FIFO of per-chunk decisions produced by the reader task.
        self.decisions: asyncio.Queue[dict] = asyncio.Queue()
        self.reader_task: asyncio.Task | None = None
        self.chunk_idx = 0
        self.kv_cache_length: int | None = None
        # Generated reply samples awaiting paced emission (f32, 24 kHz).
        self._audio_fifo = np.zeros(0, dtype=np.float32)

    def _paced_unit_audio(self, new_f32: np.ndarray | None, *, flush: bool) -> str | None:
        """Re-pace reply audio onto the official per-result rhythm.

        The official worker returns exactly one second of audio with EVERY
        result: listens carry silence, non-final speak units are left-padded
        to one second, and only the turn-final result carries an unpadded
        tail. The frontend's playback scheduler assumes that rhythm, so
        forwarding variable-size deltas makes it drift and clip. Buffer the
        generated samples and emit one unit per result.
        """
        if new_f32 is not None and len(new_f32):
            self._audio_fifo = np.concatenate([self._audio_fifo, new_f32])
        if flush:
            out = self._audio_fifo
            self._audio_fifo = np.zeros(0, dtype=np.float32)
            if not len(out):
                return None
        elif len(self._audio_fifo) >= self._UNIT_SAMPLES:
            out = self._audio_fifo[: self._UNIT_SAMPLES]
            self._audio_fifo = self._audio_fifo[self._UNIT_SAMPLES :]
        else:
            pending = self._audio_fifo
            self._audio_fifo = np.zeros(0, dtype=np.float32)
            out = np.zeros(self._UNIT_SAMPLES, dtype=np.float32)
            if len(pending):
                out[-len(pending) :] = pending
        return base64.b64encode(out.astype("<f4").tobytes()).decode("ascii")

    async def open(self) -> None:
        assert ARGS is not None
        self.ws = await websockets.connect(ARGS.vllm_ws, max_size=64 * 1024 * 1024)
        extra_body: dict = {
            "minicpmo45_native_duplex": True,
            "auto_response": True,
        }
        if self.ref_audio_b64:
            extra_body["ref_audio"] = _f32b64_to_wav_data_uri(self.ref_audio_b64)
        await self.ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "model": ARGS.model,
                        "modalities": ["audio", "text"],
                        "instructions": self.system_prompt,
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "turn_detection": None,
                        "extra_body": extra_body,
                    },
                }
            )
        )
        # Wait for session.created before declaring the session prepared.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=30))
            if msg.get("type") == "session.created":
                break
            if msg.get("type") == "error":
                raise RuntimeError(str(msg))
        self.reader_task = asyncio.create_task(self._reader())

    async def _reader(self) -> None:
        """Fold the vLLM realtime event stream into per-chunk decisions.

        The serving side keeps one response open for the whole spoken turn
        and streams one audio/transcript delta pair per speak unit, so each
        delta pair is flushed as its own per-chunk decision; ``audio.done``
        and ``response.done`` arrive once per turn.
        """
        assert self.ws is not None
        speak: dict | None = None

        async def flush_speak() -> None:
            nonlocal speak
            if speak is not None and (speak["audio"] or speak["text"]):
                await self.decisions.put(speak)
            speak = None

        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                mtype = msg.get("type", "")
                logger.info("vllm evt %s t=%.3f", mtype, time.monotonic())
                kv = msg.get("kv_cache_length")
                if isinstance(kv, int):
                    self.kv_cache_length = kv
                if mtype == "response.listen":
                    await flush_speak()
                    if msg.get("buffering") is True:
                        # Sub-chunk append still accumulating server-side;
                        # the official protocol has no such state, report listen.
                        await self.decisions.put({"is_listen": True, "buffering": True})
                    else:
                        await self.decisions.put({"is_listen": True})
                elif mtype == "response.speak":
                    text = msg.get("text") or ""
                    if speak is None:
                        speak = {"is_listen": False, "text": text, "audio": []}
                    elif text and text not in speak["text"]:
                        speak["text"] += text
                elif mtype in ("response.audio.delta", "response.output_audio.delta"):
                    if speak is not None and speak["audio"]:
                        # A new unit's audio while one is pending: flush the
                        # previous unit as its own per-chunk decision.
                        await flush_speak()
                    if speak is None:
                        speak = {"is_listen": False, "text": "", "audio": []}
                    delta = msg.get("delta") or msg.get("audio") or ""
                    if delta:
                        speak["audio"].append(delta)
                elif mtype in ("response.audio_transcript.delta",):
                    if speak is not None and isinstance(msg.get("delta"), str):
                        if msg["delta"] not in speak["text"]:
                            speak["text"] += msg["delta"]
                    # Audio + transcript complete one speak unit.
                    await flush_speak()
                elif mtype in ("response.audio.done", "response.output_audio.done"):
                    await flush_speak()
                elif mtype == "response.done":
                    await flush_speak()
                    await self.decisions.put({"end_of_turn": True})
                elif mtype == "error":
                    await self.decisions.put({"error": str(msg.get("error") or msg)})
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass
        finally:
            await self.decisions.put({"closed": True})

    async def append_chunk(self, audio_b64_f32: str, *, force_listen: bool) -> dict:
        """Send one official audio chunk; await this chunk's decision."""
        assert self.ws is not None
        t0 = time.perf_counter()
        event: dict = {
            "type": "input_audio_buffer.append",
            "audio": _f32b64_to_pcm16_b64(audio_b64_f32),
        }
        if force_listen:
            event["force_listen"] = True
        await self.ws.send(json.dumps(event))
        logger.info("append sent chunk=%d t=%.3f", self.chunk_idx, time.monotonic())
        self.chunk_idx += 1

        result = {
            "is_listen": True,
            "text": "",
            "audio_data": None,
            "end_of_turn": False,
            "current_time": self.chunk_idx,
        }
        try:
            decision = await asyncio.wait_for(self.decisions.get(), timeout=ARGS.chunk_timeout_s)
            if decision.get("closed") or decision.get("error"):
                result["error"] = decision.get("error", "session closed")
            elif decision.get("end_of_turn"):
                result["end_of_turn"] = True
                result["is_listen"] = False
                result["audio_data"] = self._paced_unit_audio(None, flush=True)
            elif decision.get("is_listen") is False:
                result["is_listen"] = False
                result["text"] = decision.get("text") or ""
                pcm16_parts = decision.get("audio") or []
                new_f32 = None
                if pcm16_parts:
                    pcm16 = np.frombuffer(
                        b"".join(base64.b64decode(p) for p in pcm16_parts),
                        dtype=np.int16,
                    )
                    new_f32 = (pcm16.astype(np.float32) / 32768.0).astype(np.float32)
                # A response.done right behind the audio marks the turn end.
                try:
                    nxt = await asyncio.wait_for(self.decisions.get(), timeout=0.25)
                    if nxt.get("end_of_turn"):
                        result["end_of_turn"] = True
                    elif not nxt.get("closed"):
                        self.decisions.put_nowait(nxt)
                except asyncio.TimeoutError:
                    pass
                result["audio_data"] = self._paced_unit_audio(new_f32, flush=result["end_of_turn"])
            else:
                # Listen result: the official worker returns one second of
                # silence so the frontend's playback clock keeps ticking.
                result["audio_data"] = self._paced_unit_audio(None, flush=False)
        except asyncio.TimeoutError:
            logger.warning("chunk %d: no decision within %ss", self.chunk_idx, ARGS.chunk_timeout_s)
            result["audio_data"] = self._paced_unit_audio(None, flush=False)

        result["wall_clock_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["kv_cache_length"] = self.kv_cache_length
        result["cost_all_ms"] = result["wall_clock_ms"]
        return result

    async def close(self) -> None:
        if self.reader_task is not None:
            self.reader_task.cancel()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass


@app.websocket("/ws/duplex")
@app.websocket("/v1/worker/duplex")
async def duplex_ws(ws: WebSocket) -> None:
    await ws.accept()
    _state["status"] = "duplex_active"
    _state["session_id"] = ws.query_params.get("session_id") or "bridge"
    _state["total"] += 1
    session: _VllmSession | None = None
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            mtype = msg.get("type", "")
            if mtype == "prepare":
                if session is not None:
                    await session.close()
                ref_b64 = msg.get("tts_ref_audio_base64") or msg.get("ref_audio_base64")
                session = _VllmSession(
                    msg.get("system_prompt") or "Streaming Omni Conversation.",
                    ref_b64,
                )
                try:
                    await session.open()
                except Exception as exc:  # noqa: BLE001 - surfaced to the client
                    logger.exception("prepare failed")
                    await ws.send_json({"type": "error", "error": str(exc)})
                    session = None
                    continue
                await ws.send_json({"type": "prepared", "prompt_length": 0})
            elif mtype == "audio_chunk":
                if session is None:
                    await ws.send_json({"type": "error", "error": "prepare first"})
                    continue
                audio_b64 = msg.get("audio_base64")
                if not audio_b64:
                    await ws.send_json({"type": "error", "error": "Missing audio_base64"})
                    continue
                result = await session.append_chunk(
                    audio_b64,
                    force_listen=bool(msg.get("force_listen", False)),
                )
                err = result.pop("error", None)
                if err:
                    await ws.send_json({"type": "error", "error": err})
                else:
                    await ws.send_json({"type": "result", **result})
            elif mtype == "pause":
                _state["status"] = "duplex_paused"
                await ws.send_json({"type": "paused", "timeout": msg.get("timeout", 60.0)})
            elif mtype == "resume":
                _state["status"] = "duplex_active"
                await ws.send_json({"type": "resumed"})
            elif mtype == "stop":
                await ws.send_json({"type": "stopped"})
                break
            else:
                await ws.send_json({"type": "error", "error": f"unknown type: {mtype}"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if session is not None:
            await session.close()
        _state["status"] = "idle"
        _state["session_id"] = None


def main() -> None:
    global ARGS  # noqa: PLW0603 - simple CLI singleton
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=22500)
    parser.add_argument("--vllm-ws", default="ws://localhost:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", default="/models/MiniCPM-o-4_5")
    parser.add_argument("--chunk-timeout-s", type=float, default=20.0)
    ARGS = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="info")


if __name__ == "__main__":
    main()
