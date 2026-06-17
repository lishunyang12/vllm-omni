# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge the JoyVL webui's ASR WebSocket protocol to an OpenAI-compatible
``/v1/audio/transcriptions`` backend (e.g. a served Qwen3-ASR).

The webui's ``ASR_URL`` streams microphone audio as binary frames
``struct.pack(">iii", seqid, 0, 0) + pcm16`` (a negative ``seqid`` marks the
final frame) and reads JSON results shaped like
``{"asr_response": {"event_type": "IS_FINAL",
"recognition_result": {"hypothesis": [{"text": ...}]}}}``.

This bridge accumulates the PCM, and on the final frame transcribes the whole
utterance and returns one ``IS_FINAL`` result. Streaming partials can be added
once a streaming transcription backend is available.

    ASR_URL=ws://127.0.0.1:8093/v1/asr
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import wave

import aiohttp
from aiohttp import web

_HEADER = struct.Struct(">iii")
_SAMPLE_RATE = 16000


def _pcm_to_wav(pcm: bytes, sample_rate: int = _SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def _result(text: str, *, final: bool) -> dict:
    return {
        "asr_response": {
            "event_type": "IS_FINAL" if final else "IS_PARTIAL",
            "recognition_result": {"hypothesis": [{"text": text, "confidence": None}]},
        },
        "mid": "",
        "code": 0,
        "msg": "ok",
    }


async def _transcribe(cfg: dict, pcm: bytes) -> str:
    if not pcm:
        return ""
    wav = _pcm_to_wav(pcm, cfg["sample_rate"])
    form = aiohttp.FormData()
    form.add_field("model", cfg["model"])
    form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
    url = cfg["backend_url"].rstrip("/") + "/v1/audio/transcriptions"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form) as resp:
            payload = await resp.json()
    return (payload.get("text") or "").strip()


async def _handle(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    cfg = request.app["cfg"]
    pcm = bytearray()
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY and len(msg.data) >= _HEADER.size:
                seqid, _, _ = _HEADER.unpack(msg.data[: _HEADER.size])
                pcm.extend(msg.data[_HEADER.size :])
                if seqid < 0:  # final frame
                    text = await _transcribe(cfg, bytes(pcm))
                    await ws.send_str(json.dumps(_result(text, final=True), ensure_ascii=False))
                    pcm.clear()
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
    finally:
        if not ws.closed:
            await ws.close()
    return ws


def create_app(backend_url: str, model: str, sample_rate: int) -> web.Application:
    app = web.Application()
    app["cfg"] = {"backend_url": backend_url, "model": model, "sample_rate": sample_rate}
    app.router.add_get("/v1/asr", _handle)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="JoyVL webui <-> ASR transcription bridge")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8094")
    parser.add_argument("--model", default="qwen3-asr")
    parser.add_argument("--sample-rate", type=int, default=_SAMPLE_RATE)
    args = parser.parse_args()
    web.run_app(create_app(args.backend_url, args.model, args.sample_rate), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
