# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge the JoyVL webui's TTS WebSocket protocol to vLLM-Omni's Qwen3-TTS.

The webui's ``TTS_URL`` speaks: ``{"config": {...}}`` then ``input_text.append`` /
``input_text.commit``, and expects ``response.audio.delta`` (or raw PCM binary)
followed by ``response.done``. vLLM-Omni's Qwen3-TTS serves ``/v1/audio/speech/
stream``: ``session.config`` then ``input.text`` / ``input.done``, replying with
binary audio plus ``audio.start`` / ``audio.done`` / ``session.done``.

Point the webui at this bridge: ``TTS_URL=ws://127.0.0.1:8092/v1/tts``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

import aiohttp
from aiohttp import web

logger = logging.getLogger("tts_bridge")


async def _pump_backend_to_front(back: aiohttp.ClientWebSocketResponse, front: web.WebSocketResponse) -> None:
    async for msg in back:
        if msg.type == aiohttp.WSMsgType.BINARY:
            await front.send_bytes(msg.data)
        elif msg.type == aiohttp.WSMsgType.TEXT:
            event = json.loads(msg.data).get("type")
            if event == "session.done":
                await front.send_str(json.dumps({"type": "response.done"}))
                return
            if event == "error":
                await front.send_str(json.dumps({"type": "error", "error": "tts backend error"}))
                return
        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
            return


async def _handle(request: web.Request) -> web.WebSocketResponse:
    front = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
    await front.prepare(request)
    cfg = request.app["cfg"]

    session = aiohttp.ClientSession()
    back: aiohttp.ClientWebSocketResponse | None = None
    pump: asyncio.Task | None = None
    try:
        async for msg in front:
            if msg.type != web.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)

            if "config" in data and back is None:
                voice = (data["config"] or {}).get("voice") or cfg["voice"]
                back = await session.ws_connect(cfg["backend_url"], max_msg_size=0)
                await back.send_str(
                    json.dumps({"type": "session.config", "response_format": cfg["response_format"], "voice": voice})
                )
                pump = asyncio.create_task(_pump_backend_to_front(back, front))
            elif data.get("type") == "input_text.append" and back is not None:
                await back.send_str(json.dumps({"type": "input.text", "text": data.get("text", "")}))
            elif data.get("type") == "input_text.commit" and back is not None:
                await back.send_str(json.dumps({"type": "input.done"}))
                if pump is not None:
                    await pump
                break
    except Exception as err:
        logger.warning("tts bridge error: %s", err)
    finally:
        if pump is not None and not pump.done():
            pump.cancel()
        if back is not None and not back.closed:
            await back.close()
        await session.close()
        if not front.closed:
            await front.close()
    return front


def create_app(backend_url: str, voice: str, response_format: str) -> web.Application:
    app = web.Application()
    app["cfg"] = {"backend_url": backend_url, "voice": voice, "response_format": response_format}
    app.router.add_get("/v1/tts", _handle)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="JoyVL webui <-> Qwen3-TTS websocket bridge")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--backend-url", default="ws://127.0.0.1:8091/v1/audio/speech/stream")
    parser.add_argument("--voice", default="vivian")
    parser.add_argument("--response-format", default="pcm")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    web.run_app(create_app(args.backend_url, args.voice, args.response_format), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
