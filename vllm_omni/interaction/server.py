# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vllm_omni.interaction.backend import OpenAIBackend
from vllm_omni.interaction.config import InteractionConfig
from vllm_omni.interaction.delegation import StubDelegationBridge
from vllm_omni.interaction.memory import Summarizer
from vllm_omni.interaction.output_parser import to_token_form
from vllm_omni.interaction.session import InteractionSession, StepResult


class SessionManager:
    def __init__(self, config: InteractionConfig) -> None:
        self.config = config
        self._backend = OpenAIBackend(
            config.main_backend_url, config.main_model, config.api_key, config.request_timeout_seconds
        )
        self._summarizer: Summarizer | None = None
        if config.enable_memory:
            summarizer_backend = OpenAIBackend(
                config.resolved_summarizer_url,
                config.resolved_summarizer_model,
                config.api_key,
                config.request_timeout_seconds,
            )
            self._summarizer = Summarizer(
                summarizer_backend,
                key_frames_per_chunk=config.mid_term_key_frames,
                mid_term_max_tokens=config.mid_term_max_tokens,
                long_term_max_tokens=config.long_term_max_tokens,
            )
        self._sessions: dict[str, InteractionSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get(self, session_id: str) -> InteractionSession:
        session = self._sessions.get(session_id)
        if session is None:
            session = InteractionSession(
                session_id,
                self.config,
                self._backend,
                summarizer=self._summarizer,
                delegation=StubDelegationBridge() if self.config.enable_delegation else None,
            )
            self._sessions[session_id] = session
            self._locks[session_id] = asyncio.Lock()
        return session

    async def step(self, session_id: str, frames: list[str], query: str | None) -> StepResult:
        self._evict_expired()
        session = self._get(session_id)
        async with self._locks[session_id]:
            return await session.step(frames, query)

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._locks.pop(session_id, None)

    def set_persona(self, session_id: str, persona: str) -> bool:
        return self._get(session_id).set_persona(persona)

    def _evict_expired(self) -> None:
        ttl = self.config.session_timeout_seconds
        if ttl <= 0:
            return
        now = time.monotonic()
        for sid in [s for s, sess in self._sessions.items() if now - sess.last_access > ttl]:
            self.reset(sid)


def _extract_frames_and_query(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    messages = payload.get("messages") or []
    frames: list[str] = []
    texts: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content or []:
            ptype = part.get("type")
            if ptype == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url:
                    frames.append(url)
            elif ptype == "text" and part.get("text"):
                texts.append(part["text"])
    query = "\n".join(t.strip() for t in texts if t.strip()) or None
    return frames, query


def _session_id(request: Request, payload: dict[str, Any]) -> str:
    return (
        request.headers.get("x-streaming-session")
        or request.headers.get("x-session-id")
        or payload.get("session_id")
        or payload.get("user")
        or "default"
    )


def _completion_response(model: str, result: StepResult) -> dict[str, Any]:
    action = result.action
    memory = {"long_term_memory": result.long_term_memory, "mid_term_summaries": result.mid_term_summaries}
    return {
        "id": f"intchat-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": to_token_form(action)},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
        "streamingharness": {
            "memory": memory,
            "timing": {"adapter_total_ms": result.latency_ms},
        },
        "interaction": {
            "action": action.action.value,
            "spoke": action.spoke,
            "text": action.text,
            "delegated_question": action.delegated_question,
            "delegation": result.delegation,
            "chunk_index": result.chunk_index,
            "frame_index": result.frame_index,
            "inference_skipped": result.inference_skipped,
            "latency_ms": result.latency_ms,
            "memory": memory,
        },
    }


def create_app(config: InteractionConfig) -> FastAPI:
    app = FastAPI(title="vLLM-Omni Interaction Server")
    manager = SessionManager(config)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": config.main_model, "object": "model", "owned_by": "vllm-omni"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        payload = await request.json()
        frames, query = _extract_frames_and_query(payload)
        if not frames:
            return JSONResponse({"error": "interaction server requires at least one image_url frame"}, status_code=400)
        result = await manager.step(_session_id(request, payload), frames, query)
        return JSONResponse(_completion_response(config.main_model, result))

    @app.post("/reset")
    @app.post("/v1/streaming/reset")
    async def reset(request: Request) -> dict[str, str]:
        payload = await request.json() if await request.body() else {}
        manager.reset(_session_id(request, payload))
        return {"status": "reset"}

    @app.post("/v1/streaming/persona")
    async def persona(request: Request) -> dict[str, Any]:
        payload = await request.json() if await request.body() else {}
        ok = manager.set_persona(_session_id(request, payload), payload.get("persona", "default"))
        return {"status": "ok" if ok else "unknown_persona"}

    return app


def _build_config(args: argparse.Namespace) -> InteractionConfig:
    config = InteractionConfig(
        main_backend_url=args.main_backend_url,
        main_model=args.main_model,
        persona=args.persona,
        enable_memory=not args.no_memory,
        summarizer_backend_url=args.summarizer_backend_url,
        summarizer_model=args.summarizer_model,
        enable_delegation=not args.no_delegation,
        force_silence_before_query=not args.no_force_silence,
    )
    if args.chunk_frames is not None:
        config.chunk_frames = args.chunk_frames
    config.sampling.max_tokens = args.max_tokens
    config.sampling.temperature = args.temperature
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM-Omni streaming interaction server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--main-backend-url", default="http://127.0.0.1:8061/v1")
    parser.add_argument("--main-model", default="JoyAI-VL-Interaction-Preview")
    parser.add_argument("--persona", default="default", choices=["default", "silent", "talkative"])
    parser.add_argument("--summarizer-backend-url", default=None)
    parser.add_argument("--summarizer-model", default=None)
    parser.add_argument("--chunk-frames", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-memory", action="store_true", help="disable mid/long-term summarizer memory")
    parser.add_argument("--no-delegation", action="store_true", help="disable the delegation bridge")
    parser.add_argument("--no-force-silence", action="store_true", help="run the model before any user query")
    args = parser.parse_args()

    app = create_app(_build_config(args))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
