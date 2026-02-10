# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-aware profiler HTTP endpoints.

Follows the same API shape as ``vllm.entrypoints.serve.profile.api_router``
(``/start_profile``, ``/stop_profile``) and extends it with an optional
``stages`` parameter for multi-stage pipeline profiling.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


class ProfileRequest(BaseModel):
    """Request body for profiler endpoints.

    Attributes:
        stages: List of stage IDs to profile. If None, profiles all stages.
            For Qwen2.5-Omni / Qwen3-Omni:
              - 0 = Thinker (multimodal understanding)
              - 1 = Talker (text to codec codes)
              - 2 = Code2Wav (codec to audio)
    """

    stages: list[int] | None = None


@router.post("/start_profile")
async def start_profile(raw_request: Request, body: ProfileRequest | None = None):
    stages = body.stages if body else None
    engine_client = raw_request.app.state.engine_client
    stage_desc = f"stages={stages}" if stages else "all stages"
    logger.info("Starting profiler for %s...", stage_desc)
    await engine_client.start_profile(stages=stages)
    logger.info("Profiler started for %s.", stage_desc)
    return Response(status_code=200)


@router.post("/stop_profile")
async def stop_profile(raw_request: Request, body: ProfileRequest | None = None):
    stages = body.stages if body else None
    engine_client = raw_request.app.state.engine_client
    stage_desc = f"stages={stages}" if stages else "all stages"
    logger.info("Stopping profiler for %s...", stage_desc)
    await engine_client.stop_profile(stages=stages)
    logger.info("Profiler stopped for %s.", stage_desc)
    return Response(status_code=200)


def attach_router(app: FastAPI):
    """Attach profiler routes if profiler is configured.

    Mirrors the check in ``vllm.entrypoints.serve.profile.api_router``
    but uses the routes defined here (which support ``stages``).
    """
    profiler_config = getattr(app.state.args, "profiler_config", None)
    if profiler_config is not None and getattr(profiler_config, "profiler", None) is not None:
        logger.warning(
            "Profiler with mode '%s' is enabled in the API server. This should ONLY be used for local development!",
            profiler_config.profiler,
        )
        app.include_router(router)
