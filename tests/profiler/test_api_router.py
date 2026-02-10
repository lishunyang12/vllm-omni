# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the stage-aware profiler HTTP endpoints.

These tests use FastAPI TestClient with a mocked engine_client,
so they run without GPU or model weights.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.serve.profile.api_router import attach_router, router


def _make_app(profiler_config=None) -> FastAPI:
    """Create a minimal FastAPI app with mocked state for testing."""
    app = FastAPI()

    mock_engine = AsyncMock()
    mock_engine.start_profile = AsyncMock()
    mock_engine.stop_profile = AsyncMock()

    app.state.engine_client = mock_engine
    app.state.args = SimpleNamespace(profiler_config=profiler_config)
    return app


class TestAttachRouter:
    """Tests for attach_router conditional logic."""

    def test_attaches_when_profiler_set(self):
        """Routes are registered when profiler_config.profiler is set."""
        config = SimpleNamespace(profiler="torch", torch_profiler_dir="/tmp")
        app = _make_app(profiler_config=config)
        attach_router(app)

        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/start_profile" in paths
        assert "/stop_profile" in paths

    def test_not_attached_when_no_config(self):
        """Routes are NOT registered when profiler_config is None."""
        app = _make_app(profiler_config=None)
        attach_router(app)

        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/start_profile" not in paths
        assert "/stop_profile" not in paths

    def test_not_attached_when_profiler_none(self):
        """Routes are NOT registered when profiler_config.profiler is None."""
        config = SimpleNamespace(profiler=None)
        app = _make_app(profiler_config=config)
        attach_router(app)

        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/start_profile" not in paths
        assert "/stop_profile" not in paths


class TestStartProfileEndpoint:
    """Tests for POST /start_profile."""

    @pytest.fixture()
    def client(self):
        app = _make_app()
        app.include_router(router)
        return TestClient(app)

    def test_start_profile_no_body(self, client):
        """Empty body profiles all stages."""
        resp = client.post("/start_profile")
        assert resp.status_code == 200
        engine = client.app.state.engine_client
        engine.start_profile.assert_awaited_once_with(stages=None)

    def test_start_profile_with_stages(self, client):
        """Body with stages=[0] profiles only stage 0."""
        resp = client.post("/start_profile", json={"stages": [0]})
        assert resp.status_code == 200
        engine = client.app.state.engine_client
        engine.start_profile.assert_awaited_once_with(stages=[0])

    def test_start_profile_multiple_stages(self, client):
        """Body with stages=[0,2] profiles stages 0 and 2."""
        resp = client.post("/start_profile", json={"stages": [0, 2]})
        assert resp.status_code == 200
        engine = client.app.state.engine_client
        engine.start_profile.assert_awaited_once_with(stages=[0, 2])


class TestStopProfileEndpoint:
    """Tests for POST /stop_profile."""

    @pytest.fixture()
    def client(self):
        app = _make_app()
        app.include_router(router)
        return TestClient(app)

    def test_stop_profile_no_body(self, client):
        """Empty body stops all stages."""
        resp = client.post("/stop_profile")
        assert resp.status_code == 200
        engine = client.app.state.engine_client
        engine.stop_profile.assert_awaited_once_with(stages=None)

    def test_stop_profile_with_stages(self, client):
        """Body with stages=[1] stops only stage 1."""
        resp = client.post("/stop_profile", json={"stages": [1]})
        assert resp.status_code == 200
        engine = client.app.state.engine_client
        engine.stop_profile.assert_awaited_once_with(stages=[1])
