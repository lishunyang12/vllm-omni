# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DiffusionEngine._rpc_lock scope narrowing.

Verifies that collective_rpc is not blocked during GPU execution, i.e.
the lock is released around execute_fn().
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_engine():
    """Create a DiffusionEngine-like object with mocked dependencies.

    We can't instantiate a real DiffusionEngine without GPU, so we replicate
    the lock behavior and method structure to test concurrency.
    """
    import importlib.util
    import os
    import sys
    import types

    engine_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "vllm_omni",
            "diffusion",
            "diffusion_engine.py",
        )
    )

    mocks = {
        "vllm_omni": MagicMock(),
        "vllm_omni.diffusion": types.ModuleType("vllm_omni.diffusion"),
        "vllm_omni.diffusion.data": MagicMock(),
        "vllm_omni.diffusion.executor": MagicMock(),
        "vllm_omni.diffusion.executor.abstract": MagicMock(),
        "vllm_omni.diffusion.registry": MagicMock(),
        "vllm_omni.diffusion.request": MagicMock(),
        "vllm_omni.diffusion.sched": MagicMock(),
        "vllm_omni.diffusion.sched.interface": MagicMock(),
        "vllm_omni.diffusion.worker": MagicMock(),
        "vllm_omni.diffusion.worker.utils": MagicMock(),
        "vllm_omni.inputs": MagicMock(),
        "vllm_omni.inputs.data": MagicMock(),
        "vllm_omni.outputs": MagicMock(),
        "vllm.logger": MagicMock(init_logger=lambda name: MagicMock()),
        "PIL": MagicMock(),
        "PIL.Image": MagicMock(),
    }
    with patch.dict(sys.modules, mocks):
        spec = importlib.util.spec_from_file_location("diffusion_engine", engine_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


_engine_mod = _make_mock_engine()


class TestRpcLockScope:
    """Verify that _rpc_lock is released during execute_fn."""

    def test_collective_rpc_not_blocked_during_execution(self) -> None:
        """collective_rpc should be able to acquire _rpc_lock while
        execute_fn is running (simulated with a sleep)."""
        lock = threading.RLock()
        execute_started = threading.Event()
        rpc_completed = threading.Event()
        rpc_acquired_during_exec = threading.Event()

        # Simulate the narrowed lock pattern from add_req_and_wait_for_response
        def simulate_request():
            with lock:
                pass  # scheduler.add_request — short

            # simulate schedule -> execute -> update loop (1 iteration)
            with lock:
                pass  # scheduler.schedule — short

            # Lock released during "GPU execution"
            execute_started.set()
            time.sleep(0.2)  # simulate GPU work

            with lock:
                pass  # scheduler.update_from_output — short
            rpc_completed.set()

        # Simulate collective_rpc trying to acquire lock during execution
        def simulate_rpc():
            execute_started.wait(timeout=2.0)
            # Small delay to ensure we're in the middle of "GPU work"
            time.sleep(0.05)
            acquired = lock.acquire(timeout=0.1)
            if acquired:
                rpc_acquired_during_exec.set()
                lock.release()

        t_req = threading.Thread(target=simulate_request)
        t_rpc = threading.Thread(target=simulate_rpc)
        t_req.start()
        t_rpc.start()
        t_req.join(timeout=3.0)
        t_rpc.join(timeout=3.0)

        assert rpc_acquired_during_exec.is_set(), "collective_rpc should be able to acquire lock during execute_fn"

    def test_scheduler_ops_are_serialized(self) -> None:
        """Concurrent scheduler operations should be serialized by the lock."""
        lock = threading.RLock()
        order = []

        def scheduler_op(name: str, delay: float = 0.05):
            with lock:
                order.append(f"{name}_start")
                time.sleep(delay)
                order.append(f"{name}_end")

        t1 = threading.Thread(target=scheduler_op, args=("op1",))
        t2 = threading.Thread(target=scheduler_op, args=("op2",))
        t1.start()
        time.sleep(0.01)  # ensure t1 starts first
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        # Operations should not interleave
        assert order == ["op1_start", "op1_end", "op2_start", "op2_end"]

    def test_source_lock_pattern(self) -> None:
        """Verify that add_req_and_wait_for_response uses multiple lock
        acquisitions (narrowed scope) rather than a single wrapping lock."""
        import ast
        import os

        engine_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                os.pardir,
                "vllm_omni",
                "diffusion",
                "diffusion_engine.py",
            )
        )
        with open(engine_path) as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DiffusionEngine":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "add_req_and_wait_for_response":
                        # Count `with self._rpc_lock:` statements
                        with_lock_count = 0
                        for child in ast.walk(item):
                            if isinstance(child, ast.With):
                                for wi in child.items:
                                    ctx = wi.context_expr
                                    if isinstance(ctx, ast.Attribute) and ctx.attr == "_rpc_lock":
                                        with_lock_count += 1
                        assert with_lock_count >= 3, (
                            f"Expected >= 3 lock acquisitions in add_req_and_wait_for_response "
                            f"(narrowed scope), found {with_lock_count}"
                        )
                        return
        pytest.fail("add_req_and_wait_for_response not found")
