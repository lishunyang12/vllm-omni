# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, Protocol

from .events import (
    AppendToEngine,
    CancelFence,
    CloseSessionResources,
    DomainEvent,
    DuplexEffect,
    EmitProtocolEvent,
    EngineFailed,
    RebuildStage0Context,
    ReserveResponse,
    ResetStage1,
    SessionCloseRequested,
)
from .ports import DuplexEnginePort, DuplexEventSink
from .state import DuplexSessionPhase, DuplexSessionState, reduce_event


class _DuplexState(Protocol):
    """Minimal state contract the runtime needs from any reducer's state."""

    session_phase: DuplexSessionPhase


Reducer = Callable[
    [Any, DomainEvent],
    tuple[Any, tuple[DuplexEffect, ...]],
]


class DuplexShutdownError(RuntimeError):
    def __init__(self, tasks: tuple[asyncio.Task[None], ...]) -> None:
        self.tasks = tasks
        task_names = ", ".join(task.get_name() for task in tasks)
        super().__init__(f"duplex shutdown timed out with live task(s): {task_names}")


class DuplexRuntime:
    """Drive one duplex session exclusively through typed events and effects."""

    def __init__(
        self,
        session_id: str,
        engine: DuplexEnginePort,
        sink: DuplexEventSink,
        *,
        capabilities: frozenset[str] = frozenset(),
        shutdown_timeout: float = 5.0,
        reduce: Reducer = reduce_event,
        initial_state: Any | None = None,
    ) -> None:
        if shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be positive")
        # The reducer and its state are pluggable: the default is the turn-based
        # `reduce_event`/`DuplexSessionState`; the continuous full-duplex path
        # injects `reduce_continuous_event`/`ContinuousDuplexState`. Both expose
        # `session_phase` and return the same effect types, so the driver below
        # is reducer-agnostic.
        self._reduce = reduce
        self.state = (
            initial_state
            if initial_state is not None
            else DuplexSessionState.open(session_id, capabilities=capabilities)
        )
        self._engine = engine
        self._sink = sink
        self._dispatch_lock = asyncio.Lock()
        self._failure_lock = asyncio.Lock()
        self._failure_started = False
        self._shutdown_timeout = shutdown_timeout
        self._session_closed = asyncio.Event()
        self._engine_effects: asyncio.Queue[DuplexEffect | None] = asyncio.Queue()
        self._sink_effects: asyncio.Queue[EmitProtocolEvent | None] = asyncio.Queue()
        self._resource_tasks: set[asyncio.Task[None]] = set()

    async def run(self, inputs: AsyncIterator[DomainEvent]) -> None:
        engine_worker = self._create_resource_task(self._run_engine_effects())
        engine_worker.set_name("duplex-engine-effects")
        sink_worker = self._create_resource_task(self._run_sink_effects())
        sink_worker.set_name("duplex-sink-effects")
        output_task = self._create_resource_task(self._consume_engine_events())
        output_task.set_name("duplex-engine-events")
        input_task = self._create_resource_task(self._consume_inputs(inputs))
        input_task.set_name("duplex-input")
        session_closed_task = self._create_resource_task(self._wait_for_session_close())
        session_closed_task.set_name("duplex-session-closed")
        try:
            done, _ = await asyncio.wait(
                (input_task, session_closed_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if input_task in done:
                await input_task
                if self.state.session_phase is not DuplexSessionPhase.CLOSED:
                    await self.dispatch(SessionCloseRequested(reason="input exhausted"))
            else:
                await self._cancel_owned_task(input_task)
        finally:
            if self.state.session_phase is not DuplexSessionPhase.CLOSED:
                await self.dispatch(SessionCloseRequested(reason="input exhausted"))
            await self._cancel_owned_task(input_task)
            await self._cancel_owned_task(session_closed_task)
            await self._shutdown(
                output_task=output_task,
                engine_worker=engine_worker,
                sink_worker=sink_worker,
            )
            self._resource_tasks.difference_update(task for task in (input_task, session_closed_task) if task.done())

    async def dispatch(self, event: DomainEvent) -> None:
        async with self._dispatch_lock:
            next_state, effects = self._reduce(self.state, event)
            self.state = next_state
            for effect in effects:
                if isinstance(effect, EmitProtocolEvent):
                    self._sink_effects.put_nowait(effect)
                else:
                    self._engine_effects.put_nowait(effect)
            if next_state.session_phase is DuplexSessionPhase.CLOSED:
                self._session_closed.set()

    async def _consume_inputs(self, inputs: AsyncIterator[DomainEvent]) -> None:
        async for event in inputs:
            await self.dispatch(event)
            if self.state.session_phase is DuplexSessionPhase.CLOSED:
                return

    async def _wait_for_session_close(self) -> None:
        await self._session_closed.wait()

    async def _consume_engine_events(self) -> None:
        try:
            async for event in self._engine.events():
                await self.dispatch(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_once(exc)

    async def _run_engine_effects(self) -> None:
        while True:
            effect = await self._engine_effects.get()
            try:
                if effect is None:
                    return
                if self._failure_started and isinstance(
                    effect,
                    (ReserveResponse, AppendToEngine, RebuildStage0Context),
                ):
                    continue
                try:
                    await self._execute_engine_effect(effect)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._fail_once(exc)
            finally:
                self._engine_effects.task_done()

    async def _run_sink_effects(self) -> None:
        while True:
            effect = await self._sink_effects.get()
            try:
                if effect is None:
                    return
                try:
                    await self._sink.emit(effect)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._fail_once(exc)
            finally:
                self._sink_effects.task_done()

    async def _fail_once(self, exc: Exception) -> None:
        async with self._failure_lock:
            if self._failure_started:
                return
            self._failure_started = True
            await self.dispatch(
                EngineFailed(
                    message=str(exc),
                    fence=self.state.fence,
                )
            )

    async def _execute_engine_effect(self, effect: DuplexEffect) -> None:
        if isinstance(effect, ReserveResponse):
            await self._engine.reserve(effect)
        elif isinstance(effect, AppendToEngine):
            await self._engine.append(effect)
        elif isinstance(effect, CancelFence):
            await self._engine.cancel(effect.fence)
        elif isinstance(effect, ResetStage1):
            await self._engine.reset(effect)
        elif isinstance(effect, RebuildStage0Context):
            await self._engine.rebuild(effect)
        elif isinstance(effect, CloseSessionResources):
            await self._engine.close(effect.fence)
        else:  # pragma: no cover - split lanes reject protocol effects
            raise TypeError(f"unsupported engine effect: {type(effect).__name__}")

    async def _shutdown(
        self,
        *,
        output_task: asyncio.Task[None],
        engine_worker: asyncio.Task[None],
        sink_worker: asyncio.Task[None],
    ) -> None:
        # Engine cleanup cannot be held hostage by a blocked protocol consumer.
        await self._drain_lane(self._engine_effects, engine_worker)
        await self._cancel_owned_task(output_task)

        await self._drain_lane(self._sink_effects, sink_worker)
        # Sink failure can enqueue fail-once engine cleanup after the first
        # engine drain, so give that lane one final independent bounded drain.
        await self._drain_lane(self._engine_effects, engine_worker)

        await self._stop_worker(self._engine_effects, engine_worker)
        await self._stop_worker(self._sink_effects, sink_worker)
        self._resource_tasks.difference_update(
            task for task in (output_task, engine_worker, sink_worker) if task.done()
        )
        live_tasks = tuple(task for task in self._resource_tasks if not task.done())
        if live_tasks:
            raise DuplexShutdownError(live_tasks)

    async def _drain_lane(
        self,
        queue: asyncio.Queue[Any],
        worker: asyncio.Task[None],
    ) -> None:
        if worker.done():
            return
        try:
            await asyncio.wait_for(
                queue.join(),
                timeout=self._shutdown_timeout,
            )
        except TimeoutError:
            await self._cancel_owned_task(worker)

    async def _stop_worker(
        self,
        queue: asyncio.Queue[Any],
        worker: asyncio.Task[None],
    ) -> None:
        if worker.done():
            return
        queue.put_nowait(None)
        done, _ = await asyncio.wait(
            (worker,),
            timeout=self._shutdown_timeout,
        )
        if worker not in done:
            await self._cancel_owned_task(worker)

    async def _cancel_owned_task(self, task: asyncio.Task[None]) -> None:
        if not task.done():
            task.cancel()
            await asyncio.wait(
                (task,),
                timeout=self._shutdown_timeout,
            )
        if task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                task.result()

    def _create_resource_task(
        self,
        coroutine: Coroutine[Any, Any, None],
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)
        self._resource_tasks.add(task)
        task.add_done_callback(self._resource_tasks.discard)
        return task


__all__ = ["DuplexRuntime", "DuplexShutdownError"]
