# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class DelegationResult:
    task_id: str
    status: str
    digest: str = ""

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"


class DelegationBridge(Protocol):
    async def submit(self, question: str, note: str, frames: list[tuple[str, str]]) -> str: ...

    async def poll(self, task_id: str) -> DelegationResult: ...


class StubDelegationBridge:
    def __init__(self, ready_after_ticks: int = 2) -> None:
        self._ready_after = max(1, ready_after_ticks)
        self._tasks: dict[str, dict[str, Any]] = {}
        self._counter = 0

    async def submit(self, question: str, note: str, frames: list[tuple[str, str]]) -> str:
        self._counter += 1
        task_id = f"deleg-{self._counter}"
        self._tasks[task_id] = {"question": question, "polls": 0}
        return task_id

    async def poll(self, task_id: str) -> DelegationResult:
        task = self._tasks.get(task_id)
        if task is None:
            return DelegationResult(task_id, "error", "unknown task")
        task["polls"] += 1
        if task["polls"] < self._ready_after:
            return DelegationResult(task_id, "pending")
        question = task["question"]
        return DelegationResult(
            task_id,
            "ready",
            digest=f"(background result for: {question}) — stub digest; wire a real brain here.",
        )
