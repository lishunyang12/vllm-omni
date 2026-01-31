# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from .config import ProfilerConfig

logger = init_logger(__name__)


class ProfilerBase(ABC):
    """Abstract base class for profilers.

    Defines the common interface used by GPU workers and engines.
    """

    @abstractmethod
    def start(self, output_prefix: str, config: "ProfilerConfig") -> str:
        """Start profiling.

        Args:
            output_prefix: Base path (without rank or extension).
                e.g. "/tmp/profiles/stage_0_1706745600"
            config: Profiler configuration.

        Returns:
            Full path of the output file this rank will write.
        """
        pass

    @abstractmethod
    def stop(self) -> dict[str, Any] | None:
        """Stop profiling and finalize/output the results.

        Returns:
            Dict with result paths and stats, or None if not active.
        """
        pass

    @abstractmethod
    def get_step_context(self):
        """Returns a context manager that advances one profiling step.

        Should be a no-op (nullcontext) when profiler is not active.
        """
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Return True if profiling is currently running."""
        pass

    @classmethod
    def _get_rank(cls) -> int:
        """Get the rank from environment variable."""
        import os

        return int(os.getenv("RANK", "0"))
