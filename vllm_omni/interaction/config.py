# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the interaction serving layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.interaction.prompts import SYSTEM_PROMPTS


@dataclass
class SamplingConfig:
    """Sampling parameters forwarded to the main VLM each tick."""

    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    presence_penalty: float = 1.5


@dataclass
class InteractionConfig:
    """Everything the interaction server needs to run.

    The defaults mirror the JoyAI-VL-Interaction reference deployment and work
    out of the box against a single ``vllm serve`` of the model.
    """

    # --- main model backend (the interaction VLM) --------------------------- #
    main_backend_url: str = "http://127.0.0.1:8061/v1"
    main_model: str = "JoyAI-VL-Interaction-Preview"
    api_key: str = "EMPTY"

    # --- per-tick behaviour ------------------------------------------------- #
    persona: str = "default"  # key into prompts.SYSTEM_PROMPTS
    frame_seconds: float = 1.0
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    #: Skip the model call and stay silent until the first user query arrives.
    #: This "arms" proactive monitoring with an instruction (e.g. "alert me if...").
    force_silence_before_query: bool = True

    # --- three-tier memory -------------------------------------------------- #
    enable_memory: bool = True
    #: Backend for the summarizer model.  ``None`` -> reuse the main backend.
    summarizer_backend_url: str | None = None
    summarizer_model: str | None = None
    #: Frames per working chunk; on overflow the chunk is summarized + evicted.
    chunk_frames: int = 200
    #: Key frames sampled from a chunk to build its mid-term summary.
    mid_term_key_frames: int = 8
    #: Compress accumulated mid-term summaries into long-term every N chunks.
    long_term_every_n_chunks: int = 5
    mid_term_max_tokens: int = 1024
    long_term_max_tokens: int = 1024
    keep_qa_history: bool = True

    # --- delegation --------------------------------------------------------- #
    enable_delegation: bool = True

    # --- session lifecycle -------------------------------------------------- #
    session_timeout_seconds: float = 3600.0
    request_timeout_seconds: float = 300.0

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPTS.get(self.persona, SYSTEM_PROMPTS["default"])

    @property
    def resolved_summarizer_url(self) -> str:
        return self.summarizer_backend_url or self.main_backend_url

    @property
    def resolved_summarizer_model(self) -> str:
        return self.summarizer_model or self.main_model
