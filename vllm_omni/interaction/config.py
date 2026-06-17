# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.interaction.prompts import SYSTEM_PROMPTS


@dataclass
class SamplingConfig:
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    presence_penalty: float = 1.5


@dataclass
class InteractionConfig:
    main_backend_url: str = "http://127.0.0.1:8061/v1"
    main_model: str = "JoyAI-VL-Interaction-Preview"
    api_key: str = "EMPTY"

    persona: str = "default"
    frame_seconds: float = 1.0
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    force_silence_before_query: bool = True

    enable_memory: bool = True

    summarizer_backend_url: str | None = None
    summarizer_model: str | None = None

    # raw-vision window; stays under --limit-mm-per-prompt, evicted chunks become summaries
    chunk_frames: int = 16

    mid_term_key_frames: int = 8

    long_term_every_n_chunks: int = 5
    mid_term_max_tokens: int = 1024
    long_term_max_tokens: int = 1024
    keep_qa_history: bool = True

    enable_delegation: bool = True

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
