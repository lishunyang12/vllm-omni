from __future__ import annotations

import base64
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vllm_omni.experimental.fullduplex.minicpmo45.policy import MiniCPMO45DuplexPolicy

_MINICPMO45_SPECIAL_TOKEN_FIELDS = MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS
_MINICPMO45_OPTIONAL_TOKEN_FIELDS = MiniCPMO45DuplexPolicy.OPTIONAL_TOKEN_FIELDS


@dataclass
class _MiniCPMO45Stage0SessionState:
    session_id: str
    session_config: dict[str, Any] = field(default_factory=dict)
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    audio_chunk_idx: int = 0
    context_embeds: list[Any] = field(default_factory=list)
    context_token_ids: list[int] = field(default_factory=list)
    system_context_len: int = 0
    pending_logits: Any | None = None
    generated_ids: list[int] = field(default_factory=list)
    current_turn_ended: bool = True
    break_requested: bool = False
    closed: bool = False
    runner_context_len: int = 0
    last_forward_metadata: dict[str, Any] = field(default_factory=dict)
    prepared_seq: int | None = None
    prepared_inputs_embeds: Any | None = None
    prepared_input_token_ids: list[int] = field(default_factory=list)
    prepared_result: dict[str, Any] = field(default_factory=dict)
    audio_past_key_values: Any | None = None
    pending_terminator_token: int | None = None
    last_terminator_token: int | None = None


class MiniCPMO45Stage0DuplexRuntime:
    """MiniCPM-o 4.5 native duplex runtime for the loaded LLM/audio stage."""

    runtime_impl = "vllm_omni_minicpmo45_stage0_experimental_worker_runtime"
    owned_runtime = False
    stage_role = "llm"
    supports_multiple_native_duplex_sessions = True

    def __init__(self, stage_model: Any, *, model_path: str | None = None, device: str = "cuda") -> None:
        self.stage_model = stage_model
        self.model_path = model_path
        self.device = device
        self.session_config: dict[str, Any] = {}
        self.sessions: dict[str, _MiniCPMO45Stage0SessionState] = {}
        self.thinker = getattr(stage_model, "thinker", None) or getattr(stage_model, "model", None) or stage_model
        self.processor = (
            getattr(stage_model, "processor", None)
            or getattr(self.thinker, "processor", None)
            or self._load_processor_from_path(model_path)
        )
        self.tokenizer = (
            getattr(self.processor, "tokenizer", None)
            if self.processor is not None
            else getattr(stage_model, "tokenizer", None)
        )
        self._init_token_ids()

    @classmethod
    def can_wrap(cls, stage_model: Any) -> bool:
        stage = getattr(stage_model, "model_stage", None)
        return stage in {"llm", "thinker"} or hasattr(stage_model, "thinker")

    def open_duplex_session(self, **kwargs: Any) -> dict[str, Any]:
        session_id = str(kwargs.get("session_id") or "")
        if not session_id:
            raise ValueError("MiniCPM-o stage0 duplex session_id is required")
        if not self._uses_runner_context_forward():
            raise RuntimeError(
                "MiniCPM-o stage0 native duplex requires "
                "duplex_forward_with_runner_context on the loaded stage model "
                "to reuse scheduler attention metadata/KV."
            )
        if not self._has_runner_context_contract():
            raise RuntimeError(
                "MiniCPM-o stage0 native duplex requires a runner-context contract "
                "hook injected by the vLLM model runner; model-local forward hooks "
                "are not accepted."
            )
        self._require_special_token_ids()
        session_config = kwargs.get("session_config")
        self.session_config = dict(session_config) if isinstance(session_config, dict) else {}
        state = _MiniCPMO45Stage0SessionState(session_id=session_id, session_config=dict(self.session_config))
        self.sessions[session_id] = state
        self._configure_streaming_processor()
        self._prepare_session_context(state, self.session_config)
        return {
            "supported": True,
            "stage_role": self.stage_role,
            "runtime_impl": self.runtime_impl,
            "owned_runtime": self.owned_runtime,
            "stage_runtime_ready": self._stage_runtime_ready(),
        }

    def append_duplex_input(self, **kwargs: Any) -> dict[str, Any]:
        session_id = str(kwargs.get("session_id") or "")
        mode = kwargs.get("mode")
        payload = kwargs.get("payload")
        if mode != MiniCPMO45DuplexPolicy.INPUT_AUDIO_MODE:
            raise ValueError(f"MiniCPM-o stage0 duplex expects append_audio_chunk, got {mode!r}")
        native = self._append_with_loaded_backend(
            payload,
            session_id=session_id,
            force_listen=bool(isinstance(payload, dict) and payload.get("force_listen", False)),
        )
        native.setdefault("stage_role", self.stage_role)
        native.setdefault("runtime_impl", self.runtime_impl)
        native.setdefault("owned_runtime", self.owned_runtime)
        state = self.sessions.get(session_id)
        forward_metadata = state.last_forward_metadata if state is not None else {}
        uses_model_runner_scheduler = bool(forward_metadata.get("uses_model_runner_scheduler", False))
        runner_kv_backed = bool(forward_metadata.get("runner_kv_backed", False))
        native.setdefault("uses_model_runner_scheduler", uses_model_runner_scheduler)
        native.setdefault("runner_kv_backed", runner_kv_backed)
        kv_cache_length = forward_metadata.get("kv_cache_length")
        if isinstance(kv_cache_length, int):
            native.setdefault("kv_cache_length", kv_cache_length)
        native.setdefault("experimental_worker_control_rpc", True)
        native.setdefault("per_step_tensor_handoff", False)
        native.setdefault("runner_local_payload_ref", False)
        native.setdefault("experimental_eager_decoder", False)
        if native.get("is_listen") is False:
            handoff = self._extract_tts_handoff(native)
            if handoff is not None:
                native["requires_stage_handoff"] = True
                native["stage_handoff"] = {
                    "target_stage_role": "tts",
                    "mode": MiniCPMO45DuplexPolicy.STAGE_HANDOFF_MODE,
                    "payload": handoff,
                }
                native["runner_local_payload_ref"] = True
                self._strip_tts_handoff_fields(native)
        return native

    def signal_duplex_turn(self, **kwargs: Any) -> dict[str, Any]:
        session_id = str(kwargs.get("session_id") or "")
        event = kwargs.get("event")
        if event in {"barge_in", "input.cancel", "response.cancel"}:
            self.set_break(session_id=session_id)
        if event == "conversation.item.truncate":
            payload = kwargs.get("payload")
            history = payload.get("history") if isinstance(payload, dict) else None
            state = self.sessions.get(session_id)
            if state is not None and isinstance(history, list):
                self._rebuild_context_from_history(state, history)
        return {
            "supported": True,
            "stage_role": self.stage_role,
            "event": event,
        }

    def close_duplex_session(self, **kwargs: Any) -> dict[str, Any]:
        session_id = str(kwargs.get("session_id") or "")
        self.stop(session_id=session_id)
        return {
            "supported": True,
            "stage_role": self.stage_role,
            "reason": kwargs.get("reason"),
        }

    def stop(self, *, session_id: str | None = None) -> None:
        if session_id:
            state = self.sessions.pop(session_id, None)
            if state is not None:
                state.closed = True
        else:
            self.sessions.clear()

    def cleanup(self) -> None:
        self.sessions.clear()

    def set_break(self, *, session_id: str | None = None) -> None:
        if session_id and session_id in self.sessions:
            self.sessions[session_id].break_requested = True
        elif not session_id:
            for state in self.sessions.values():
                state.break_requested = True

    def _append_with_loaded_backend(
        self,
        payload: Any,
        *,
        session_id: str,
        force_listen: bool,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("append_audio_chunk payload must be a dict")
        state = self.sessions.get(session_id)
        if state is None:
            raise KeyError(f"unknown MiniCPM-o stage0 duplex session: {session_id}")
        audio_waveform = self._decode_audio_payload(payload)
        prefill = self._stage_prefill(state, audio_waveform)
        if prefill.get("success") is False:
            return prefill
        return self._stage_generate(state, force_listen=force_listen)

    def _stage_runtime_ready(self) -> bool:
        return self.processor is not None and self.tokenizer is not None and self.thinker is not None

    def _configure_streaming_processor(self) -> None:
        if self.processor is None:
            return
        set_streaming_mode = getattr(self.processor, "set_streaming_mode", None)
        if callable(set_streaming_mode):
            set_streaming_mode(
                mode="exact",
                chunk_ms=int(self._stage_param("chunk_ms", 1000)),
                first_chunk_ms=int(self._stage_param("first_chunk_ms", 1035)),
                cnn_redundancy_ms=int(self._stage_param("cnn_redundancy_ms", 20)),
                enable_sliding_window=True,
                slide_trigger_seconds=30.0,
                slide_stride_seconds=10.0,
            )
            # Match official init_streaming_processor: reset the streaming mel-processor
            # buffers at session init (modeling_minicpmo_unified.py:207).
            reset_streaming = getattr(self.processor, "reset_streaming", None)
            if callable(reset_streaming):
                reset_streaming()
            return
        configure_streaming = getattr(self.processor, "configure_streaming", None)
        if callable(configure_streaming):
            configure_streaming(
                chunk_ms=int(self._stage_param("chunk_ms", 1000)),
                enable_sliding_window=True,
                slide_trigger_seconds=30.0,
                slide_stride_seconds=10.0,
            )

    def _prepare_session_context(
        self,
        state: _MiniCPMO45Stage0SessionState,
        session_config: dict[str, Any],
    ) -> None:
        if not self._stage_runtime_ready():
            return
        self._require_special_token_ids()
        ref_audio = self._decode_ref_audio_from_session_config(session_config)
        # Matches MiniCPMODuplex.prepare() in the released checkpoint's
        # modeling_minicpmo.py: the <|audio_start|>/<|audio_end|> markers are
        # only present when reference audio is embedded between them. The
        # template is shared with the serving adapter so the first-append
        # scheduler reserve can count these tokens exactly.
        prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
            session_config.get("instructions"),
            ref_audio is not None,
        )
        for token_id in self._encode_text(prefix):
            state.context_embeds.append(self._embed_token(token_id))
            state.context_token_ids.append(token_id)
        if ref_audio is not None:
            ref_audio_embeds = self._stage_ref_audio_embeddings(ref_audio, state=state)
            if ref_audio_embeds is not None:
                ref_audio_embeds = self._as_2d_tensor(ref_audio_embeds)
                state.context_embeds.append(ref_audio_embeds)
                state.context_token_ids.extend([self.unit_token_id] * int(ref_audio_embeds.shape[0]))
        for token_id in self._encode_text(suffix):
            state.context_embeds.append(self._embed_token(token_id))
            state.context_token_ids.append(token_id)
        state.system_context_len = len(state.context_embeds)

    def _rebuild_context_from_history(self, state: _MiniCPMO45Stage0SessionState, history: list[object]) -> None:
        state.context_embeds.clear()
        state.context_token_ids.clear()
        state.pending_logits = None
        state.generated_ids.clear()
        state.current_turn_ended = True
        state.prepared_seq = None
        state.prepared_inputs_embeds = None
        state.prepared_input_token_ids.clear()
        state.prepared_result.clear()
        state.runner_context_len = 0
        state.last_forward_metadata.clear()
        state.audio_past_key_values = None
        state.audio_buffer = np.zeros(0, dtype=np.float32)
        state.audio_chunk_idx = 0
        state.pending_terminator_token = None
        state.last_terminator_token = None
        self._prepare_session_context(state, state.session_config)
        for message in history:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in {"system", "user", "assistant"}:
                continue
            text = self._history_message_text(message)
            if not text:
                continue
            for token_id in self._encode_text(f"<|im_start|>{role}\n{text}<|im_end|>"):
                state.context_embeds.append(self._embed_token(token_id))
                state.context_token_ids.append(token_id)
        self._enforce_context_window(state)

    @staticmethod
    def _history_message_text(message: dict[str, object]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        chunks: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in ("text", "transcript"):
                value = part.get(key)
                if isinstance(value, str) and value:
                    chunks.append(value)
        return "".join(chunks).strip()

    def _stage_prefill(
        self,
        state: _MiniCPMO45Stage0SessionState,
        audio_waveform: Any,
    ) -> dict[str, Any]:
        start_time = time.time()
        if state.closed:
            return self._stage_prefill_result(False, start_time, "session closed")
        if audio_waveform is None or len(audio_waveform) == 0:
            return self._stage_prefill_result(False, start_time, "empty audio")
        state.audio_buffer = np.concatenate([state.audio_buffer, np.asarray(audio_waveform, dtype=np.float32)])
        chunk_size = self._streaming_chunk_size()
        self._pad_first_audio_chunk_if_needed(state)
        if len(state.audio_buffer) < chunk_size:
            return self._stage_prefill_result(
                False,
                start_time,
                f"audio not enough: need {chunk_size} samples, only {len(state.audio_buffer)}",
            )

        audio_chunk = state.audio_buffer[:chunk_size]
        batch_feature = self._process_streaming_audio(audio_chunk, state.audio_chunk_idx)
        for name, value in (
            ("chunk_idx", state.audio_chunk_idx),
            ("use_extra_context", True),
            ("prefix_extra_frames", 0 if state.audio_chunk_idx == 0 else 2),
            ("suffix_extra_frames", 2),
        ):
            try:
                setattr(batch_feature, name, value)
            except Exception:
                pass
        audio_embeds = self._stage_audio_embeddings(batch_feature, state=state)
        if audio_embeds is None:
            return self._stage_prefill_result(False, start_time, "streaming audio embedding returned empty")
        self._require_special_token_ids()
        context_len_before_forward = len(state.context_embeds)
        pending_logits_before_forward = state.pending_logits
        try:
            state.context_embeds.append(self._embed_token(self.unit_token_id))
            state.context_embeds.append(audio_embeds)
            state.pending_logits, _ = self._forward_context(state)
        except Exception:
            del state.context_embeds[context_len_before_forward:]
            state.pending_logits = pending_logits_before_forward
            raise
        state.audio_buffer = state.audio_buffer[self._consumed_audio_samples(state.audio_chunk_idx, chunk_size) :]
        state.audio_chunk_idx += 1
        self._enforce_context_window(state)
        return self._stage_prefill_result(True, start_time)

    def _stage_prefill_embeddings_only(
        self,
        state: _MiniCPMO45Stage0SessionState,
        audio_waveform: Any,
        *,
        seq: int | None = None,
        final: bool = False,
        new_speech: bool = False,
        new_user_turn: bool = False,
        new_user_turn_prefix_variant: object = None,
    ) -> dict[str, Any]:
        """Build scheduler-owned Stage0 input embeddings for one audio append.

        Unlike the legacy worker-control path, this method never calls an eager
        model forward. The normal vLLM runner consumes the returned embeddings
        and owns attention metadata, block tables, KV cache, and sampling.
        """
        start_time = time.time()
        if seq is not None and state.prepared_seq == seq and state.prepared_inputs_embeds is not None:
            result = dict(state.prepared_result)
            result["inputs_embeds"] = state.prepared_inputs_embeds
            result["input_token_ids"] = list(state.prepared_input_token_ids)
            return result
        self._require_special_token_ids()
        if state.closed:
            return self._stage_prefill_result(False, start_time, "session closed")
        if audio_waveform is None or len(audio_waveform) == 0:
            return self._stage_prefill_result(False, start_time, "empty audio")
        prefix_new_user_turn = bool(new_user_turn)
        if prefix_new_user_turn:
            state.current_turn_ended = True
        state.audio_buffer = np.concatenate([state.audio_buffer, np.asarray(audio_waveform, dtype=np.float32)])
        chunk_size = self._streaming_chunk_size()
        self._pad_first_audio_chunk_if_needed(state)
        if len(state.audio_buffer) < chunk_size:
            return self._stage_prefill_result(
                False,
                start_time,
                f"audio not enough: need {chunk_size} samples, only {len(state.audio_buffer)}",
            )

        embed_parts: list[Any] = []
        token_ids: list[int] = []
        new_user_turn_prefix_ids = (
            list(self._encode_text(MiniCPMO45DuplexPolicy.new_user_turn_prefix_text(new_user_turn_prefix_variant)))
            if prefix_new_user_turn
            else []
        )
        if state.audio_chunk_idx == 0 and state.context_embeds:
            embed_parts.extend(state.context_embeds)
            token_ids.extend(state.context_token_ids)

        # Consume every complete chunk in the buffer so the appended span and
        # the scheduler's slot reservation for this append agree exactly. A
        # final append may zero-pad a real residual chunk, but it must not add
        # a whole silence unit after all input was already consumed: official
        # duplex generation runs once per microphone unit and client commit is
        # not an additional model decision.
        units_built = 0
        while True:
            if len(state.audio_buffer) < chunk_size:
                if not final or len(state.audio_buffer) == 0:
                    break
                pad = np.zeros(chunk_size - len(state.audio_buffer), dtype=np.float32)
                state.audio_buffer = np.concatenate([state.audio_buffer, pad])
            audio_chunk = state.audio_buffer[:chunk_size]
            batch_feature = self._process_streaming_audio(audio_chunk, state.audio_chunk_idx)
            for name, value in (
                ("chunk_idx", state.audio_chunk_idx),
                ("use_extra_context", True),
                ("prefix_extra_frames", 0 if state.audio_chunk_idx == 0 else 2),
                ("suffix_extra_frames", 2),
            ):
                with suppress(Exception):
                    setattr(batch_feature, name, value)
            audio_embeds = self._stage_audio_embeddings(batch_feature, state=state)
            if audio_embeds is None:
                if units_built == 0:
                    return self._stage_prefill_result(False, start_time, "streaming audio embedding returned empty")
                break
            if state.audio_chunk_idx > 0:
                # Official duplex closes every unit (finalize_unit feeds the
                # sampled terminator + </unit>) before the next <unit> opens.
                # The scheduler session update discards the previous segment's
                # sampled terminator token, so it is re-injected here ahead of
                # the closure; the model's listen/speak policy depends on
                # seeing its own past decisions in context.
                pending_terminator = state.pending_terminator_token
                if pending_terminator is not None and units_built == 0:
                    state.pending_terminator_token = None
                    embed_parts.append(self._embed_token(pending_terminator))
                    token_ids.append(int(pending_terminator))
                embed_parts.append(self._embed_token(self.unit_end_token_id))
                token_ids.append(self.unit_end_token_id)
            if prefix_new_user_turn and units_built == 0:
                for token_id in new_user_turn_prefix_ids:
                    embed_parts.append(self._embed_token(token_id))
                    token_ids.append(int(token_id))
            embed_parts.append(self._embed_token(self.unit_token_id))
            token_ids.append(self.unit_token_id)
            embed_parts.append(audio_embeds)
            token_ids.extend(
                [self._audio_embedding_placeholder_token_id()] * int(self._as_2d_tensor(audio_embeds).shape[0])
            )
            state.audio_buffer = state.audio_buffer[self._consumed_audio_samples(state.audio_chunk_idx, chunk_size) :]
            state.audio_chunk_idx += 1
            units_built += 1
            chunk_size = self._streaming_chunk_size()
        # Match official streaming_prefill: per chunk feed ONLY <unit>+audio. The assistant
        # turn is opened once at session init; re-emitting the turn-open prefix per chunk
        # re-opened the turn each chunk -> degenerate repetition. tts_bos/listen/turn_eos are
        # model-generated and tracked via current_turn_ended (mirrors streaming_generate).
        prompt_suffix_len = 0

        import torch

        inputs_embeds = torch.cat([self._as_2d_tensor(embed) for embed in embed_parts], dim=0)
        state.runner_context_len += int(inputs_embeds.shape[0])
        result = self._stage_prefill_result(True, start_time)
        result.update(
            {
                "inputs_embeds": inputs_embeds,
                "input_token_ids": token_ids,
                "special_token_ids": self._special_token_ids(),
                "num_input_tokens": int(inputs_embeds.shape[0]),
                "prompt_suffix_len": prompt_suffix_len,
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
                "runtime_impl": "scheduler_data_plane",
            }
        )
        if seq is not None:
            state.prepared_seq = seq
            state.prepared_inputs_embeds = inputs_embeds
            state.prepared_input_token_ids = list(token_ids)
            state.prepared_result = {k: v for k, v in result.items() if k not in {"inputs_embeds", "input_token_ids"}}
        return result

    @staticmethod
    def _stage_prefill_result(success: bool, start_time: float, reason: str = "") -> dict[str, Any]:
        return {
            "success": success,
            "prefill_success": success,
            "is_buffering": not success,
            "reason": reason,
            "cost_all": time.time() - start_time,
            "stage_runtime_ready": True,
        }

    def _stage_generate(
        self,
        state: _MiniCPMO45Stage0SessionState,
        *,
        force_listen: bool,
    ) -> dict[str, Any]:
        start_time = time.time()
        if state.pending_logits is None:
            return self._listen_result(state, start_time, reason="missing pending logits")

        logits = state.pending_logits
        state.pending_logits = None
        generated_ids: list[int] = []
        generated_hidden: list[Any] = []
        is_listen = False
        listen_reason = "model_listen"
        end_of_turn = False
        max_tokens = int(
            self._native_param(
                state,
                "max_new_speak_tokens_per_chunk",
                MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK,
            )
        )
        for step in range(max_tokens):
            if step == max_tokens - 1:
                state.context_embeds.append(self._embed_token(self.chunk_eos_token_id))
                state.generated_ids.append(self.chunk_eos_token_id)
                break
            if force_listen or state.break_requested:
                token_id = self.listen_token_id
                listen_reason = "forced_listen" if force_listen else "break_requested"
            else:
                token_id = self._decode_next_token(logits, state)
                if token_id == self.listen_token_id and not state.current_turn_ended:
                    token_id = self.tts_bos_token_id
            if token_id == self.listen_token_id:
                is_listen = True
                break
            if token_id in self.chunk_terminator_token_ids:
                end_of_turn = token_id in self.turn_terminator_token_ids
                state.context_embeds.append(self._embed_token(token_id))
                break
            state.current_turn_ended = False
            hidden, logits = self._append_token_and_forward(state, token_id)
            end_of_turn = token_id in self.turn_terminator_token_ids
            if end_of_turn:
                state.current_turn_ended = True
            if token_id not in self.chunk_speak_token_ids:
                if step != 0:
                    generated_ids.append(token_id)
                    generated_hidden.append(hidden)
                state.generated_ids.append(token_id)

        state.break_requested = False
        if is_listen or not generated_ids:
            return self._listen_result(
                state,
                start_time,
                reason=listen_reason if is_listen else "empty_speak",
            )
        state.context_embeds.append(self._embed_token(self.unit_end_token_id))
        self._enforce_context_window(state)
        text = self._decode_tokens(generated_ids)
        omni_payload = self._build_tts_omni_payload(generated_ids, generated_hidden)
        tts_handoff = {
            "type": MiniCPMO45DuplexPolicy.TTS_HANDOFF_TYPE,
            "omni_payload": omni_payload,
            "llm_output_text": [text],
            "meta": {
                "native_duplex_segment_text": text,
                "override_keys": [
                    "llm_output_text",
                    ["meta", "native_duplex_segment_text"],
                ],
            },
            "end_of_turn": end_of_turn,
            "session_id": state.session_id,
        }
        return {
            "success": True,
            "is_listen": False,
            "text": text,
            "end_of_turn": end_of_turn,
            "current_time": state.audio_chunk_idx,
            "stage_runtime_ready": True,
            "cost_llm": time.time() - start_time,
            "cost_all": time.time() - start_time,
            "n_tokens": len(generated_ids),
            "requires_stage_handoff": True,
            "stage_handoff": {
                "target_stage_role": "tts",
                "mode": MiniCPMO45DuplexPolicy.STAGE_HANDOFF_MODE,
                "payload": tts_handoff,
            },
        }

    def _enforce_context_window(self, state: _MiniCPMO45Stage0SessionState) -> None:
        max_tokens = int(
            self._native_param(
                state,
                "stage0_context_max_tokens",
                self._native_param(state, "basic_window_high_tokens", 8000),
            )
        )
        if max_tokens <= 0 or len(state.context_embeds) <= max_tokens:
            return

        system_len = max(0, min(int(getattr(state, "system_context_len", 0)), len(state.context_embeds)))
        previous_max = int(
            self._native_param(
                state,
                "stage0_context_previous_max_tokens",
                self._native_param(state, "context_previous_max_tokens", max_tokens - system_len),
            )
        )
        recent_budget = max(0, min(previous_max, max_tokens - system_len))
        prefix = state.context_embeds[:system_len]
        recent = state.context_embeds[-recent_budget:] if recent_budget > 0 else []
        state.context_embeds = [*prefix, *recent]

    def _listen_result(
        self,
        state: _MiniCPMO45Stage0SessionState,
        start_time: float,
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        return {
            "success": True,
            "is_listen": True,
            "model_listen": reason in {"", "model_listen"},
            "listen_source": reason or "model_listen",
            "text": "",
            "end_of_turn": False,
            "current_time": state.audio_chunk_idx,
            "stage_runtime_ready": True,
            "reason": reason,
            "cost_llm": time.time() - start_time,
            "cost_all": time.time() - start_time,
        }

    def _append_token_and_forward(self, state: _MiniCPMO45Stage0SessionState, token_id: int) -> tuple[Any, Any]:
        state.context_embeds.append(self._embed_token(token_id))
        logits, hidden_states = self._forward_context(state)
        return self._last_hidden(hidden_states), logits

    def _forward_context(self, state: _MiniCPMO45Stage0SessionState) -> tuple[Any, Any]:
        import torch

        runner_forward = getattr(self.stage_model, "duplex_forward_with_runner_context", None)
        if callable(runner_forward):
            if not self._has_runner_context_contract():
                raise RuntimeError(
                    "MiniCPM-o stage0 native duplex requires a runner-context contract "
                    "hook injected by the vLLM model runner."
                )
            raw_previous_context_len = max(0, int(state.runner_context_len))
            reset_kv = raw_previous_context_len == 0 or raw_previous_context_len > len(state.context_embeds)
            if reset_kv:
                previous_context_len = 0
            else:
                previous_context_len = raw_previous_context_len
            new_context_embeds = state.context_embeds[previous_context_len:]
            if not new_context_embeds:
                raise RuntimeError("MiniCPM-o stage0 runner forward has no new embeds to append")
            inputs_embeds = torch.cat([self._as_2d_tensor(embed) for embed in new_context_embeds], dim=0)
            with torch.inference_mode():
                output = runner_forward(
                    session_id=state.session_id,
                    inputs_embeds=inputs_embeds,
                    context_len=len(state.context_embeds),
                    previous_context_len=previous_context_len,
                    reset_kv=reset_kv,
                )
            if not isinstance(output, dict):
                raise TypeError("duplex_forward_with_runner_context must return a metadata dict")
            logits = output.get("logits")
            hidden_states = output.get("hidden_states")
            state.last_forward_metadata = {
                key: output[key]
                for key in (
                    "uses_model_runner_scheduler",
                    "runner_kv_backed",
                    "kv_cache_length",
                    "sampled_token_id",
                )
                if key in output
            }
            if (
                state.last_forward_metadata.get("uses_model_runner_scheduler") is not True
                or state.last_forward_metadata.get("runner_kv_backed") is not True
            ):
                raise RuntimeError(
                    "MiniCPM-o stage0 runner forward must be scheduler/KV-backed; "
                    "refusing to run an unbacked direct model forward."
                )
            if logits is None or hidden_states is None:
                raise ValueError("duplex_forward_with_runner_context returned incomplete logits/hidden_states")
            kv_cache_length = state.last_forward_metadata.get("kv_cache_length")
            if isinstance(kv_cache_length, int) and kv_cache_length >= 0:
                state.runner_context_len = kv_cache_length
            else:
                state.runner_context_len = len(state.context_embeds)
            return logits, hidden_states

        raise RuntimeError(
            "MiniCPM-o stage0 native duplex cannot run through vLLM forward "
            "without scheduler attention metadata/KV. Provide "
            "duplex_forward_with_runner_context on the loaded stage model."
        )

    def _uses_runner_context_forward(self) -> bool:
        return callable(getattr(self.stage_model, "duplex_forward_with_runner_context", None))

    def _has_runner_context_contract(self) -> bool:
        runner_forward = getattr(self.stage_model, "duplex_forward_with_runner_context", None)
        return bool(
            callable(runner_forward)
            and getattr(runner_forward, "uses_scheduler_metadata", False) is True
            and getattr(runner_forward, "uses_runner_kv_cache", False) is True
            and getattr(runner_forward, "vllm_omni_runner_context_contract", False) is True
        )

    @staticmethod
    def _last_hidden(hidden_states: Any) -> Any:
        if hasattr(hidden_states, "ndim"):
            if hidden_states.ndim == 3:
                return hidden_states[:, -1, :]
            if hidden_states.ndim == 2:
                return hidden_states[-1:, :]
        return hidden_states

    @staticmethod
    def _as_2d_tensor(value: Any) -> Any:
        if value.ndim == 1:
            return value.unsqueeze(0)
        if value.ndim == 3 and value.shape[0] == 1:
            return value.squeeze(0)
        return value

    def _embed_token(self, token_id: int) -> Any:
        import torch

        token = torch.tensor([int(token_id)], dtype=torch.long, device=self._model_device())
        embedder = self._token_embedder()
        embeds = embedder(token)
        return self._as_2d_tensor(embeds)

    def _token_embedder(self) -> Any:
        nested_embed = getattr(getattr(getattr(self.thinker, "llm", None), "model", None), "embed_tokens", None)
        if callable(nested_embed):
            return nested_embed
        for target in (self.thinker, self.stage_model):
            embedder = getattr(target, "get_input_embeddings", None)
            if callable(embedder):
                try:
                    embeddings = embedder()
                    if callable(embeddings):
                        return embeddings
                except TypeError:
                    return embedder
        raise AttributeError("MiniCPM-o stage0 model does not expose token embeddings")

    def _model_device(self) -> Any:
        try:
            return next(self.thinker.parameters()).device
        except Exception:
            pass
        try:
            return next(self.stage_model.parameters()).device
        except Exception:
            pass
        return self.device

    def _streaming_chunk_size(self) -> int:
        get_chunk = getattr(self.processor, "get_streaming_chunk_size", None)
        if callable(get_chunk):
            return int(get_chunk())
        return 16000

    def _sample_rate(self) -> int:
        return int(
            self._stage_param(
                "sample_rate",
                getattr(getattr(self.processor, "_streaming_mel_processor", None), "sample_rate", 16000),
            )
        )

    def _first_chunk_samples(self, default_chunk_size: int) -> int:
        if getattr(self.processor, "_streaming_mel_processor", None) is None:
            return default_chunk_size
        return int(self._stage_param("first_chunk_ms", 1035) * self._sample_rate() / 1000)

    def _pad_first_audio_chunk_if_needed(self, state: _MiniCPMO45Stage0SessionState) -> None:
        if state.audio_chunk_idx != 0 or len(state.audio_buffer) == 0:
            return
        first_chunk_samples = self._first_chunk_samples(self._streaming_chunk_size())
        if len(state.audio_buffer) >= first_chunk_samples:
            return
        padding = np.zeros(first_chunk_samples - len(state.audio_buffer), dtype=np.float32)
        state.audio_buffer = np.concatenate([padding, state.audio_buffer])

    def _stage_param(self, name: str, default: Any) -> Any:
        for target in (self.stage_model, self.thinker, getattr(self.thinker, "llm", None)):
            value = getattr(target, name, None)
            if value is not None:
                return value
            value = getattr(target, name.upper(), None)
            if value is not None:
                return value
        return default

    def _consumed_audio_samples(self, chunk_idx: int, default_chunk_size: int) -> int:
        if chunk_idx != 0:
            chunk_ms = int(self._stage_param("chunk_ms", 1000))
            return int(chunk_ms * self._sample_rate() / 1000)
        mel_processor = getattr(self.processor, "_streaming_mel_processor", None)
        get_config = getattr(mel_processor, "get_config", None)
        if callable(get_config):
            cfg = get_config()
            if isinstance(cfg, dict):
                consumed_ms = int(cfg.get("effective_first_chunk_ms", self._stage_param("first_chunk_ms", 1035)))
                return int(consumed_ms * self._sample_rate() / 1000)
        return default_chunk_size

    def _process_streaming_audio(self, audio_chunk: Any, chunk_idx: int) -> Any:
        process = getattr(self.processor, "process_audio_streaming", None)
        if callable(process):
            try:
                return process(audio_chunk, reset=False, return_batch_feature=True)
            except TypeError:
                return process(audio_chunk, chunk_idx=chunk_idx)
        return {"audio_features": audio_chunk, "audio_feature_lens": [[len(audio_chunk)]]}

    def _stage_audio_embeddings(
        self,
        batch_feature: Any,
        *,
        state: _MiniCPMO45Stage0SessionState | None = None,
    ) -> Any | None:
        if hasattr(batch_feature, "to"):
            batch_feature = batch_feature.to(self.device)
        self._ensure_dynamic_cache_compat()
        has_audio_cache = state is not None and hasattr(self.thinker, "audio_past_key_values")
        previous_audio_past_key_values = (
            getattr(self.thinker, "audio_past_key_values", None) if has_audio_cache else None
        )
        if has_audio_cache:
            self.thinker.audio_past_key_values = state.audio_past_key_values
        try:
            for target in (self.stage_model, self.thinker):
                get_streaming = getattr(target, "get_audio_embedding_streaming", None)
                if callable(get_streaming):
                    try:
                        result = self._cat_nested_tensors(
                            get_streaming(
                                batch_feature,
                                use_extra_context=True,
                                prefix_extra_frames=0 if int(getattr(batch_feature, "chunk_idx", 0)) == 0 else 2,
                                suffix_extra_frames=2,
                            )
                        )
                        if has_audio_cache:
                            state.audio_past_key_values = getattr(self.thinker, "audio_past_key_values", None)
                        return result
                    except TypeError:
                        result = self._cat_nested_tensors(get_streaming(batch_feature))
                        if has_audio_cache:
                            state.audio_past_key_values = getattr(self.thinker, "audio_past_key_values", None)
                        return result
                get_hidden = getattr(target, "get_audio_hidden_states", None)
                if callable(get_hidden):
                    result = self._cat_nested_tensors(get_hidden(batch_feature))
                    if has_audio_cache:
                        state.audio_past_key_values = getattr(self.thinker, "audio_past_key_values", None)
                    return result
            return None
        finally:
            if has_audio_cache:
                self.thinker.audio_past_key_values = previous_audio_past_key_values

    @staticmethod
    def _decode_ref_audio_from_session_config(session_config: dict[str, Any]) -> Any | None:
        try:
            from vllm_omni.experimental.fullduplex.engine.worker import decode_native_ref_audio_from_config

            return decode_native_ref_audio_from_config(session_config)
        except Exception:
            return None

    def _stage_ref_audio_embeddings(
        self,
        ref_audio: Any,
        *,
        state: _MiniCPMO45Stage0SessionState | None = None,
    ) -> Any | None:
        process_audio = getattr(self.processor, "process_audio", None)
        if callable(process_audio):
            batch_feature = process_audio([ref_audio])
            if hasattr(batch_feature, "to"):
                batch_feature = batch_feature.to(self.device)
            self._ensure_dynamic_cache_compat()
            for target in (self.stage_model, self.thinker):
                get_audio_embedding = getattr(target, "get_audio_embedding", None)
                if callable(get_audio_embedding):
                    try:
                        chunk_length = getattr(getattr(target, "config", None), "audio_chunk_length", None)
                        if chunk_length is not None:
                            return self._cat_nested_tensors(
                                get_audio_embedding(batch_feature, chunk_length=chunk_length)
                            )
                    except TypeError:
                        pass
                    return self._cat_nested_tensors(get_audio_embedding(batch_feature))
                # The split vLLM stage0 wrapper ports official
                # get_audio_embedding(chunk_length=...) as
                # get_audio_hidden_states (chunk_length comes from config).
                get_hidden = getattr(target, "get_audio_hidden_states", None)
                if callable(get_hidden):
                    return self._cat_nested_tensors(get_hidden(batch_feature))
        # The split vLLM stage0 wrapper may only expose the streaming encoder
        # path.  Use it as a fallback so the server-resolved reference audio is
        # still represented in the same system context location as official
        # MiniCPM-o prepare().
        batch_feature = self._process_streaming_audio(ref_audio, 0)
        for name, value in (
            ("chunk_idx", 0),
            ("use_extra_context", True),
            ("prefix_extra_frames", 0),
            ("suffix_extra_frames", 2),
        ):
            with suppress(Exception):
                setattr(batch_feature, name, value)
        return self._stage_audio_embeddings(batch_feature, state=state)

    @staticmethod
    def _ensure_dynamic_cache_compat() -> None:
        try:
            from transformers.cache_utils import DynamicCache
        except Exception:
            return
        if hasattr(DynamicCache, "get_usable_length"):
            return

        def get_usable_length(self, new_seq_length: int | None = None, layer_idx: int = 0) -> int:
            get_seq_length = getattr(self, "get_seq_length", None)
            if not callable(get_seq_length):
                return 0
            try:
                return int(get_seq_length(layer_idx))
            except TypeError:
                return int(get_seq_length())

        DynamicCache.get_usable_length = get_usable_length  # type: ignore[attr-defined]

    @staticmethod
    def _cat_nested_tensors(value: Any) -> Any | None:
        import torch

        tensors = []

        def collect(item: Any) -> None:
            if item is None:
                return
            if hasattr(item, "detach"):
                tensors.append(item)
                return
            if isinstance(item, dict):
                for child in item.values():
                    collect(child)
                return
            if isinstance(item, (list, tuple)):
                for child in item:
                    collect(child)

        collect(value)
        if not tensors:
            return None
        return torch.cat([tensor.reshape(-1, tensor.shape[-1]) for tensor in tensors], dim=0)

    def _encode_text(self, text: str) -> list[int]:
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            return list(encode(text, add_special_tokens=False))
        return []

    def _decode_tokens(self, token_ids: list[int]) -> str:
        decode = getattr(self.tokenizer, "decode", None)
        if callable(decode):
            return str(decode(token_ids, skip_special_tokens=True))
        return ""

    def _special_token_ids(self) -> dict[str, int]:
        return {
            name: value
            for name, value in {
                "unit_token_id": self.unit_token_id,
                "unit_end_token_id": self.unit_end_token_id,
                "listen_token_id": self.listen_token_id,
                "speak_token_id": self.speak_token_id,
                "tts_bos_token_id": self.tts_bos_token_id,
                "tts_eos_token_id": self.tts_eos_token_id,
                "tts_pad_token_id": self.tts_pad_token_id,
                "chunk_eos_token_id": self.chunk_eos_token_id,
                "chunk_tts_eos_token_id": self.chunk_tts_eos_token_id,
                "turn_eos_token_id": self.turn_eos_token_id,
            }.items()
            if isinstance(value, int) and value >= 0
        }

    @staticmethod
    def _build_tts_omni_payload(token_ids: list[int], hidden_states: list[Any]) -> Any:
        import torch

        from vllm_omni.data_entry_keys import serialize_payload

        hidden_tensors = [hidden for hidden in hidden_states if hasattr(hidden, "detach")]
        if not hidden_tensors:
            hidden = torch.empty((0, 0), dtype=torch.float32)
        else:
            hidden = torch.cat(
                [tensor.detach().float().reshape(-1, tensor.shape[-1]).cpu() for tensor in hidden_tensors],
                dim=0,
            ).contiguous()
        return serialize_payload(
            {
                "ids": {"output": list(token_ids)},
                "hidden_states": {"output": hidden},
            }
        )

    @staticmethod
    def _load_processor_from_path(model_path: str | None) -> Any | None:
        if not model_path:
            return None
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
        try:
            from processing_minicpmo import MiniCPMOProcessor

            return MiniCPMOProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            return None

    def _init_token_ids(self) -> None:
        if self.tokenizer is None:
            for field_name in _MINICPMO45_SPECIAL_TOKEN_FIELDS:
                setattr(self, field_name, -1)
            for field_name in _MINICPMO45_OPTIONAL_TOKEN_FIELDS:
                setattr(self, field_name, -1)
        else:
            for field_name, token in _MINICPMO45_SPECIAL_TOKEN_FIELDS.items():
                setattr(self, field_name, self._resolve_special_token_id(token))
            for field_name, token in _MINICPMO45_OPTIONAL_TOKEN_FIELDS.items():
                setattr(self, field_name, self._resolve_special_token_id(token))
        self.chunk_terminator_token_ids = [
            self.listen_token_id,
            self.chunk_eos_token_id,
            self.chunk_tts_eos_token_id,
        ]
        self.turn_terminator_token_ids = [self.turn_eos_token_id]
        self.chunk_speak_token_ids = [self.speak_token_id]
        bad_token_ids = getattr(self.tokenizer, "bad_token_ids", []) if self.tokenizer is not None else []
        self.forbidden_token_ids = self._valid_token_ids(
            [self.tts_pad_token_id, *list(bad_token_ids), self.chunk_eos_token_id]
        )

    def _resolve_special_token_id(self, token: str) -> int:
        if self.tokenizer is None:
            return -1

        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        candidate = None
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            value = convert(token)
            if isinstance(value, list):
                value = value[0] if len(value) == 1 else None
            with suppress(TypeError, ValueError):
                candidate = int(value)
        if candidate is not None and candidate >= 0 and candidate != unk_token_id:
            return candidate

        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            ids = list(encode(token, add_special_tokens=False))
            if len(ids) == 1:
                value = int(ids[0])
                if value >= 0 and value != unk_token_id:
                    return value
        return -1

    def _require_special_token_ids(self) -> None:
        missing = [
            token
            for field_name, token in _MINICPMO45_SPECIAL_TOKEN_FIELDS.items()
            if not isinstance(getattr(self, field_name, None), int) or getattr(self, field_name) < 0
        ]
        if missing:
            raise ValueError(
                "MiniCPM-o 4.5 native duplex requires tokenizer-defined special "
                f"tokens, missing or unknown: {', '.join(missing)}"
            )

    def _required_token_id(self, field_name: str) -> int:
        token_id = getattr(self, field_name, None)
        if not isinstance(token_id, int) or token_id < 0:
            token = _MINICPMO45_SPECIAL_TOKEN_FIELDS.get(field_name, field_name)
            raise ValueError(f"MiniCPM-o 4.5 missing required special token id for {token}")
        return token_id

    def stage_padding_token_id(self) -> int:
        return self._required_token_id("unit_end_token_id")

    def _audio_embedding_placeholder_token_id(self) -> int:
        token_id = getattr(self, "audio_placeholder_token_id", -1)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
        return self.stage_padding_token_id()

    @staticmethod
    def _decode_audio_payload(payload: dict[str, Any]) -> Any:
        audio = payload.get("audio") or payload.get("data")
        if not isinstance(audio, str):
            raise ValueError("audio append payload requires base64 audio")
        fmt = payload.get("format") or "pcm_f32le"
        if fmt != "pcm_f32le":
            raise ValueError(f"MiniCPM-o stage0 expects pcm_f32le audio, got {fmt!r}")
        return np.frombuffer(base64.b64decode(audio), dtype=np.float32)

    def _native_param(self, state: _MiniCPMO45Stage0SessionState, name: str, default: Any) -> Any:
        session_config = state.session_config
        extra_body = session_config.get("extra_body")
        if isinstance(extra_body, dict) and name in extra_body:
            return extra_body[name]
        if name in session_config:
            return session_config[name]
        return getattr(self, name, default)

    def _decode_next_token(self, logits: Any, state: _MiniCPMO45Stage0SessionState) -> int:
        del logits
        sampled_token_id = state.last_forward_metadata.get("sampled_token_id")
        if not isinstance(sampled_token_id, int):
            raise RuntimeError(
                "MiniCPM-o stage0 runner forward must return sampled_token_id; "
                "token selection must stay inside the model runner/sampler path."
            )
        return sampled_token_id

    @staticmethod
    def _valid_token_ids(token_ids: list[Any]) -> list[int]:
        valid: list[int] = []
        for token_id in token_ids:
            if isinstance(token_id, int) and token_id >= 0:
                valid.append(token_id)
        return valid

    @staticmethod
    def _extract_tts_handoff(native: dict[str, Any]) -> dict[str, Any] | None:
        stage_handoff = native.get("stage_handoff")
        if isinstance(stage_handoff, dict) and isinstance(stage_handoff.get("payload"), dict):
            return stage_handoff["payload"]
        handoff = native.get("tts_handoff")
        if isinstance(handoff, dict):
            return handoff
        if "omni_payload" in native:
            text = native.get("text", "")
            return {
                "type": MiniCPMO45DuplexPolicy.TTS_HANDOFF_TYPE,
                "omni_payload": native.get("omni_payload"),
                "llm_output_text": [text],
                "meta": {
                    "native_duplex_segment_text": text if isinstance(text, str) else "",
                    "override_keys": [
                        "llm_output_text",
                        ["meta", "native_duplex_segment_text"],
                    ],
                },
                "end_of_turn": bool(native.get("end_of_turn", False)),
            }
        return None

    @staticmethod
    def _strip_tts_handoff_fields(native: dict[str, Any]) -> None:
        native.pop("tts_token_ids", None)
        native.pop("tts_hidden_states", None)
        native.pop("omni_payload", None)
        native.pop("tts_handoff", None)
