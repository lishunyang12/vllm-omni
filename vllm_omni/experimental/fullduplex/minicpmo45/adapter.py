from __future__ import annotations

import base64
import binascii
import os
from typing import Any

import numpy as np
from vllm.multimodal.media import MediaConnector

from vllm_omni.experimental.fullduplex.core.events import (
    ModelAudioDelta,
    ModelListening,
    ModelSegmentEnded,
    ModelSpeaking,
    ModelTextDelta,
    ModelTurnEnded,
)
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence
from vllm_omni.experimental.fullduplex.minicpmo45.policy import MiniCPMO45DuplexPolicy
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexSessionConfig


class MiniCPMO45ModelEventAdapter:
    """Translate model-owned listen/speak boundaries into domain events.

    Playback cursors carried on :class:`ModelAudioDelta` are measured in output
    audio *samples* (one ``float32`` sample per 4 bytes, at the model's output
    sample rate, 24 kHz). Samples are exact, so the reducer's monotonic
    ``committed <= played <= sent <= generated`` invariant holds without the
    rounding drift a millisecond cursor would introduce. The cursor is
    cumulative per fence and resets when the turn ends (or when a barge-in mints
    a new epoch/fence, at which point the reducer also resets its cursor).
    """

    def __init__(self) -> None:
        self._speaking_fences: set[DuplexFence] = set()
        self._audio_samples: dict[DuplexFence, int] = {}

    def map_output(self, output: object, fence: DuplexFence) -> tuple[object, ...]:
        if not isinstance(output, dict):
            raise TypeError("MiniCPM-o duplex output must be a dictionary")
        meta = output.get("meta")
        metadata = meta if isinstance(meta, dict) else {}

        if output.get("is_listen") is True:
            return (ModelListening(fence=fence),)

        events: list[object] = []
        if output.get("is_listen") is False and fence not in self._speaking_fences:
            self._speaking_fences.add(fence)
            events.append(ModelSpeaking(fence=fence))
        text = output.get("text")
        if isinstance(text, str) and text:
            events.append(ModelTextDelta(text=text, fence=fence))

        audio_delta = self._audio_delta(output, fence)
        if audio_delta is not None:
            events.append(audio_delta)

        segment_end = bool(
            output.get("segment_end")
            or output.get("tts_is_last_chunk")
            or metadata.get("segment_end")
            or metadata.get("tts_is_last_chunk")
        )
        turn_end = bool(output.get("turn_end") or metadata.get("turn_end"))
        if segment_end:
            events.append(ModelSegmentEnded(fence=fence))
        if turn_end:
            self._speaking_fences.discard(fence)
            self._audio_samples.pop(fence, None)
            events.append(ModelTurnEnded(reason="model_turn_eos", fence=fence))
        return tuple(events)

    def _audio_delta(self, output: dict, fence: DuplexFence) -> ModelAudioDelta | None:
        """Decode the worker-normalized ``audio_data`` into a fenced audio delta.

        The worker (``normalize_native_audio_result``) turns the stage TTS
        ``audio_waveform`` into ``audio_data``: base64 of contiguous little-endian
        ``float32`` PCM. An empty/absent payload (a speak chunk that produced no
        audio yet) yields no delta.
        """
        encoded = output.get("audio_data")
        if not isinstance(encoded, str) or not encoded:
            return None
        try:
            pcm = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return None
        # float32 samples are 4 bytes; a ragged tail means a corrupt payload.
        if not pcm or len(pcm) % 4 != 0:
            return None
        cursor = self._audio_samples.get(fence, 0) + len(pcm) // 4
        self._audio_samples[fence] = cursor
        return ModelAudioDelta(
            data=pcm,
            generated_cursor=cursor,
            sent_cursor=cursor,
            fence=fence,
        )


class MiniCPMO45NativeDuplexServingAdapter:
    """Serving-side MiniCPM-o 4.5 native duplex session preparation.

    The generic duplex WebSocket handler should not let client-supplied local
    paths reach workers.  This adapter follows the existing media connector
    boundary: serving resolves client media URIs, then workers receive only
    normalized PCM payloads.  Server-owned model assets remain local paths here.
    """

    @classmethod
    def is_enabled(cls, config: DuplexSessionConfig) -> bool:
        extra_body = config.extra_body
        explicit = extra_body.get("native_duplex") or extra_body.get("minicpmo45_native_duplex")
        if explicit is not None:
            return bool(explicit)
        mode = extra_body.get("duplex_mode") or extra_body.get("runtime")
        return isinstance(mode, str) and mode.lower() in {"native", "model_native", "minicpmo45_native"}

    @classmethod
    async def prepare_session_config(cls, config: DuplexSessionConfig, *, model_config: Any) -> None:
        extra_body = dict(config.extra_body)
        if any(key in extra_body for key in ("ref_audio_path", "tts_ref_audio_path")):
            raise ValueError("ref_audio_path is not accepted by native duplex; use ref_audio URI instead")
        cls._apply_default_scheduler_policy(extra_body, model_config=model_config)

        ref_audio = config.ref_audio
        if ref_audio is None and isinstance(extra_body.get("ref_audio"), str):
            ref_audio = extra_body.pop("ref_audio")
        if ref_audio is None and isinstance(extra_body.get("tts_ref_audio"), str):
            ref_audio = extra_body.pop("tts_ref_audio")

        if ref_audio is None:
            default_ref = cls._default_ref_audio_path(config, model_config=model_config)
            if default_ref is None:
                cls._apply_first_append_context_tokens(
                    extra_body,
                    model_config=model_config,
                    instructions=config.instructions,
                    ref_sample_count=None,
                )
                cls._apply_new_user_turn_prefix_tokens(extra_body, model_config=model_config)
                config.extra_body = extra_body
                return
            wav_np, sr = cls._load_local_ref_audio(default_ref)
        else:
            wav_np, sr = await cls.resolve_ref_audio(ref_audio, model_config=model_config)

        wav_np = cls.normalize_ref_audio(wav_np, int(sr), target_sr=16000)
        # Trim to a whole number of pooled audio embeddings (100 ms frames) so
        # the first-append scheduler reserve can count them exactly.
        usable = (len(wav_np) // MiniCPMO45DuplexPolicy.SAMPLES_PER_AUDIO_TOKEN) * (
            MiniCPMO45DuplexPolicy.SAMPLES_PER_AUDIO_TOKEN
        )
        wav_np = wav_np[:usable]
        ref_audio_bytes = np.ascontiguousarray(wav_np, dtype=np.float32).tobytes()
        extra_body["ref_audio_data"] = base64.b64encode(ref_audio_bytes).decode("ascii")
        extra_body["ref_audio_format"] = "pcm_f32le"
        extra_body["ref_audio_sample_rate_hz"] = 16000
        cls._apply_first_append_context_tokens(
            extra_body,
            model_config=model_config,
            instructions=config.instructions,
            ref_sample_count=len(wav_np),
        )
        cls._apply_new_user_turn_prefix_tokens(extra_body, model_config=model_config)
        config.extra_body = extra_body
        config.ref_audio = None

    @classmethod
    def _apply_default_scheduler_policy(cls, extra_body: dict[str, object], *, model_config: Any) -> None:
        if "duplex_stage_max_tokens" in extra_body:
            raw_stage0 = None
        else:
            raw_stage0 = extra_body.get("duplex_stage0_max_tokens")
            if not isinstance(raw_stage0, int | float) or int(raw_stage0) <= 0:
                raw_stage0 = 20
            extra_body["duplex_stage_max_tokens"] = {"0": int(raw_stage0), "1": 8192}
        if "duplex_stage_sampling_params" not in extra_body:
            stage0_params: dict[str, object] = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 100,
                "repetition_penalty": 1.05,
            }
            stop_token_ids = cls._native_stage0_stop_token_ids(model_config)
            if stop_token_ids:
                stage0_params["stop_token_ids"] = stop_token_ids
            extra_body["duplex_stage_sampling_params"] = {"0": stage0_params}
        if "duplex_scheduler_token_id" not in extra_body:
            scheduler_token_id = cls._native_scheduler_token_id(model_config)
            if scheduler_token_id is not None:
                extra_body["duplex_scheduler_token_id"] = scheduler_token_id

    @classmethod
    def _apply_first_append_context_tokens(
        cls,
        extra_body: dict[str, object],
        *,
        model_config: Any,
        instructions: object,
        ref_sample_count: int | None,
    ) -> None:
        """Precompute the exact session-context token count for the engine.

        The first data-plane append carries the system template and optional
        reference-audio embeddings ahead of the first unit. The engine
        reserves scheduler slots from this count; an inexact count turns into
        pad embeddings inside the model KV (surplus) or truncated context
        (deficit), so it is computed with the same template and pooling math
        the worker uses.
        """
        if "duplex_first_append_context_tokens" in extra_body:
            return
        tokenizer = cls._load_native_tokenizer(model_config)
        if tokenizer is None:
            return
        prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
            instructions,
            ref_sample_count is not None,
        )
        try:
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        except Exception:
            return
        ref_tokens = MiniCPMO45DuplexPolicy.audio_token_count(ref_sample_count or 0)
        extra_body["duplex_first_append_context_tokens"] = len(prefix_ids) + ref_tokens + len(suffix_ids)

    @classmethod
    def _apply_new_user_turn_prefix_tokens(
        cls,
        extra_body: dict[str, object],
        *,
        model_config: Any,
    ) -> None:
        if "duplex_new_user_turn_prefix_tokens" in extra_body:
            return
        tokenizer = cls._load_native_tokenizer(model_config)
        if tokenizer is None:
            return
        try:
            prefix_ids = tokenizer.encode(MiniCPMO45DuplexPolicy.new_user_turn_prefix_text(), add_special_tokens=False)
            variant_counts = {
                variant: len(
                    tokenizer.encode(
                        MiniCPMO45DuplexPolicy.new_user_turn_prefix_text(variant),
                        add_special_tokens=False,
                    )
                )
                for variant in MiniCPMO45DuplexPolicy.new_user_turn_prefix_variants()
            }
        except Exception:
            return
        extra_body["duplex_new_user_turn_prefix_tokens"] = len(prefix_ids)
        extra_body["duplex_new_user_turn_prefix_tokens_by_variant"] = variant_counts

    @staticmethod
    def _native_stage0_stop_token_ids(model_config: Any) -> list[int]:
        tokenizer = MiniCPMO45NativeDuplexServingAdapter._load_native_tokenizer(model_config)
        if tokenizer is None:
            return []
        out: list[int] = []
        stop_token_fields = (
            "chunk_eos_token_id",
            "chunk_tts_eos_token_id",
            "listen_token_id",
            "turn_eos_token_id",
        )
        for field in stop_token_fields:
            token = MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS[field]
            token_id = MiniCPMO45NativeDuplexServingAdapter._convert_token_to_id(tokenizer, token)
            if token_id is not None and token_id not in out:
                out.append(token_id)
        return out

    @staticmethod
    def _native_scheduler_token_id(model_config: Any) -> int | None:
        tokenizer = MiniCPMO45NativeDuplexServingAdapter._load_native_tokenizer(model_config)
        if tokenizer is None:
            return None
        scheduler_tokens = (
            MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS["unit_token_id"],
            MiniCPMO45DuplexPolicy.OPTIONAL_TOKEN_FIELDS["audio_placeholder_token_id"],
        )
        for token in scheduler_tokens:
            token_id = MiniCPMO45NativeDuplexServingAdapter._convert_token_to_id(tokenizer, token)
            if token_id is not None:
                return token_id
        eos_id = getattr(tokenizer, "eos_token_id", None)
        try:
            return int(eos_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _load_native_tokenizer(model_config: Any) -> Any | None:
        model_path = getattr(model_config, "model", None)
        if not isinstance(model_path, str) or not model_path:
            return None
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            return None

    @staticmethod
    def _convert_token_to_id(tokenizer: Any, token: str) -> int | None:
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        value = None
        if callable(convert):
            value = convert(token)
            if isinstance(value, list):
                value = value[0] if len(value) == 1 else None
        try:
            token_id = int(value)
        except (TypeError, ValueError):
            token_id = -1
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if token_id >= 0 and token_id != unk_token_id:
            return token_id
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                ids = list(encode(token, add_special_tokens=False))
            except TypeError:
                ids = list(encode(token))
            if len(ids) == 1:
                try:
                    token_id = int(ids[0])
                except (TypeError, ValueError):
                    token_id = -1
                if token_id >= 0 and token_id != unk_token_id:
                    return token_id
        return None

    @staticmethod
    async def resolve_ref_audio(ref_audio: str, *, model_config: Any) -> tuple[np.ndarray, int]:
        connector = MediaConnector(
            allowed_local_media_path=getattr(model_config, "allowed_local_media_path", None),
            allowed_media_domains=getattr(model_config, "allowed_media_domains", None),
        )
        wav_np, sr = await connector.fetch_audio_async(ref_audio)
        return np.asarray(wav_np, dtype=np.float32), int(sr)

    @staticmethod
    def _default_ref_audio_path(config: DuplexSessionConfig, *, model_config: Any) -> str | None:
        candidates = (config.model, getattr(model_config, "model", None))
        for model in candidates:
            if not isinstance(model, str) or not os.path.isdir(model):
                continue
            ref_path = os.path.join(model, "assets", "HT_ref_audio.wav")
            if os.path.exists(ref_path):
                return ref_path
        return None

    @staticmethod
    def _load_local_ref_audio(path: str) -> tuple[np.ndarray, int]:
        import soundfile as sf

        wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
        return wav_np, int(sr)

    @staticmethod
    def normalize_ref_audio(wav_np: np.ndarray, sample_rate: int, *, target_sr: int) -> np.ndarray:
        wav_np = np.asarray(wav_np, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=-1)
        wav_np = wav_np.reshape(-1)
        if sample_rate <= 0 or sample_rate == target_sr or wav_np.size == 0:
            return wav_np.astype(np.float32, copy=False)
        import torch
        import torchaudio

        audio = torch.from_numpy(wav_np).to(dtype=torch.float32).unsqueeze(0)
        resampled = torchaudio.functional.resample(audio, int(sample_rate), int(target_sr))
        return resampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
