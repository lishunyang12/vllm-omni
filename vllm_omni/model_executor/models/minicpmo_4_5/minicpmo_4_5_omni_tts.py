"""MiniCPM-o 4.5 Talker + Token2Wav: MiniCPMTTS with hidden_text_merge condition.

Pipeline:
  1. Receive thinker hidden_states + full token IDs via additional_information
  2. Extract tts_bos..tts_eos region
  3. Build condition: emb_text(tokens) + projector_semantic(hidden) (hidden_text_merge)
  4. Run MiniCPMTTS.generate() -> discrete audio tokens
  5. Run Token2wav(tokens) -> waveform bytes -> numpy array
"""

import hashlib
import io
import logging
import os
import tempfile
import time
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.v1.outputs import SamplerOutput

from vllm_omni.experimental.fullduplex.engine.intermediate import get_stream_request_key, get_tts_handoff
from vllm_omni.platforms import current_omni_platform

try:
    from stepaudio2 import Token2wav as _Token2wav

    _stepaudio2_available = True
    _stepaudio2_import_error = None
except ImportError as e:
    _Token2wav = None
    _stepaudio2_available = False
    _stepaudio2_import_error = e

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MiniCPMO45TTSRuntimeConfig:
    """Internal MiniCPM-o 4.5 Talker runtime defaults.

    These values are deliberately not process environment knobs. If a value
    needs to become user configurable, route it through stage/model config with
    an explicit API contract instead of adding another ad hoc env var.
    """

    token2wav_n_timesteps: int = 10
    tts_dtype: torch.dtype = torch.float32
    token2wav_autocast_dtype: torch.dtype | None = None
    use_direct_token2wav: bool = True
    ref_audio_file_cache_size: int = 16
    max_token_ratio: int = 32
    min_max_new_tokens: int = 256
    hard_max_new_tokens: int = 16384
    min_new_tokens: int = 50
    streaming_generator_chunk: int = 25
    streaming_vocoder_threshold: int = 2500
    streaming_vocoder_chunk: int = 50
    enable_streaming_probe: bool = False
    enable_fast_generate: bool = False
    enable_static_kv: bool = False
    enable_prealloc_kv: bool = False
    enable_torch_compile: bool = False
    torch_compile_mode: str = "default"
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = True
    torch_compile_target: str = "module"
    torch_compile_strict: bool = False
    enable_compile_warmup: bool = False
    compile_warmup_seq_lens: tuple[int, ...] = (16,)
    compile_warmup_decode_steps: int = 2
    compile_warmup_generate: bool = False


def _install_torchaudio_soundfile_shim() -> None:
    """Monkey-patch torchaudio.load to use soundfile instead of the default
    torchcodec backend, which requires libtorchcodec/ffmpeg shared libs that
    may be missing on the deployment machine."""
    try:
        import torchaudio

        if getattr(torchaudio, "_soundfile_shim_installed", False):
            return
        _orig_load = torchaudio.load

        def _patched_load(uri, *args, **kwargs):
            try:
                return _orig_load(uri, *args, **kwargs)
            except Exception:
                import numpy as _np
                import soundfile as _sf

                data, sr = _sf.read(uri, dtype="float32", always_2d=True)
                wav = torch.from_numpy(_np.ascontiguousarray(data.T))
                return wav, sr

        torchaudio.load = _patched_load
        torchaudio._soundfile_shim_installed = True
        logger.info("Installed torchaudio.load soundfile shim")
    except Exception as _e:
        logger.warning("Could not install torchaudio shim: %s", _e)


_install_torchaudio_soundfile_shim()


class _PreallocatedKVCache:
    """Dynamic-length KV cache backed by fixed per-layer buffers.

    HuggingFace's default DynamicLayer appends one token with torch.cat on
    every decode step. For MiniCPM-o 4.5 TTS bs=1 decode that shows up as
    thousands of small cat kernels. This cache preserves the same growing
    attention length while avoiding per-step cache reallocation/copy.
    """

    is_compilable = False

    def __init__(self, *, num_layers: int, max_cache_len: int):
        self.max_cache_len = max_cache_len
        self.is_sliding = [False] * num_layers
        self._layers: list[dict[str, torch.Tensor | int | None]] = [
            {"keys": None, "values": None, "length": 0} for _ in range(num_layers)
        ]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self._layers[layer_idx]
        keys = layer["keys"]
        values = layer["values"]
        if keys is None or values is None:
            key_shape = list(key_states.shape)
            value_shape = list(value_states.shape)
            key_shape[-2] = self.max_cache_len
            value_shape[-2] = self.max_cache_len
            keys = torch.empty(key_shape, dtype=key_states.dtype, device=key_states.device)
            values = torch.empty(value_shape, dtype=value_states.dtype, device=value_states.device)
            layer["keys"] = keys
            layer["values"] = values

        start = int(layer["length"] or 0)
        step = key_states.shape[-2]
        end = start + step
        if end > self.max_cache_len:
            raise RuntimeError(
                f"MiniCPM-o 4.5 preallocated TTS KV cache overflow: end={end}, max_cache_len={self.max_cache_len}"
            )

        keys[:, :, start:end, :].copy_(key_states)
        values[:, :, start:end, :].copy_(value_states)
        layer["length"] = end
        return keys[:, :, :end, :], values[:, :, :end, :]

    def get_seq_length(self) -> int:
        if not self._layers:
            return 0
        return int(self._layers[0]["length"] or 0)

    def get_mask_sizes(self, query_length: int, layer_idx: int | None = None) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len


class _TalkerTurnState:
    """Per-turn talker continuity for native duplex.

    The official duplex implementation runs ONE TTS stream per spoken turn:
    each 1s unit appends its condition to the same KV (carried
    past_key_values + text_start_pos) and the token2wav stream caches and
    token buffer persist across units, reset only at turn end. Synthesizing
    units as independent utterances resets prosody every second and garbles
    the reply.
    """

    __slots__ = (
        "past_key_values",
        "text_start_pos",
        "token2wav_buffer",
        "prompt_wav_path",
        "temp_prompt_wav_path",
        "epoch",
        "turn_id",
    )

    def __init__(
        self,
        prompt_wav_path,
        temp_prompt_wav_path,
        *,
        epoch: int | None = None,
        turn_id: int | None = None,
    ):
        self.past_key_values = None
        self.text_start_pos = 0
        # Official seeds each turn's vocoder buffer with three silence
        # tokens so the first synthesized window does not directly abut the
        # reference-audio prompt cache (audible ref-voice bleed otherwise).
        self.token2wav_buffer: list[int] = [_T2W_SILENCE_TOKEN] * 3
        self.prompt_wav_path = prompt_wav_path
        self.temp_prompt_wav_path = temp_prompt_wav_path
        self.epoch = epoch
        self.turn_id = turn_id


_T2W_SILENCE_TOKEN = 4218


def _soundfile_patched_save(orig_save):
    def _patched_save(uri, src, sample_rate, **kw):
        kw.pop("backend", None)
        if hasattr(uri, "write"):
            sf.write(uri, src.cpu().numpy().T, sample_rate, format="WAV")
            return
        return orig_save(uri, src, sample_rate, backend="soundfile", **kw)

    return _patched_save


def _torch_clone_recursive(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _torch_clone_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_torch_clone_recursive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_torch_clone_recursive(v) for v in obj)
    return obj


class MiniCPMO45OmniTTSForConditionalGeneration(nn.Module, SupportsPP):
    """MiniCPM-o 4.5 Talker: MiniCPMTTS + Token2wav in a single forward pass."""

    # llm2tts hands the FULL accumulated condition per handoff (the runner's
    # resume-prefill path REPLACES the streaming buffer, so runner-side
    # accumulation is lossy); the per-turn state consumes it by cursor.

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMOConfig

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self._runtime_config = MiniCPMO45TTSRuntimeConfig()

        self.tts = None
        self.audio_tokenizer = None
        self._assets_loaded = False
        self._stream_gens: dict[str, Any] = {}
        self._talker_turn_states: dict[str, _TalkerTurnState] = {}
        # Consumed-cursor into the accumulated handoff condition for the
        # currently open spoken turn.
        self._talker_consumed_tokens: dict[str, int] = {}
        self._talker_request_keys: dict[str, str] = {}
        self._t2w_base_caches: dict[str, tuple[Any, Any]] = {}
        self._ar_last_chunk_flags: list[bool] = [True]
        self._ar_turn_end_flags: list[bool] = [False]
        self._text_tokenizer = None

        tts_config = getattr(config, "tts_config", None)
        if tts_config is not None:
            self._tts_config = tts_config
            self._tts_bos_id = self._config_token_id(tts_config, "audio_bos_token_id")
            self._text_eos_id = self._config_token_id(tts_config, "text_eos_token_id")
            self._num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            self._hidden_size = getattr(tts_config, "hidden_size", 768)
            self._normalize = getattr(tts_config, "normalize_projected_hidden", True)
        else:
            self._tts_config = None

    def _tts_runtime_config(self) -> MiniCPMO45TTSRuntimeConfig:
        cfg = getattr(self, "_runtime_config", None)
        if cfg is None:
            cfg = MiniCPMO45TTSRuntimeConfig()
            self._runtime_config = cfg
        return cfg

    @staticmethod
    def _config_token_id(config: Any, attr: str) -> int:
        value = getattr(config, attr, None)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            raise ValueError(f"MiniCPM-o 4.5 TTS config missing required {attr}")
        return int(value)

    def _get_text_tokenizer(self) -> Any:
        tokenizer = getattr(self, "_text_tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

        tokenizer = cached_tokenizer_from_config(self.vllm_config.model_config)
        self._text_tokenizer = tokenizer
        return tokenizer

    def _tokenizer_token_id(self, token: str) -> int | None:
        tokenizer = self._get_text_tokenizer()
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            value = convert(token)
            if isinstance(value, list):
                value = value[0] if len(value) == 1 else None
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and candidate >= 0 and candidate != unk_token_id:
                return candidate
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            ids = list(encode(token, add_special_tokens=False))
            if len(ids) == 1:
                candidate = int(ids[0])
                if candidate >= 0 and candidate != unk_token_id:
                    return candidate
        return None

    def _scheduler_eos_token_id(self) -> int:
        eos_raw = getattr(self.config, "eos_token_id", None)
        if isinstance(eos_raw, (list, tuple)):
            eos_raw = eos_raw[0] if eos_raw else None
        if eos_raw is not None:
            return int(eos_raw)
        eos_id = self._tokenizer_token_id("<|im_end|>")
        if eos_id is None:
            raise ValueError(
                "MiniCPM-o 4.5 TTS scheduler EOS requires config.eos_token_id or tokenizer-defined <|im_end|>"
            )
        return eos_id

    def _lazy_init_tts(self):
        if self._assets_loaded or self._tts_config is None:
            return
        self._assets_loaded = True
        try:
            model_path = self.vllm_config.model_config.model
            import os
            import sys

            if model_path not in sys.path:
                sys.path.insert(0, model_path)
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            MiniCPMTTS = get_class_from_dynamic_module("modeling_minicpmo.MiniCPMTTS", model_path)

            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                for name, default in (
                    ("top_p", 0.85),
                    ("top_k", 25),
                    ("repetition_penalty", 1.05),
                    ("temperature", 0.8),
                ):
                    if not hasattr(self._tts_config, name):
                        setattr(self._tts_config, name, default)
                self.tts_obj = MiniCPMTTS(config=self._tts_config, audio_tokenizer=None)
            finally:
                torch.set_default_dtype(prev_dtype)
            tts_module = import_module(self.tts_obj.__class__.__module__)

            def get_tts_module_attr(name: str):
                return self.tts_obj.generate.__globals__.get(name) or getattr(tts_module, name, None)

            self._tts_sampling_params_cls = get_tts_module_attr("TTSSamplingParams")
            self._tts_gen_logits = get_tts_module_attr("gen_logits")
            self._tts_parametrize = get_tts_module_attr("P")
            self._tts_streaming_generator_cls = get_tts_module_attr("TTSStreamingGenerator")
            self.emb_text = self.tts_obj.emb_text
            self.projector_semantic = self.tts_obj.projector_semantic

            token2wav_dir = os.path.join(model_path, "assets", "token2wav")
            if os.path.isdir(token2wav_dir):
                if not _stepaudio2_available:
                    raise ImportError(
                        "MiniCPM-o 4.5 token2wav stage requires the stepaudio2 package, "
                        "and all of its runtime dependencies."
                    ) from _stepaudio2_import_error
                self._token2wav_n_timesteps = self._tts_runtime_config().token2wav_n_timesteps
                prev_dtype2 = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                try:
                    self.audio_tokenizer = _Token2wav(
                        token2wav_dir,
                        float16=False,
                        n_timesteps=self._token2wav_n_timesteps,
                    )
                finally:
                    torch.set_default_dtype(prev_dtype2)
                self.tts_obj.audio_tokenizer = self.audio_tokenizer
                logger.info(
                    "Loaded Token2wav from %s (n_timesteps=%d)",
                    token2wav_dir,
                    self._token2wav_n_timesteps,
                )
        except ImportError:
            # Surface missing dependencies directly so users can act on them
            # instead of getting a silent None waveform downstream.
            raise
        except Exception as e:
            logger.error("Failed to init 4.5 TTS: %s", e, exc_info=True)

    def _build_tts_sampling_params(self):
        params_cls = getattr(self, "_tts_sampling_params_cls", None)
        if params_cls is None or not hasattr(self, "tts_obj"):
            return None

        tts = self.tts_obj

        top_p = getattr(tts, "top_p", getattr(tts.config, "top_p", 0.85))
        top_k = getattr(tts, "top_k", getattr(tts.config, "top_k", 25))
        repetition_penalty = getattr(
            tts,
            "repetition_penalty",
            getattr(tts.config, "repetition_penalty", 1.05),
        )
        temperature = getattr(tts.config, "temperature", 0.8)

        return params_cls(
            top_p=None if top_p is not None and top_p >= 1.0 else top_p,
            top_k=None if top_k is not None and top_k <= 0 else top_k,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

    def _target_tts_dtype(self) -> torch.dtype:
        return self._tts_runtime_config().tts_dtype

    def _token2wav_autocast_context(self):
        dtype = self._tts_runtime_config().token2wav_autocast_dtype
        if dtype is None:
            return torch.amp.autocast("cuda", enabled=False), "off"
        if dtype is torch.bfloat16:
            return torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16), "bf16"
        if dtype is torch.float16:
            return torch.amp.autocast("cuda", enabled=True, dtype=torch.float16), "fp16"
        raise ValueError("MiniCPM-o 4.5 token2wav autocast only supports None, bfloat16, or float16")

    def _should_use_direct_token2wav(self) -> bool:
        return self._tts_runtime_config().use_direct_token2wav

    def _should_stream_output(self, info: dict[str, Any] | None = None) -> bool:
        if isinstance(info, dict):
            for key in ("stream_output", "minicpmo45_native_duplex"):
                value = info.get(key)
                if isinstance(value, bool):
                    return value
        return False

    def _token2wav_prompt_cache_key(self, prompt_wav: str | None) -> str | None:
        return os.path.abspath(prompt_wav) if prompt_wav else None

    def _reset_token2wav_cache_if_needed(self, prompt_wav: str | None) -> None:
        token2wav = self.audio_tokenizer
        if token2wav is None:
            return
        cache_key = self._token2wav_prompt_cache_key(prompt_wav)
        if getattr(self, "_token2wav_prompt_cache_id", None) != cache_key:
            token2wav.cache = None
            self._token2wav_prompt_cache_id = cache_key

    def _normalize_ref_audio_tensor(self, ref_audio) -> np.ndarray | None:
        if ref_audio is None:
            return None
        if isinstance(ref_audio, torch.Tensor):
            waveform = ref_audio.detach().float().cpu().numpy()
        else:
            waveform = np.asarray(ref_audio, dtype=np.float32)
        if waveform.ndim > 1:
            if waveform.shape[0] <= 2 and waveform.shape[-1] > waveform.shape[0]:
                waveform = waveform.mean(axis=0)
            else:
                waveform = waveform.mean(axis=-1)
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        return waveform if waveform.size else None

    def _write_ref_audio_prompt_wav(self, ref_audio, ref_audio_sr: int | None) -> str | None:
        waveform = self._normalize_ref_audio_tensor(ref_audio)
        if waveform is None:
            return None
        sample_rate = int(ref_audio_sr or 24000)
        cache_size = self._tts_runtime_config().ref_audio_file_cache_size
        if cache_size > 0:
            digest = hashlib.sha256()
            digest.update(str(sample_rate).encode("ascii"))
            digest.update(waveform.tobytes())
            cache_key = digest.hexdigest()
            cache = getattr(self, "_ref_audio_prompt_files", None)
            if cache is None:
                cache = OrderedDict()
                self._ref_audio_prompt_files = cache
            cached_path = cache.get(cache_key)
            if cached_path and os.path.exists(cached_path):
                cache.move_to_end(cache_key)
                return cached_path

            tmp_path = os.path.join(tempfile.gettempdir(), f"minicpmo45_ref_{cache_key[:24]}_{sample_rate}.wav")
            if not os.path.exists(tmp_path):
                sf.write(tmp_path, waveform, sample_rate, format="WAV")
            cache[cache_key] = tmp_path
            cache.move_to_end(cache_key)
            while len(cache) > cache_size:
                _, evicted_path = cache.popitem(last=False)
                try:
                    os.unlink(evicted_path)
                except OSError:
                    pass
            return tmp_path

        tmp = tempfile.NamedTemporaryFile(prefix="minicpmo45_ref_", suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        sf.write(tmp_path, waveform, sample_rate, format="WAV")
        return tmp_path

    def _is_cached_ref_audio_prompt_wav(self, prompt_wav: str | None) -> bool:
        cache = getattr(self, "_ref_audio_prompt_files", None)
        return bool(prompt_wav and cache and prompt_wav in cache.values())

    def _run_token2wav_direct(
        self,
        generated_speech_tokens: list[int] | torch.Tensor,
        prompt_wav: str | None,
    ) -> tuple[torch.Tensor, int]:
        """Run Token2wav without WAV encode/decode round-tripping.

        ``stepaudio2.Token2wav.__call__`` renders the GPU waveform to a WAV
        BytesIO object and the vLLM-Omni adapter immediately decodes it back to
        a numpy waveform.  The engine already expects a float waveform tensor,
        so keep the same flow/HIFT computation and return the waveform directly.
        This also makes the configured ``n_timesteps`` apply to one-shot
        inference; upstream currently hard-codes ``10`` in ``__call__``.
        """
        token2wav = self.audio_tokenizer
        if token2wav is None:
            raise RuntimeError("Token2wav is not initialized")
        required_attrs = ("_prepare_prompt", "flow", "hift")
        if any(not hasattr(token2wav, attr) for attr in required_attrs):
            raise RuntimeError("Token2wav direct path is incompatible with the installed stepaudio2 package")

        self._reset_token2wav_cache_if_needed(prompt_wav)
        if token2wav.cache is None:
            token2wav.cache = token2wav._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = token2wav.cache

        device = prompt_speech_tokens.device
        if isinstance(generated_speech_tokens, torch.Tensor):
            generated = generated_speech_tokens
            if generated.ndim == 1:
                generated = generated.unsqueeze(0)
            elif generated.ndim == 3 and generated.shape[-1] == 1:
                generated = generated.squeeze(-1)
            generated = generated.to(device=device, dtype=torch.int32)
        else:
            generated = torch.tensor([generated_speech_tokens], dtype=torch.int32, device=device)
        generated_lens = torch.tensor([generated.shape[1]], dtype=torch.int32, device=device)
        mel = token2wav.flow.inference(
            generated,
            generated_lens,
            prompt_speech_tokens,
            prompt_speech_tokens_lens,
            prompt_mels,
            prompt_mels_lens,
            spk_emb,
            self._token2wav_n_timesteps,
        )
        wav, _ = token2wav.hift(speech_feat=mel)
        waveform = wav.squeeze(0).detach().float().reshape(-1).cpu().contiguous()
        return waveform, 24000

    def _build_tts_condition_embeds(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        tts = self.tts_obj
        device = tts.emb_text.weight.device
        dtype = tts.emb_text.weight.dtype
        llm_embeds = tts.emb_text(tts_token_ids.to(device))
        hidden_embeds = tts.projector_semantic(tts_hidden_states.to(device=device, dtype=dtype))
        if getattr(tts.config, "normalize_projected_hidden", False):
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        return llm_embeds + hidden_embeds

    def _normalize_tts_handoff_tensors(
        self,
        tts_token_ids: Any,
        tts_hidden_states: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.as_tensor(tts_token_ids, dtype=torch.long)
        if token_ids.ndim == 0:
            token_ids = token_ids.reshape(1)
        elif token_ids.ndim > 1:
            token_ids = token_ids.reshape(-1)

        hidden_states = torch.as_tensor(tts_hidden_states, dtype=torch.float32)
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(0)
        elif hidden_states.ndim == 3 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states.squeeze(0)
        elif hidden_states.ndim != 2:
            hidden_states = hidden_states.reshape(token_ids.numel(), -1)

        if hidden_states.shape[0] != token_ids.numel():
            raise ValueError(
                "MiniCPM-o 4.5 TTS handoff has mismatched token/hidden lengths: "
                f"tokens={token_ids.numel()} hidden_rows={hidden_states.shape[0]}"
            )
        return token_ids.contiguous(), hidden_states.contiguous()

    def _resolve_prompt_wav_path(self, ref_audio, ref_audio_sr: int | None) -> tuple[str | None, str | None]:
        model_path = self.vllm_config.model_config.model
        default_ref = os.path.join(model_path, "assets", "HT_ref_audio.wav")
        temp_prompt_wav_path = self._write_ref_audio_prompt_wav(ref_audio, ref_audio_sr)
        prompt_wav_path = temp_prompt_wav_path or (default_ref if os.path.exists(default_ref) else None)
        return prompt_wav_path, temp_prompt_wav_path

    def _max_tts_tokens_for_text(self, num_text: int) -> tuple[int, int]:
        cfg = self._tts_runtime_config()
        max_new_token = min(
            cfg.hard_max_new_tokens,
            max(cfg.min_max_new_tokens, num_text * cfg.max_token_ratio),
        )
        return cfg.min_new_tokens, max_new_token

    def _stream_request_key(self, info: dict[str, Any]) -> str:
        return get_stream_request_key(info)

    @staticmethod
    def _coerce_request_key(value: Any) -> str | None:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if value is None:
            return None
        text = str(value)
        return text if text else None

    @staticmethod
    def _coerce_epoch(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    _coerce_turn_id = _coerce_epoch

    def _stream_epoch(self, info: dict[str, Any]) -> int | None:
        duplex = info.get("duplex")
        if isinstance(duplex, dict):
            epoch = self._coerce_epoch(duplex.get("epoch"))
            if epoch is not None:
                return epoch
        meta = info.get("meta")
        if isinstance(meta, dict):
            epoch = self._coerce_epoch(meta.get("epoch"))
            if epoch is not None:
                return epoch
        return self._coerce_epoch(info.get("epoch"))

    def _stream_turn_id(self, info: dict[str, Any]) -> int | None:
        duplex = info.get("duplex")
        if isinstance(duplex, dict):
            turn_id = self._coerce_turn_id(duplex.get("turn_id"))
            if turn_id is not None:
                return turn_id
        meta = info.get("meta")
        if isinstance(meta, dict):
            turn_id = self._coerce_turn_id(meta.get("turn_id"))
            if turn_id is not None:
                return turn_id
        return self._coerce_turn_id(info.get("turn_id"))

    def _remember_talker_request_key(self, info: dict[str, Any], key: str) -> None:
        aliases = {
            self._coerce_request_key(info.get("request_id")),
            self._coerce_request_key(info.get("_omni_req_id")),
        }
        duplex = info.get("duplex")
        if isinstance(duplex, dict):
            aliases.add(self._coerce_request_key(duplex.get("request_id")))
        request_keys = getattr(self, "_talker_request_keys", None)
        if request_keys is None:
            request_keys = {}
            self._talker_request_keys = request_keys
        for alias in aliases:
            if alias and alias != key:
                request_keys[alias] = key

    def _empty_audio_chunk(self) -> torch.Tensor:
        return torch.zeros((0,), dtype=torch.float32)

    @staticmethod
    def _extract_tts_handoff(info: dict[str, Any]) -> tuple[Any, Any]:
        return get_tts_handoff(info)

    def _native_duplex_input_ends_turn(self, info: dict[str, Any]) -> bool:
        if info.get("minicpmo45_native_duplex") is not True:
            return False
        meta = info.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        if bool(info.get("turn_end") or info.get("end_of_turn") or meta.get("turn_end") or meta.get("end_of_turn")):
            return True
        turn_eos_id = self._coerce_epoch(meta.get("turn_eos_token_id"))
        if turn_eos_id is None:
            return False
        tts_token_ids, _ = self._extract_tts_handoff(info)
        if isinstance(tts_token_ids, torch.Tensor):
            return bool((tts_token_ids == turn_eos_id).any().item())
        if isinstance(tts_token_ids, np.ndarray):
            return bool(np.any(tts_token_ids == turn_eos_id))
        if isinstance(tts_token_ids, (list, tuple)):
            return turn_eos_id in tts_token_ids
        return False

    def _t2w_pre_lookahead(self) -> int:
        flow = getattr(self.audio_tokenizer, "flow", None)
        try:
            return int(getattr(flow, "pre_lookahead_len", 3) or 3)
        except (TypeError, ValueError):
            return 3

    def _begin_turn_vocoder_cache(self, prompt_wav_path: str | None) -> None:
        """Restore a fresh per-turn clone of the ref-audio vocoder caches."""
        import torchaudio

        cache_key = prompt_wav_path or ""
        base = self._t2w_base_caches.get(cache_key)
        if base is None:
            _orig_save = torchaudio.save
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                torchaudio.save = _soundfile_patched_save(_orig_save)
                stream_cache, hift_cache_dict = self.audio_tokenizer.set_stream_cache(prompt_wav_path)
            finally:
                torch.set_default_dtype(prev_dtype)
                torchaudio.save = _orig_save
            base = (
                _torch_clone_recursive(stream_cache),
                _torch_clone_recursive(hift_cache_dict),
            )
            self._t2w_base_caches[cache_key] = base
        self.audio_tokenizer.stream_cache = _torch_clone_recursive(base[0])
        self.audio_tokenizer.hift_cache_dict = _torch_clone_recursive(base[1])

    def _t2w_stream_window(self, token_list: list[int], prompt_wav_path: str | None, *, last_chunk: bool):
        import torchaudio

        _orig_save = torchaudio.save
        prev_dtype = torch.get_default_dtype()
        autocast_context, _ = self._token2wav_autocast_context()
        torch.set_default_dtype(torch.float32)
        try:
            torchaudio.save = _soundfile_patched_save(_orig_save)
            with autocast_context:
                wav_np = self.audio_tokenizer.stream(
                    token_list,
                    prompt_wav_path,
                    last_chunk=bool(last_chunk),
                    return_waveform=True,
                )
        finally:
            torch.set_default_dtype(prev_dtype)
            torchaudio.save = _orig_save
        return torch.as_tensor(np.asarray(wav_np).reshape(-1), dtype=torch.float32).cpu().contiguous()

    def _native_duplex_vocode_tokens(
        self,
        state: _TalkerTurnState,
        new_tokens: torch.Tensor,
        *,
        turn_end: bool,
        force_flush: bool,
        chunk_size: int,
    ) -> list[torch.Tensor]:
        """Run Token2wav with the same buffering policy as official duplex."""
        pre_lookahead = self._t2w_pre_lookahead()
        token_list = new_tokens.reshape(-1).detach().cpu().tolist()
        state.token2wav_buffer.extend(int(t) for t in token_list)
        pieces: list[torch.Tensor] = []

        if force_flush:
            while len(state.token2wav_buffer) >= pre_lookahead + 5:
                chunk_to_process = min(chunk_size + pre_lookahead, len(state.token2wav_buffer))
                window = state.token2wav_buffer[:chunk_to_process]
                pieces.append(self._t2w_stream_window(window, state.prompt_wav_path, last_chunk=False))
                state.token2wav_buffer = state.token2wav_buffer[min(chunk_size, chunk_to_process - pre_lookahead) :]
        else:
            while len(state.token2wav_buffer) >= chunk_size + pre_lookahead:
                window = state.token2wav_buffer[: chunk_size + pre_lookahead]
                pieces.append(self._t2w_stream_window(window, state.prompt_wav_path, last_chunk=False))
                state.token2wav_buffer = state.token2wav_buffer[chunk_size:]

        if turn_end and state.token2wav_buffer:
            pieces.append(
                self._t2w_stream_window(
                    list(state.token2wav_buffer),
                    state.prompt_wav_path,
                    last_chunk=True,
                )
            )
            state.token2wav_buffer = []

        return pieces

    def _close_turn_state(
        self,
        key: str,
        *,
        expected_epoch: int | None = None,
        expected_turn_id: int | None = None,
    ) -> bool:
        state = self._talker_turn_states.get(key)
        if (
            state is not None
            and expected_epoch is not None
            and state.epoch is not None
            and state.epoch != expected_epoch
        ):
            return False
        if (
            state is not None
            and expected_turn_id is not None
            and state.turn_id is not None
            and state.turn_id != expected_turn_id
        ):
            return False
        state = self._talker_turn_states.pop(key, None)
        self._talker_consumed_tokens.pop(key, None)
        request_keys = getattr(self, "_talker_request_keys", None)
        if isinstance(request_keys, dict):
            request_keys.pop(key, None)
            for alias, mapped_key in list(request_keys.items()):
                if mapped_key == key:
                    request_keys.pop(alias, None)
        if state is None:
            return True
        if self.audio_tokenizer is not None:
            self.audio_tokenizer.stream_cache = None
            self.audio_tokenizer.hift_cache_dict = {}
        temp_path = state.temp_prompt_wav_path
        if temp_path and not self._is_cached_ref_audio_prompt_wav(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        return True

    @staticmethod
    def _stream_identity_relation(
        state: _TalkerTurnState | None,
        *,
        epoch: int | None,
        turn_id: int | None,
    ) -> str:
        if state is None:
            return "current"
        if epoch is not None and state.epoch is not None:
            if epoch < state.epoch:
                return "stale"
            if epoch > state.epoch:
                return "newer"
        if turn_id is not None and state.turn_id is not None:
            if turn_id < state.turn_id:
                return "stale"
            if turn_id > state.turn_id:
                return "newer"
        return "current"

    def _warmup_duplex_vocoder(self) -> None:
        """Pre-compile the two token2wav stream() modes used per turn.

        The first stream() call of each mode (mid-turn 25+pre_lookahead
        window, and the variable-size last_chunk tail flush) costs ~20s of
        one-time compilation; without this warmup the first spoken turn of
        the first session stalls for both. Mirrors the official demo's
        precompile step.
        """
        if getattr(self, "_t2w_warmed", False):
            return
        self._t2w_warmed = True
        if os.environ.get("MINICPMO45_SKIP_T2W_WARMUP") == "1":
            return
        try:
            self._lazy_init_tts()
            if self.audio_tokenizer is None:
                return
            prompt_wav_path, _ = self._resolve_prompt_wav_path(None, None)
            if prompt_wav_path is None:
                return
            t0 = time.perf_counter()
            chunk_size = self._tts_runtime_config().streaming_generator_chunk
            pre_lookahead = self._t2w_pre_lookahead()
            self._begin_turn_vocoder_cache(prompt_wav_path)
            self._t2w_stream_window(
                [_T2W_SILENCE_TOKEN] * (chunk_size + pre_lookahead),
                prompt_wav_path,
                last_chunk=False,
            )
            self._begin_turn_vocoder_cache(prompt_wav_path)
            self._t2w_stream_window(
                [_T2W_SILENCE_TOKEN] * (chunk_size + pre_lookahead),
                prompt_wav_path,
                last_chunk=True,
            )
            self.audio_tokenizer.stream_cache = None
            self.audio_tokenizer.hift_cache_dict = {}
            logger.info("4.5 Talker duplex vocoder warmup done in %.1fs", time.perf_counter() - t0)
        except Exception:
            logger.exception("4.5 Talker duplex vocoder warmup failed")

    def _create_native_duplex_stream_gen(self, info: dict[str, Any]):
        """Per-segment generator over a persistent per-turn talker stream.

        Mirrors the official duplex talker: each unit calls
        ``MiniCPMTTS.generate_chunk`` while carrying ``past_key_values`` and
        ``text_start_pos`` across the spoken turn, then feeds the generated
        audio tokens through the same per-turn Token2wav buffer/cache.
        """
        key = self._stream_request_key(info)
        self._remember_talker_request_key(info, key)
        stream_epoch = self._stream_epoch(info)
        stream_turn_id = self._stream_turn_id(info)
        meta_info = info.get("meta") if isinstance(info.get("meta"), dict) else {}
        codes_info = info.get("codes") if isinstance(info.get("codes"), dict) else {}
        tts_token_ids, tts_hidden_states = self._extract_tts_handoff(info)

        self._lazy_init_tts()
        if getattr(self, "tts_obj", None) is None or self.audio_tokenizer is None:
            logger.warning("4.5 Talker duplex streaming: TTS runtime unavailable")
            yield self._empty_audio_chunk(), True
            return

        if isinstance(tts_token_ids, torch.Tensor):
            ids_list = tts_token_ids.reshape(-1).tolist()
        elif isinstance(tts_token_ids, list):
            ids_list = [int(t) for t in tts_token_ids]
        else:
            ids_list = []

        state = self._talker_turn_states.get(key)
        identity_relation = self._stream_identity_relation(
            state,
            epoch=stream_epoch,
            turn_id=stream_turn_id,
        )
        if identity_relation == "stale":
            logger.info(
                "4.5 Talker duplex drop stale handoff: key=%s state_epoch=%s state_turn_id=%s "
                "handoff_epoch=%s handoff_turn_id=%s",
                key,
                getattr(state, "epoch", None),
                getattr(state, "turn_id", None),
                stream_epoch,
                stream_turn_id,
            )
            yield self._empty_audio_chunk(), True
            return
        if identity_relation == "newer":
            self._close_turn_state(key)
            state = None
        consumed = self._talker_consumed_tokens.get(key, 0)
        if consumed > len(ids_list):
            consumed = 0
        pending_ids = ids_list[consumed:]

        turn_eos_raw = meta_info.get("turn_eos_token_id")
        try:
            turn_eos_id = int(turn_eos_raw) if turn_eos_raw is not None else None
        except (TypeError, ValueError):
            turn_eos_id = None
        explicit_turn_end = bool(
            info.get("end_of_turn") or info.get("turn_end") or meta_info.get("end_of_turn") or meta_info.get("turn_end")
        )
        turn_end = explicit_turn_end or (turn_eos_id is not None and turn_eos_id in pending_ids)
        if state is None and not pending_ids:
            # No turn open and nothing new to speak: nothing to synthesize.
            yield self._empty_audio_chunk(), True
            return

        self._maybe_compile_tts_model()
        tts = self.tts_obj
        if not hasattr(tts.model.config, "rope_theta"):
            tts.model.config.rope_theta = 10000.0
        if not callable(getattr(tts, "generate_chunk", None)):
            logger.warning("4.5 Talker duplex streaming: MiniCPMTTS.generate_chunk unavailable")
            yield self._empty_audio_chunk(), True
            return
        sampling_params = self._build_tts_sampling_params()
        if sampling_params is None:
            logger.warning("4.5 Talker duplex streaming: sampling params unavailable")
            yield self._empty_audio_chunk(), True
            return
        chunk_size = self._tts_runtime_config().streaming_generator_chunk
        if chunk_size <= 0:
            raise ValueError("MiniCPM-o 4.5 TTS streaming generator chunk must be positive")

        if state is None:
            ref_audio = codes_info.get("ref", info.get("ref_audio"))
            ref_audio_sr = meta_info.get("ref_audio_sr", info.get("ref_audio_sr"))
            prompt_wav_path, temp_prompt_wav_path = self._resolve_prompt_wav_path(ref_audio, ref_audio_sr)
            self._begin_turn_vocoder_cache(prompt_wav_path)
            state = _TalkerTurnState(
                prompt_wav_path,
                temp_prompt_wav_path,
                epoch=stream_epoch,
                turn_id=stream_turn_id,
            )
            self._talker_turn_states[key] = state

        if pending_ids:
            pending_hidden = (
                tts_hidden_states[consumed:]
                if isinstance(tts_hidden_states, list)
                else torch.as_tensor(tts_hidden_states)[consumed:]
            )
            cond_ids, cond_hidden = self._normalize_tts_handoff_tensors(pending_ids, pending_hidden)
            condition = self._build_tts_condition_embeds(cond_ids, cond_hidden).unsqueeze(0)
        else:
            emb_dim = int(tts.emb_text.weight.shape[1])
            condition = tts.emb_text.weight.new_zeros((1, 0, emb_dim))
        audio_bos = tts.emb_text(
            torch.tensor(
                [tts.audio_bos_token_id],
                dtype=torch.long,
                device=tts.emb_text.weight.device,
            )
        ).unsqueeze(0)
        condition = torch.cat([condition, audio_bos], dim=1)
        profile_enabled = os.environ.get("MINICPMO45_PROFILE_LOGS") == "1"
        if profile_enabled:
            logger.info(
                "4.5 Talker duplex unit: key=%s pending_tokens=%d turn_end=%s t2w_buffer=%d",
                key,
                len(pending_ids),
                turn_end,
                len(state.token2wav_buffer),
            )

        cond_dump_dir = os.environ.get("MINICPMO45_DUMP_TTS_COND")
        if cond_dump_dir:
            try:
                if pending_ids:
                    raw_hidden = (
                        tts_hidden_states[consumed:]
                        if isinstance(tts_hidden_states, list)
                        else torch.as_tensor(tts_hidden_states)[consumed:]
                    )
                    _, hid_t = self._normalize_tts_handoff_tensors(pending_ids, raw_hidden)
                    hid_np = hid_t.cpu().numpy().astype(np.float32)
                else:
                    hid_np = np.zeros((0, 4096), dtype=np.float32)
                np.savez(
                    f"{cond_dump_dir}/unit_{time.monotonic_ns()}.npz",
                    ids=np.asarray(pending_ids, dtype=np.int64),
                    hidden=hid_np,
                    turn_end=np.asarray([int(bool(turn_end))]),
                )
            except Exception:
                logger.exception("MiniCPM-o 4.5 tts condition dump failed")

        unit_t0 = time.perf_counter()
        gen_t0 = time.perf_counter()
        max_token_per_chunk = chunk_size + 1
        min_token_per_chunk = 0 if turn_end else max_token_per_chunk
        force_flush = False
        if state.text_start_pos == 0:
            min_token_per_chunk = 0
            force_flush = True
        eos_token = torch.tensor(
            [tts.config.num_audio_tokens - 1],
            dtype=torch.long,
            device=tts.emb_text.weight.device,
        )
        temperature = torch.tensor(
            [float(sampling_params.temperature)],
            dtype=torch.float,
            device=tts.emb_text.weight.device,
        )
        new_tokens, past_key_values = tts.generate_chunk(
            inputs_embeds=condition,
            temperature=temperature,
            repetition_penalty=sampling_params.repetition_penalty,
            eos_token=eos_token,
            force_no_stop=False,
            max_new_token=max_token_per_chunk,
            min_new_tokens=min_token_per_chunk,
            past_key_values=state.past_key_values,
            logits_processors=None,
            text_start_pos=state.text_start_pos,
        )
        generate_ms = (time.perf_counter() - gen_t0) * 1000
        if turn_end:
            state.past_key_values = None
            state.text_start_pos = 0
        else:
            state.past_key_values = past_key_values
            state.text_start_pos += int(condition.shape[1]) + int(new_tokens.shape[1])
        vocode_ms = 0.0
        voc_t0 = time.perf_counter()
        waveforms = self._native_duplex_vocode_tokens(
            state,
            new_tokens,
            turn_end=turn_end,
            force_flush=force_flush,
            chunk_size=chunk_size,
        )
        self._talker_consumed_tokens[key] = len(ids_list)
        vocode_ms = (time.perf_counter() - voc_t0) * 1000
        for waveform in waveforms:
            if profile_enabled:
                logger.info(
                    "4.5 Talker duplex unit timing: key=%s generate_ms=%.1f vocode_ms=%.1f total_ms=%.1f",
                    key,
                    generate_ms,
                    vocode_ms,
                    (time.perf_counter() - unit_t0) * 1000,
                )
            yield waveform, False
        if profile_enabled:
            logger.info(
                "4.5 Talker duplex unit end: key=%s turn_end=%s generate_ms=%.1f vocode_ms=%.1f total_ms=%.1f",
                key,
                turn_end,
                generate_ms,
                vocode_ms,
                (time.perf_counter() - unit_t0) * 1000,
            )
        if turn_end:
            self._close_turn_state(key)
            yield self._empty_audio_chunk(), True
            return
        yield self._empty_audio_chunk(), True

    def _create_stream_gen(self, info: dict[str, Any]):
        """Yield waveform chunks from MiniCPM-o remote-code TTS streaming.

        This is the real vLLM streaming path: each yielded tensor is returned
        through one scheduler step. The older streaming probe still concatenates
        chunks inside generate_speech(), so it cannot improve API TTFA.
        """
        if info.get("minicpmo45_native_duplex") is True:
            yield from self._create_native_duplex_stream_gen(info)
            return
        tts_token_ids, tts_hidden_states = self._extract_tts_handoff(info)
        codes_info = info.get("codes")
        meta_info = info.get("meta")
        if not isinstance(codes_info, dict):
            codes_info = {}
        if not isinstance(meta_info, dict):
            meta_info = {}

        ref_audio = codes_info.get("ref", info.get("ref_audio"))
        ref_audio_sr = meta_info.get("ref_audio_sr", info.get("ref_audio_sr"))

        if tts_token_ids is None or tts_hidden_states is None:
            logger.warning("4.5 Talker streaming: missing tts_token_ids or tts_hidden_states")
            yield self._empty_audio_chunk(), True
            return
        tts_token_ids, tts_hidden_states = self._normalize_tts_handoff_tensors(
            tts_token_ids,
            tts_hidden_states,
        )

        self._lazy_init_tts()
        if not hasattr(self, "tts_obj") or self.tts_obj is None:
            logger.warning("4.5 Talker streaming: tts_obj not initialized")
            yield self._empty_audio_chunk(), True
            return
        if self.audio_tokenizer is None:
            logger.warning("4.5 Talker streaming: audio_tokenizer not initialized")
            yield self._empty_audio_chunk(), True
            return

        generator_cls = getattr(self, "_tts_streaming_generator_cls", None)
        if generator_cls is None or self._tts_gen_logits is None:
            logger.warning("4.5 Talker streaming: remote-code TTSStreamingGenerator unavailable")
            waveform = self.generate_speech(
                tts_token_ids,
                tts_hidden_states,
                ref_audio=ref_audio,
                ref_audio_sr=ref_audio_sr,
            )
            if waveform is None:
                yield self._empty_audio_chunk(), True
            else:
                yield torch.as_tensor(waveform, dtype=torch.float32).reshape(-1).cpu().contiguous(), True
            return

        self._maybe_compile_tts_model()
        tts = self.tts_obj
        if not hasattr(tts.model.config, "rope_theta"):
            tts.model.config.rope_theta = 10000.0

        profile_enabled = os.environ.get("MINICPMO45_PROFILE_LOGS") == "1"
        total_t0 = time.perf_counter()
        tts_embeds = self._build_tts_condition_embeds(tts_token_ids, tts_hidden_states)
        num_text = int(tts_token_ids.shape[-1]) if tts_token_ids.ndim > 0 else 0
        min_new_token, max_new_token = self._max_tts_tokens_for_text(num_text)
        sampling_params = self._build_tts_sampling_params()
        if sampling_params is None:
            logger.warning("4.5 Talker streaming: sampling params unavailable")
            yield self._empty_audio_chunk(), True
            return

        logits_warpers, logits_processors = self._tts_gen_logits(
            num_code=tts.config.num_audio_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )
        eos_token = torch.tensor([tts.config.num_audio_tokens - 1], dtype=torch.long, device=tts.emb_text.weight.device)
        chunk_size = self._tts_runtime_config().streaming_generator_chunk
        if chunk_size <= 0:
            raise ValueError("MiniCPM-o 4.5 TTS streaming generator chunk must be positive")

        tts_streaming_generator = generator_cls(
            model=tts,
            temperature=sampling_params.temperature,
            eos_token=eos_token,
            chunk_size=chunk_size,
            logits_processors=logits_processors,
            logits_warpers=logits_warpers,
        )

        prompt_wav_path, temp_prompt_wav_path = self._resolve_prompt_wav_path(ref_audio, ref_audio_sr)
        stream_cache = hift_cache_dict = None
        import torchaudio

        _orig_save = torchaudio.save

        def _patched_save(uri, src, sample_rate, **kw):
            kw.pop("backend", None)
            if hasattr(uri, "write"):
                sf.write(uri, src.cpu().numpy().T, sample_rate, format="WAV")
                return
            return _orig_save(uri, src, sample_rate, backend="soundfile", **kw)

        first_audio_ms = None
        num_chunks = 0
        num_tokens = 0
        generate_ms = 0.0
        vocoder_ms = 0.0
        yielded_any = False
        try:
            torchaudio.save = _patched_save
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                stream_cache, hift_cache_dict = self.audio_tokenizer.set_stream_cache(prompt_wav_path)
            finally:
                torch.set_default_dtype(prev_dtype)
                torchaudio.save = _orig_save
            self.audio_tokenizer.stream_cache = stream_cache
            self.audio_tokenizer.hift_cache_dict = hift_cache_dict
            token_iter = tts_streaming_generator.generate_with_buffer(
                condition=tts_embeds.unsqueeze(0),
                text_finished=True,
                max_new_token=max_new_token,
            )
            while True:
                iter_t0 = time.perf_counter()
                try:
                    audio_token_chunk, is_last = next(token_iter)
                except StopIteration:
                    break
                generate_ms += (time.perf_counter() - iter_t0) * 1000
                if audio_token_chunk is None:
                    break

                token_list = audio_token_chunk.reshape(-1).detach().cpu().tolist()
                if not token_list:
                    if is_last:
                        yield self._empty_audio_chunk(), True
                        yielded_any = True
                        break
                    continue

                vocoder_t0 = time.perf_counter()
                autocast_context, token2wav_autocast = self._token2wav_autocast_context()
                torchaudio.save = _patched_save
                prev_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                try:
                    with autocast_context:
                        wav_np = self.audio_tokenizer.stream(
                            token_list,
                            prompt_wav_path,
                            last_chunk=bool(is_last),
                            return_waveform=True,
                        )
                finally:
                    torch.set_default_dtype(prev_dtype)
                    torchaudio.save = _orig_save
                vocoder_ms += (time.perf_counter() - vocoder_t0) * 1000
                chunk = torch.as_tensor(np.asarray(wav_np).reshape(-1), dtype=torch.float32).cpu().contiguous()
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - total_t0) * 1000
                num_chunks += 1
                num_tokens += len(token_list)
                yielded_any = True
                if profile_enabled:
                    logger.info(
                        "generate_speech stream_chunk: chunk=%d samples=%d tokens=%d is_last=%s "
                        "first_audio_ms=%.3f token2wav_autocast=%s",
                        num_chunks,
                        chunk.numel(),
                        len(token_list),
                        bool(is_last),
                        -1.0 if first_audio_ms is None else first_audio_ms,
                        token2wav_autocast,
                    )
                yield chunk, bool(is_last)
                if is_last:
                    break
        finally:
            torchaudio.save = _orig_save
            self.audio_tokenizer.stream_cache = None
            self.audio_tokenizer.hift_cache_dict = {}
            if temp_prompt_wav_path and not self._is_cached_ref_audio_prompt_wav(temp_prompt_wav_path):
                try:
                    os.unlink(temp_prompt_wav_path)
                except OSError:
                    pass
            if profile_enabled:
                logger.info(
                    "generate_speech stream_profile: text_tokens=%d audio_tokens=%d chunks=%d "
                    "tts_generate_ms=%.3f vocoder_ms=%.3f first_audio_ms=%.3f total_ms=%.3f "
                    "min_new_token=%d max_new_token=%d token2wav_n_timesteps=%d",
                    num_text,
                    num_tokens,
                    num_chunks,
                    generate_ms,
                    vocoder_ms,
                    -1.0 if first_audio_ms is None else first_audio_ms,
                    (time.perf_counter() - total_t0) * 1000,
                    min_new_token,
                    max_new_token,
                    getattr(self, "_token2wav_n_timesteps", 10),
                )

        if not yielded_any:
            yield self._empty_audio_chunk(), True

    def _move_tts_modules_to_cuda(self) -> torch.dtype:
        target_dtype = self._target_tts_dtype()
        if target_dtype is torch.float32:
            self.tts_obj = self.tts_obj.to("cuda")
            logger.info("Moved MiniCPM-o 4.5 TTS object to cuda dtype=%s", target_dtype)
            return target_dtype

        for module_name in (
            "emb_text",
            "model",
            "projector_spk",
            "projector_semantic",
            "emb_code",
            "head_code",
        ):
            module = getattr(self.tts_obj, module_name, None)
            if module is not None:
                module.to(device="cuda", dtype=target_dtype)
        logger.info("Moved MiniCPM-o 4.5 TTS AR modules to cuda dtype=%s", target_dtype)
        return target_dtype

    def _maybe_compile_tts_model(self, *, allow_module_target: bool = True) -> None:
        cfg = self._tts_runtime_config()
        if not cfg.enable_torch_compile:
            return
        if getattr(self, "_tts_torch_compile_applied", False):
            return

        # `reduce-overhead` enables Inductor CUDA Graph capture and currently
        # fails on this dynamic decode loop with overwritten CUDAGraph outputs.
        # `default` still compiles/fuses the decoder and has shown steady-state
        # wins after the first compile warmup.
        mode = cfg.torch_compile_mode
        fullgraph = cfg.torch_compile_fullgraph
        dynamic = cfg.torch_compile_dynamic
        target = cfg.torch_compile_target
        if target not in ("module", "forward"):
            raise ValueError("MiniCPM-o 4.5 TTS torch compile target must be 'module' or 'forward'")
        if target == "module" and not allow_module_target:
            logger.warning(
                "Skipping MiniCPM-o 4.5 module-level torch.compile during load_weights; "
                "load-time compile requires a forward-level target."
            )
            return
        try:
            if target == "forward":
                self.tts_obj.model.forward = torch.compile(
                    self.tts_obj.model.forward,
                    mode=mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                )
            else:
                self.tts_obj.model = torch.compile(
                    self.tts_obj.model,
                    mode=mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                )
            self._tts_torch_compile_applied = True
            logger.info(
                "Enabled experimental torch.compile for MiniCPM-o 4.5 TTS model "
                "(target=%s, mode=%s, fullgraph=%s, dynamic=%s)",
                target,
                mode,
                fullgraph,
                dynamic,
            )
        except Exception as exc:
            logger.warning("Failed to enable MiniCPM-o 4.5 TTS torch.compile: %s", exc, exc_info=True)
            if cfg.torch_compile_strict:
                raise

    def _maybe_warmup_tts_compile(self) -> None:
        cfg = self._tts_runtime_config()
        if not cfg.enable_compile_warmup:
            return
        if getattr(self, "_tts_compile_warmup_done", False):
            return
        if not getattr(self, "_tts_torch_compile_applied", False):
            return

        tts = self.tts_obj
        hidden_size = int(getattr(tts.config, "hidden_size", tts.emb_text.weight.shape[-1]))
        seq_lens = list(cfg.compile_warmup_seq_lens)
        decode_steps = cfg.compile_warmup_decode_steps
        use_generate = cfg.compile_warmup_generate
        if not seq_lens or any(seq_len <= 0 for seq_len in seq_lens) or decode_steps < 0:
            raise ValueError(
                "MiniCPM-o 4.5 TTS compile warmup seq_lens must be positive and decode_steps must be non-negative"
            )

        device = tts.emb_text.weight.device
        dtype = tts.emb_text.weight.dtype
        warmup_t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                for seq_len in seq_lens:
                    inputs_embeds = torch.zeros(1, seq_len, hidden_size, device=device, dtype=dtype)
                    if use_generate:
                        eos_token = torch.tensor([tts.config.num_audio_tokens - 1], dtype=torch.long, device=device)
                        generate_kwargs = {
                            "inputs_embeds": inputs_embeds,
                            "eos_token": eos_token,
                            "force_no_stop": True,
                            "max_new_token": max(1, decode_steps),
                            "min_new_token": max(1, decode_steps),
                            "show_tqdm": False,
                        }
                        sampling_params = self._build_tts_sampling_params()
                        if sampling_params is not None:
                            generate_kwargs["sampling_params"] = sampling_params
                        _ = tts.generate(**generate_kwargs)
                        continue

                    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
                    outputs = tts.model(
                        position_ids=position_ids,
                        cache_position=position_ids,
                        past_key_values=None,
                        inputs_embeds=inputs_embeds,
                        attention_mask=None,
                        use_cache=True,
                        output_attentions=False,
                    )
                    past_key_values = outputs.past_key_values
                    for step in range(decode_steps):
                        step_pos = torch.tensor([[seq_len + step]], dtype=torch.long, device=device)
                        step_embeds = torch.zeros(1, 1, hidden_size, device=device, dtype=dtype)
                        outputs = tts.model(
                            position_ids=step_pos,
                            cache_position=step_pos,
                            past_key_values=past_key_values,
                            inputs_embeds=step_embeds,
                            attention_mask=None,
                            use_cache=True,
                            output_attentions=False,
                        )
                        past_key_values = outputs.past_key_values
                if device.type == "cuda":
                    torch.accelerator.synchronize(device)
            self._tts_compile_warmup_done = True
            logger.info(
                "MiniCPM-o 4.5 TTS compile warmup finished: seq_lens=%s decode_steps=%d generate=%s total_ms=%.3f",
                seq_lens,
                decode_steps,
                use_generate,
                (time.perf_counter() - warmup_t0) * 1000,
            )
        except Exception as exc:
            logger.warning("MiniCPM-o 4.5 TTS compile warmup failed: %s", exc, exc_info=True)
            if cfg.torch_compile_strict:
                raise

    def _generate_speech_streaming_probe(
        self,
        *,
        tts_embeds: torch.Tensor,
        eos_token: torch.Tensor,
        max_new_token: int,
        sampling_params,
        prompt_wav_path: str | None,
    ) -> tuple[np.ndarray, int, int, int, float, float, float | None] | None:
        """Probe MiniCPM-o's remote-code streaming generator.

        This is intentionally opt-in and still returns a single concatenated
        waveform to the current engine. It measures whether the model can
        produce early audio chunks before we restructure vLLM-Omni to stream
        those chunks through the API.
        """
        generator_cls = getattr(self, "_tts_streaming_generator_cls", None)
        if generator_cls is None or self._tts_gen_logits is None:
            logger.warning("MiniCPM-o 4.5 streaming probe unavailable in remote code")
            return None

        tts = self.tts_obj
        chunk_size = self._tts_runtime_config().streaming_generator_chunk
        if chunk_size <= 0:
            raise ValueError("MiniCPM-o 4.5 TTS streaming generator chunk must be positive")

        logits_warpers, logits_processors = self._tts_gen_logits(
            num_code=tts.config.num_audio_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )
        if not hasattr(tts.model.config, "rope_theta"):
            tts.model.config.rope_theta = 10000.0
        tts_streaming_generator = generator_cls(
            model=tts,
            temperature=sampling_params.temperature,
            eos_token=eos_token,
            chunk_size=chunk_size,
            logits_processors=logits_processors,
            logits_warpers=logits_warpers,
        )

        stream_cache, hift_cache_dict = self.audio_tokenizer.set_stream_cache(prompt_wav_path)
        self.audio_tokenizer.stream_cache = stream_cache
        self.audio_tokenizer.hift_cache_dict = hift_cache_dict

        pieces: list[np.ndarray] = []
        total_tokens = 0
        num_chunks = 0
        first_audio_ms: float | None = None
        generate_ms = 0.0
        vocoder_ms = 0.0
        total_t0 = time.perf_counter()
        try:
            token_iter = tts_streaming_generator.generate_with_buffer(
                condition=tts_embeds.unsqueeze(0),
                text_finished=True,
                max_new_token=max_new_token,
            )
            while True:
                iter_t0 = time.perf_counter()
                try:
                    audio_token_chunk, is_last = next(token_iter)
                except StopIteration:
                    break
                generate_ms += (time.perf_counter() - iter_t0) * 1000
                if audio_token_chunk is None:
                    break

                token_list = audio_token_chunk.reshape(-1).detach().cpu().tolist()
                if not token_list:
                    if is_last:
                        break
                    continue

                num_chunks += 1
                total_tokens += len(token_list)
                vocoder_t0 = time.perf_counter()
                autocast_context, _ = self._token2wav_autocast_context()
                with autocast_context:
                    wav_np = self.audio_tokenizer.stream(
                        token_list,
                        prompt_wav_path,
                        last_chunk=bool(is_last),
                        return_waveform=True,
                    )
                vocoder_ms += (time.perf_counter() - vocoder_t0) * 1000
                pieces.append(np.asarray(wav_np).reshape(-1))
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - total_t0) * 1000
                if is_last:
                    break
        finally:
            self.audio_tokenizer.stream_cache = None
            self.audio_tokenizer.hift_cache_dict = {}

        if not pieces:
            waveform = np.zeros((0,), dtype=np.float32)
        else:
            waveform = np.concatenate(pieces, axis=0).astype(np.float32)
        return waveform, 24000, total_tokens, num_chunks, generate_ms, vocoder_ms, first_audio_ms

    @torch.inference_mode()
    def _generate_tts_tokens_fast(
        self,
        inputs_embeds: torch.Tensor,
        eos_token: torch.Tensor,
        max_new_token: int,
        sampling_params,
    ):
        """Single-codebook fast path for MiniCPM-o 4.5 TTS token generation.

        The upstream remote-code loop is generic over num_vq, so for the 4.5
        checkpoint's num_vq=1 it still allocates 4D logits, projects every
        condition token on the first step, and permutes/reshapes tensors every
        decode step. This path keeps the same model, cache, logits processors,
        and sampling semantics, but operates directly on the last hidden state.
        """
        tts = self.tts_obj
        decode_profile_enabled = os.environ.get("MINICPMO45_PROFILE_LOGS") == "1"
        num_vq = getattr(tts, "num_vq", None)
        if num_vq != 1 or self._tts_gen_logits is None:
            if decode_profile_enabled:
                logger.info(
                    "generate_speech fast_decode_profile: skipped fast path (num_vq=%s, has_gen_logits=%s)",
                    num_vq,
                    self._tts_gen_logits is not None,
                )
            return None

        device = inputs_embeds.device
        temperature_value = float(getattr(sampling_params, "temperature", 0.8))
        logits_warpers, logits_processors = self._tts_gen_logits(
            num_code=tts.config.num_audio_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )

        finish = torch.zeros(inputs_embeds.shape[0], device=device, dtype=torch.bool)
        condition_length = inputs_embeds.shape[1]
        new_tokens = torch.empty(
            inputs_embeds.shape[0],
            max_new_token,
            1,
            device=device,
            dtype=torch.long,
        )
        static_kv = self._tts_runtime_config().enable_static_kv
        prealloc_kv = self._tts_runtime_config().enable_prealloc_kv
        if static_kv:
            from transformers.cache_utils import StaticCache

            past_key_values = StaticCache(
                config=tts.model.config,
                max_cache_len=condition_length + max_new_token,
            )
            cache_mode = "static"
        elif prealloc_kv:
            num_layers = int(getattr(tts.model.config, "num_hidden_layers", 0))
            past_key_values = _PreallocatedKVCache(
                num_layers=num_layers,
                max_cache_len=condition_length + max_new_token,
            )
            cache_mode = "prealloc"
        else:
            past_key_values = None
            cache_mode = "dynamic"
        prefill_position_ids = torch.arange(condition_length, dtype=torch.long, device=device).unsqueeze(0)
        decode_position_ids = torch.arange(
            condition_length,
            condition_length + max_new_token,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        min_new_token = self._tts_runtime_config().min_new_tokens
        last_t = 0
        eos_token_id = int(eos_token.item())
        model_ms = 0.0
        head_ms = 0.0
        processors_ms = 0.0
        sample_ms = 0.0

        def profile_mark():
            if decode_profile_enabled:
                torch.accelerator.synchronize(device)
            return time.perf_counter()

        def head_context():
            return self._tts_parametrize.cached() if self._tts_parametrize is not None else nullcontext()

        for t in range(max_new_token):
            audio_bos = t == 0
            if audio_bos:
                step_embeds = inputs_embeds
                position_ids = prefill_position_ids
            else:
                step_embeds = tts.emb_code[0](new_tokens[:, t - 1 : t, 0])
                position_ids = decode_position_ids[:, t - 1 : t]

            t0 = profile_mark()
            outputs = tts.model(
                position_ids=position_ids,
                cache_position=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=step_embeds,
                attention_mask=None,
                use_cache=True,
                output_attentions=False,
            )
            if decode_profile_enabled:
                model_ms += (profile_mark() - t0) * 1000
            past_key_values = outputs.past_key_values
            hidden_state = outputs.last_hidden_state[:, -1]

            t0 = profile_mark()
            with head_context():
                logits = tts.head_code[0](hidden_state).float()
            logits.div_(temperature_value)
            if decode_profile_enabled:
                head_ms += (profile_mark() - t0) * 1000

            t0 = profile_mark()
            if not audio_bos:
                logits_token = new_tokens[:, :t, 0]
                for logits_processor in logits_processors:
                    logits = logits_processor(logits_token, logits)
                for logits_warper in logits_warpers:
                    logits = logits_warper(logits_token, logits)

            if t < min_new_token:
                logits[:, eos_token_id] = -torch.inf
            if decode_profile_enabled:
                processors_ms += (profile_mark() - t0) * 1000

            t0 = profile_mark()
            scores = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)
            if decode_profile_enabled:
                sample_ms += (profile_mark() - t0) * 1000
            idx_next = idx_next.view(-1, 1)
            finish.logical_or_(idx_next.eq(eos_token_id).any(1))
            new_tokens[:, t] = idx_next
            last_t = t

            if t == 0 and finish.any():
                break
            if finish.all():
                break

        generated_input_ids = new_tokens[:, :last_t, :]
        if decode_profile_enabled:
            steps = max(last_t + 1, 1)
            logger.info(
                "generate_speech fast_decode_profile: steps=%d model_ms=%.3f head_ms=%.3f "
                "processors_ms=%.3f sample_ms=%.3f per_step_ms=%.3f cache=%s",
                steps,
                model_ms,
                head_ms,
                processors_ms,
                sample_ms,
                (model_ms + head_ms + processors_ms + sample_ms) / steps,
                cache_mode,
            )
        output_cls = tts.generate.__globals__.get("MiniCPMTTSGenerationOutput")
        if output_cls is None:
            return None
        return output_cls(
            new_ids=generated_input_ids,
            audio_input_ids=None,
            past_key_values=None,
            past_input_ids=None,
            finished=finish.all(),
        )

    def generate_speech(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
        ref_audio=None,
        ref_audio_sr: int | None = None,
    ) -> torch.Tensor | np.ndarray | None:
        """Run full 4.5 TTS pipeline using original MiniCPMTTS.generate."""
        self._lazy_init_tts()
        if not hasattr(self, "tts_obj") or self.tts_obj is None:
            logger.warning("generate_speech: tts_obj not initialized")
            return None

        self._maybe_compile_tts_model()
        tts = self.tts_obj
        device = tts.emb_text.weight.device
        dtype = tts.emb_text.weight.dtype
        profile_enabled = os.environ.get("MINICPMO45_PROFILE_LOGS") == "1"
        total_t0 = time.perf_counter()

        llm_embeds = tts.emb_text(tts_token_ids.to(device))
        hidden_embeds = tts.projector_semantic(tts_hidden_states.to(device=device, dtype=dtype))
        if getattr(tts.config, "normalize_projected_hidden", False):
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        tts_embeds = llm_embeds + hidden_embeds

        text_eos = tts.emb_text(torch.tensor([tts.config.text_eos_token_id], device=device, dtype=torch.long))
        audio_bos = tts.emb_text(torch.tensor([tts.audio_bos_token_id], device=device, dtype=torch.long))
        spk_embeds = torch.zeros(0, tts.config.hidden_size, device=device, dtype=tts_embeds.dtype)

        inputs_embeds = torch.cat([spk_embeds, tts_embeds, text_eos, audio_bos], dim=0).unsqueeze(0)
        prep_ms = (time.perf_counter() - total_t0) * 1000
        if profile_enabled:
            logger.info("generate_speech: inputs_embeds shape=%s", list(inputs_embeds.shape))

        # Scale max_new_token with input text length. A fixed 2048-token floor
        # can turn an EOS miss on a very short response into ~82s of audio and
        # ~18s E2E latency. Keep a conservative short-text floor while bounding
        # the tail.
        num_text = int(tts_token_ids.shape[-1]) if tts_token_ids.ndim > 0 else 0
        min_new_token, max_new_token = self._max_tts_tokens_for_text(num_text)

        eos_token = torch.tensor([tts.config.num_audio_tokens - 1], dtype=torch.long, device=device)
        sampling_params = self._build_tts_sampling_params()
        generate_t0 = time.perf_counter()
        generate_kwargs = {
            "inputs_embeds": inputs_embeds,
            "eos_token": eos_token,
            "max_new_token": max_new_token,
            "min_new_token": min_new_token,
            "show_tqdm": False,
        }
        if sampling_params is not None:
            generate_kwargs["sampling_params"] = sampling_params

        if self.audio_tokenizer is None:
            logger.warning("No audio_tokenizer")
            return None

        model_path = self.vllm_config.model_config.model
        default_ref = os.path.join(model_path, "assets", "HT_ref_audio.wav")
        temp_prompt_wav_path = self._write_ref_audio_prompt_wav(ref_audio, ref_audio_sr)
        prompt_wav_path = temp_prompt_wav_path or (default_ref if os.path.exists(default_ref) else None)

        try:
            streaming_probe = self._tts_runtime_config().enable_streaming_probe
            if streaming_probe and sampling_params is not None:
                import torchaudio

                _orig_save = torchaudio.save

                def _patched_save(uri, src, sample_rate, **kw):
                    kw.pop("backend", None)
                    if hasattr(uri, "write"):
                        sf.write(uri, src.cpu().numpy().T, sample_rate, format="WAV")
                        return
                    return _orig_save(uri, src, sample_rate, backend="soundfile", **kw)

                torchaudio.save = _patched_save
                prev_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                try:
                    _, token2wav_autocast = self._token2wav_autocast_context()
                    probe_result = self._generate_speech_streaming_probe(
                        tts_embeds=tts_embeds,
                        eos_token=eos_token,
                        max_new_token=max_new_token,
                        sampling_params=sampling_params,
                        prompt_wav_path=prompt_wav_path,
                    )
                finally:
                    torch.set_default_dtype(prev_dtype)
                    torchaudio.save = _orig_save

                if probe_result is not None:
                    waveform, sr, num_tokens, num_chunks, generate_ms, vocoder_ms, first_audio_ms = probe_result
                    if profile_enabled:
                        logger.info("generate_speech: waveform %d samples, sr=%d", waveform.shape[0], sr)
                        logger.info(
                            "generate_speech streaming_probe_profile: text_tokens=%d audio_tokens=%d "
                            "chunks=%d prep_ms=%.3f tts_generate_ms=%.3f vocoder_ms=%.3f "
                            "first_audio_ms=%.3f total_ms=%.3f min_new_token=%d max_new_token=%d "
                            "token2wav_n_timesteps=%d token2wav_autocast=%s",
                            num_text,
                            num_tokens,
                            num_chunks,
                            prep_ms,
                            generate_ms,
                            vocoder_ms,
                            -1.0 if first_audio_ms is None else first_audio_ms,
                            (time.perf_counter() - total_t0) * 1000,
                            min_new_token,
                            max_new_token,
                            getattr(self, "_token2wav_n_timesteps", 10),
                            token2wav_autocast,
                        )
                    return waveform

            # Keep this experimental path opt-in: remote A/B showed it does not
            # reduce tts_generate_ms reliably yet, despite lower Python-side shape
            # churn.
            fast_generate = self._tts_runtime_config().enable_fast_generate
            if fast_generate and profile_enabled:
                logger.info(
                    "generate_speech fast_decode_profile: requested fast path "
                    "(has_sampling_params=%s, has_gen_logits=%s, num_vq=%s)",
                    sampling_params is not None,
                    self._tts_gen_logits is not None,
                    getattr(tts, "num_vq", None),
                )
            outputs = None
            if fast_generate and sampling_params is not None:
                outputs = self._generate_tts_tokens_fast(
                    inputs_embeds=inputs_embeds,
                    eos_token=eos_token,
                    max_new_token=max_new_token,
                    sampling_params=sampling_params,
                )
                if outputs is not None and not bool(outputs.finished):
                    logger.warning(
                        "generate_speech: fast TTS generate hit max_new_token without EOS; "
                        "falling back to remote-code generate"
                    )
                    outputs = None
            if outputs is None:
                outputs = tts.generate(**generate_kwargs)
            generate_ms = (time.perf_counter() - generate_t0) * 1000
            generated_tokens = outputs.new_ids.squeeze(-1)
            if profile_enabled:
                logger.info(
                    "generate_speech: generated %d audio tokens (cap=%d, text_tokens=%d)",
                    generated_tokens.shape[-1],
                    max_new_token,
                    num_text,
                )

            import torchaudio

            _orig_save = torchaudio.save

            def _patched_save(uri, src, sample_rate, **kw):
                kw.pop("backend", None)
                if hasattr(uri, "write"):
                    sf.write(uri, src.cpu().numpy().T, sample_rate, format="WAV")
                    return
                return _orig_save(uri, src, sample_rate, backend="soundfile", **kw)

            torchaudio.save = _patched_save
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            vocoder_t0 = time.perf_counter()
            token2wav_direct_used = False
            try:
                autocast_context, token2wav_autocast = self._token2wav_autocast_context()
                with autocast_context:
                    num_tokens = int(generated_tokens.shape[-1])

                    # For long outputs, the one-shot vocoder path
                    # (Token2wav.__call__ -> flow.inference) runs full O(N^2) self-
                    # attention over all audio tokens and OOMs on a 24GB card once
                    # N exceeds a few thousand (e.g. 4964 tokens needs ~3GiB for a
                    # single attention matmul). Switch to the chunked / streaming
                    # vocoder (set_stream_cache + stream) which truncates the flow
                    # attention caches to prompt_len + 100 steps on every chunk,
                    # keeping peak memory bounded regardless of total length.
                    STREAM_THRESHOLD = self._tts_runtime_config().streaming_vocoder_threshold  # ~100s @ 25Hz
                    CHUNK_SIZE = self._tts_runtime_config().streaming_vocoder_chunk  # ~2s per chunk
                    MIN_TAIL = 6  # must exceed flow.pre_lookahead_len (typically 3)

                    if num_tokens <= STREAM_THRESHOLD:
                        if self._should_use_direct_token2wav() and token2wav_autocast == "off":
                            try:
                                waveform, sr = self._run_token2wav_direct(generated_tokens, prompt_wav_path)
                                token2wav_direct_used = True
                            except Exception as exc:
                                logger.warning(
                                    "MiniCPM-o 4.5 direct Token2wav path failed; falling back to WAV path: %s",
                                    exc,
                                    exc_info=True,
                                )
                                token_list = generated_tokens.squeeze(0).tolist()
                                self._reset_token2wav_cache_if_needed(prompt_wav_path)
                                wav_bytes = self.audio_tokenizer(token_list, prompt_wav_path)
                                waveform, sr = sf.read(io.BytesIO(wav_bytes))
                                waveform = waveform.astype(np.float32)
                        else:
                            token_list = generated_tokens.squeeze(0).tolist()
                            self._reset_token2wav_cache_if_needed(prompt_wav_path)
                            wav_bytes = self.audio_tokenizer(token_list, prompt_wav_path)
                            waveform, sr = sf.read(io.BytesIO(wav_bytes))
                            waveform = waveform.astype(np.float32)
                    else:
                        token_list = generated_tokens.squeeze(0).tolist()
                        # Build chunk boundaries, merging a too-small tail into the
                        # previous chunk so every chunk satisfies MIN_TAIL.
                        boundaries = []
                        i = 0
                        while i < num_tokens:
                            end = min(i + CHUNK_SIZE, num_tokens)
                            if 0 < num_tokens - end < MIN_TAIL:
                                end = num_tokens
                            boundaries.append((i, end))
                            i = end

                        logger.info(
                            "generate_speech: streaming vocoder, %d tokens -> %d chunks (chunk=%d)",
                            num_tokens,
                            len(boundaries),
                            CHUNK_SIZE,
                        )

                        stream_cache, hift_cache_dict = self.audio_tokenizer.set_stream_cache(prompt_wav_path)
                        self.audio_tokenizer.stream_cache = stream_cache
                        self.audio_tokenizer.hift_cache_dict = hift_cache_dict

                        try:
                            pieces = []
                            for idx, (s, e) in enumerate(boundaries):
                                is_last = idx == len(boundaries) - 1
                                wav_np = self.audio_tokenizer.stream(
                                    token_list[s:e],
                                    prompt_wav_path,
                                    last_chunk=is_last,
                                    return_waveform=True,
                                )
                                pieces.append(np.asarray(wav_np).reshape(-1))
                            waveform = np.concatenate(pieces, axis=0).astype(np.float32)
                            sr = 24000
                        finally:
                            # Free per-request streaming state so the next request starts clean
                            self.audio_tokenizer.stream_cache = None
                            self.audio_tokenizer.hift_cache_dict = {}
            finally:
                vocoder_ms = (time.perf_counter() - vocoder_t0) * 1000
                torch.set_default_dtype(prev_dtype)
                torchaudio.save = _orig_save

            if profile_enabled:
                logger.info("generate_speech: waveform %d samples, sr=%d", waveform.shape[0], sr)
            if profile_enabled:
                logger.info(
                    "generate_speech profile: text_tokens=%d audio_tokens=%d prep_ms=%.3f "
                    "tts_generate_ms=%.3f vocoder_ms=%.3f total_ms=%.3f "
                    "min_new_token=%d max_new_token=%d token2wav_n_timesteps=%d "
                    "token2wav_autocast=%s token2wav_direct=%s",
                    num_text,
                    num_tokens,
                    prep_ms,
                    generate_ms,
                    vocoder_ms,
                    (time.perf_counter() - total_t0) * 1000,
                    min_new_token,
                    max_new_token,
                    getattr(self, "_token2wav_n_timesteps", 10),
                    token2wav_autocast,
                    token2wav_direct_used,
                )
            return waveform
        finally:
            if temp_prompt_wav_path and not self._is_cached_ref_audio_prompt_wav(temp_prompt_wav_path):
                try:
                    os.unlink(temp_prompt_wav_path)
                except OSError:
                    pass

    def _generate_tokens(self, inputs_embeds: torch.Tensor, max_new_token: int = 2048) -> torch.Tensor | None:
        """Autoregressive generation of audio tokens using the TTS LlamaModel."""
        device = inputs_embeds.device
        eos_token = self._num_audio_tokens - 1
        condition_length = inputs_embeds.shape[1]
        num_vq = len(self.emb_code)

        new_tokens = torch.zeros(1, max_new_token, num_vq, device=device, dtype=torch.long)
        past_key_values = None
        finished = False

        for t in range(max_new_token):
            if t == 0:
                emb = inputs_embeds
                position_ids = torch.arange(condition_length, device=device).unsqueeze(0)
            else:
                code_emb = [self.emb_code[q](new_tokens[:, t - 1 : t, q]) for q in range(num_vq)]
                emb = torch.stack(code_emb, -1).sum(-1)
                position_ids = torch.tensor([[condition_length + t - 1]], device=device)

            outputs = self.tts_model(
                inputs_embeds=emb,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            logits = torch.stack([self.head_code[q](hidden[:, -1]) for q in range(num_vq)], dim=-1)
            logits = logits.float() / 0.8

            if t < 50:
                logits[:, eos_token, :] = -float("inf")

            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(probs.view(-1, probs.shape[1]), 1).view(1, num_vq)
            new_tokens[:, t] = idx

            if (idx == eos_token).any():
                finished = True
                break

        return new_tokens[:, : t + 1 if finished else t, :]

    def _dummy_hidden_states(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shape-correct zero tensor for vllm KV cache profiling.

        vllm's gpu_model_runner._dummy_run takes forward()'s return value as
        ``hidden_states`` and does ``hidden_states[logit_indices_device]``;
        returning None on the dummy path crashes with
        ``TypeError: 'NoneType' object is not subscriptable``.
        """
        for ref in (input_ids, positions, inputs_embeds):
            if isinstance(ref, torch.Tensor):
                num_tokens = int(ref.shape[0]) if ref.ndim >= 1 else 1
                device = ref.device
                break
        else:
            num_tokens = 1
            device = current_omni_platform.get_torch_device()
        hidden_size = int(getattr(self, "_hidden_size", 768) or 768)
        return torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        additional_information=None,
        **kwargs,
    ):
        if additional_information is None:
            additional_information = {}
        if not additional_information:
            # Profile/dummy run: use it to pre-compile the per-turn vocoder
            # stream modes so the first real spoken turn does not stall.
            self._warmup_duplex_vocoder()

        tts_token_ids, tts_hidden_states = self._extract_tts_handoff(additional_information)
        tts_text = additional_information.get("llm_output_text", [""])
        if isinstance(tts_text, list):
            tts_text = tts_text[0] if tts_text else ""
        codes_info = additional_information.get("codes")
        meta_info = additional_information.get("meta")
        if not isinstance(codes_info, dict):
            codes_info = {}
        if not isinstance(meta_info, dict):
            meta_info = {}
        ref_audio = codes_info.get("ref")
        if ref_audio is None:
            ref_audio = additional_information.get("ref_audio")
        ref_audio_sr = meta_info.get("ref_audio_sr")
        if ref_audio_sr is None:
            ref_audio_sr = additional_information.get("ref_audio_sr")

        if tts_token_ids is None or tts_hidden_states is None:
            logger.warning("4.5 Talker: missing tts_token_ids or tts_hidden_states")
            self._ar_last_chunk_flags = [True]
            self._ar_turn_end_flags = [False]
            return None, None
        tts_token_ids, tts_hidden_states = self._normalize_tts_handoff_tensors(
            tts_token_ids,
            tts_hidden_states,
        )

        if self._should_stream_output(additional_information):
            request_key = self._stream_request_key(additional_information)
            input_ends_turn = self._native_duplex_input_ends_turn(additional_information)
            if request_key not in self._stream_gens:
                self._stream_gens[request_key] = self._create_stream_gen(additional_information)
            generator = self._stream_gens[request_key]
            try:
                waveform_chunk, is_last = next(generator)
            except StopIteration:
                self._stream_gens.pop(request_key, None)
                waveform_chunk = self._empty_audio_chunk()
                is_last = True
            if is_last:
                self._stream_gens.pop(request_key, None)
            self._ar_last_chunk_flags = [bool(is_last)]
            # A TTS generator also ends at ordinary chunk boundaries. Export
            # turn_end only on the terminal output for a condition that
            # actually contains the model's <|turn_eos|> decision.
            self._ar_turn_end_flags = [bool(is_last and input_ends_turn)]
            return None, waveform_chunk.reshape(-1).contiguous()

        self._ar_last_chunk_flags = [True]
        self._ar_turn_end_flags = [False]
        if os.environ.get("MINICPMO45_PROFILE_LOGS") == "1":
            logger.info("4.5 Talker: generating speech for %d tokens", tts_token_ids.shape[0])
        waveform = self.generate_speech(
            tts_token_ids,
            tts_hidden_states,
            ref_audio=ref_audio,
            ref_audio_sr=ref_audio_sr,
        )
        if waveform is not None:
            waveform_tensor = torch.as_tensor(waveform, dtype=torch.float32).detach()
            if waveform_tensor.device.type != "cpu":
                waveform_tensor = waveform_tensor.cpu()
            return waveform_tensor.reshape(-1).contiguous(), None
        return None, None

    def compute_logits(self, hidden_states, *args, **kwargs):
        device = hidden_states.device if isinstance(hidden_states, torch.Tensor) else torch.device("cuda")
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.ndim == 1:
                num_rows = 1
            else:
                num_rows = max(1, int(hidden_states.shape[0]))
        else:
            num_rows = 1
        eos_id = self._scheduler_eos_token_id()
        vocab_size = max(int(getattr(self.config, "vocab_size", eos_id + 1) or (eos_id + 1)), eos_id + 1, 3)
        safe_id = 1 if eos_id != 1 else 0
        logits = torch.full((num_rows, vocab_size), -1.0e9, dtype=torch.float32, device=device)
        flags = self._ar_last_chunk_flags
        default_is_last = bool(flags[-1]) if flags else True
        for row in range(num_rows):
            is_last = bool(flags[row]) if row < len(flags) else default_is_last
            if is_last:
                logits[row, eos_id] = 1.0e6
            else:
                logits[row, safe_id] = 1.0e6
        if os.environ.get("MINICPMO45_PROFILE_LOGS") == "1":
            logger.info(
                "4.5 Talker scheduler logits: rows=%d flags=%s default_is_last=%s eos_id=%d safe_id=%d sampled=%s",
                num_rows,
                flags,
                default_is_last,
                eos_id,
                safe_id,
                torch.argmax(logits, dim=-1).detach().cpu().tolist(),
            )
        return logits

    def sample(self, logits, sampling_metadata):
        if logits is None or logits.numel() == 0:
            return None
        sampled = torch.argmax(logits, dim=-1).to(torch.int32)
        if os.environ.get("MINICPMO45_PROFILE_LOGS") == "1":
            logger.info(
                "4.5 Talker scheduler sample: sampled=%s",
                sampled.detach().cpu().tolist(),
            )
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for req_id in finished_req_ids:
            request_key = str(req_id)
            mapped_key = getattr(self, "_talker_request_keys", {}).pop(request_key, None)
            keys = {request_key}
            if mapped_key:
                keys.add(mapped_key)
            for key in list(self._stream_gens):
                if key in keys:
                    gen = self._stream_gens.pop(key, None)
                    if gen is not None:
                        try:
                            gen.close()
                        except Exception:
                            logger.exception("MiniCPM-o 4.5 failed to close stream gen for request %s", req_id)
            for key in list(self._talker_turn_states):
                if key in keys:
                    self._close_turn_state(key)
            for key in list(self._talker_consumed_tokens):
                if key in keys:
                    self._talker_consumed_tokens.pop(key, None)

    def signal_duplex_turn(self, **kwargs: Any) -> dict[str, Any]:
        session_id = self._coerce_request_key(kwargs.get("session_id"))
        event = kwargs.get("event")
        expected_epoch = self._coerce_epoch(kwargs.get("epoch"))
        payload = kwargs.get("payload")
        expected_turn_id = self._coerce_turn_id(payload.get("turn_id")) if isinstance(payload, dict) else None
        if event in {"barge_in", "input.cancel", "response.cancel"} and session_id:
            closed = self._close_turn_state(
                session_id,
                expected_epoch=expected_epoch,
                expected_turn_id=expected_turn_id,
            )
        else:
            closed = True
        result = {
            "supported": True,
            "stage_role": "tts",
            "event": event,
        }
        if closed is False:
            result["stale_signal_ignored"] = True
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loaded = set()
        tts_weights = {}
        for k, v in weights:
            if k.startswith("tts."):
                local_name = k.replace("tts.", "", 1)
                tts_weights[local_name] = v
                loaded.add(f"tts_obj.{local_name}")

        if tts_weights and self._tts_config is not None:
            self._lazy_init_tts()
            if hasattr(self, "tts_obj") and self.tts_obj is not None:
                missing, unexpected = self.tts_obj.load_state_dict(tts_weights, strict=False)
                if missing:
                    logger.warning("TTS missing keys (%d): %s", len(missing), missing[:5])
                if unexpected:
                    logger.warning("TTS unexpected keys (%d): %s", len(unexpected), unexpected[:5])
                tts_dtype = self._move_tts_modules_to_cuda()
                if self.audio_tokenizer is not None and hasattr(self.audio_tokenizer, "to"):
                    self.audio_tokenizer.to("cuda")
                self.emb_text = self.tts_obj.emb_text
                self.projector_semantic = self.tts_obj.projector_semantic
                if self._tts_runtime_config().enable_torch_compile:
                    self._maybe_compile_tts_model(allow_module_target=False)
                    self._maybe_warmup_tts_compile()
                logger.info("Loaded %d TTS weights, moved AR modules to cuda dtype=%s", len(tts_weights), tts_dtype)

        return loaded

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        if hasattr(self, "emb_text") and self.emb_text is not None:
            return self.emb_text(input_ids)
        return torch.zeros(input_ids.shape[0], 1)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
