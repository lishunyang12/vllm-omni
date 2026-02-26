"""Chatterbox Turbo T3 AR model for vLLM-Omni.

T3 is the autoregressive text-to-speech-token stage of Chatterbox.  It uses a
GPT-2-medium backbone (24 layers, 1024 hidden, 16 heads) with custom text and
speech embeddings plus speaker conditioning.

The vLLM integration follows the same pattern as Qwen3-TTS Talker:
* ``preprocess`` builds full prompt embeddings from ``additional_information``
  (text, reference audio path, optional exaggeration).
* ``forward`` runs the GPT-2 backbone.
* ``postprocess`` caches last hidden state for the next decode step.
* ``make_omni_output`` wraps generated speech tokens for the stage connector.

Weight loading maps from Chatterbox safetensors keys (``tfmr.*``,
``text_emb.*``, ``speech_emb.*``, ``speech_head.*``, ``cond_enc.*``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2Config
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.gpt2 import GPT2Model
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_chatterbox import ChatterboxTurboConfig

logger = init_logger(__name__)

# Constants from Chatterbox
S3_SR = 16000  # S3Tokenizer input sample rate
S3_TOKEN_HOP = 640  # 25 tokens/sec
SPEECH_VOCAB_SIZE = 6561  # Excludes SOS/EOS
SOS_TOKEN = 6561
EOS_TOKEN = 6562
ENC_COND_LEN = 15 * S3_SR  # 15 seconds conditioning for VoiceEncoder


# Conditioning
@dataclasses.dataclass
class ChatterboxT3Cond:
    """Conditioning container passed to T3 during inference."""

    speaker_emb: torch.Tensor  # (1, 256) from VoiceEncoder
    cond_prompt_speech_tokens: torch.Tensor | None = None  # (1, plen)
    cond_prompt_speech_emb: torch.Tensor | None = None  # (1, plen, hidden_size)


class ChatterboxT3CondEnc(nn.Module):
    """Conditioning encoder: project speaker embedding + cond prompt speech."""

    def __init__(self, speaker_embed_size: int, hidden_size: int):
        super().__init__()
        self.speaker_proj = nn.Linear(speaker_embed_size, hidden_size)

    def forward(
        self,
        speaker_emb: torch.Tensor,
        cond_prompt_speech_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return conditioning sequence: [speaker_proj || cond_speech_emb]."""
        spk = self.speaker_proj(speaker_emb)  # (1, hidden_size)
        if spk.ndim == 2:
            spk = spk.unsqueeze(1)  # (1, 1, hidden_size)
        if cond_prompt_speech_emb is not None:
            return torch.cat([spk, cond_prompt_speech_emb], dim=1)
        return spk


# Main AR model
class ChatterboxTurboT3ForGeneration(nn.Module):
    """Chatterbox Turbo T3 — AR speech token generator for vLLM-Omni.

    Stage 0 of the Chatterbox TTS pipeline.
    """

    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        config = vllm_config.model_config.hf_config
        if not isinstance(config, ChatterboxTurboConfig):
            config = ChatterboxTurboConfig()

        self.config = config
        hidden_size = config.hidden_size

        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_embd=hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_inner=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            add_cross_attention=False,
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
        )
        # Patch hf_config so GPT2Model reads the right config.
        orig_hf_config = vllm_config.model_config.hf_config
        vllm_config.model_config.hf_config = gpt2_config
        self.model = GPT2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "tfmr"),
        )
        vllm_config.model_config.hf_config = orig_hf_config

        # Custom embeddings (separate from GPT-2 wte which we won't use).
        self.text_emb = nn.Embedding(config.vocab_size, hidden_size)
        self.speech_emb = nn.Embedding(config.speech_vocab_size, hidden_size)

        self.speech_head = ParallelLMHead(
            config.speech_vocab_size,
            hidden_size,
        )
        self.logits_processor = LogitsProcessor(config.speech_vocab_size)

        self.cond_enc = ChatterboxT3CondEnc(config.speaker_embed_size, hidden_size)

        # Valid token mask: allow [0, speech_vocab_size) plus stop token.
        speech_mask = torch.zeros((config.speech_vocab_size,), dtype=torch.bool)
        speech_mask[:SPEECH_VOCAB_SIZE] = True
        speech_mask[EOS_TOKEN] = True
        self.register_buffer("_speech_allowed_mask", speech_mask, persistent=False)

        # Lazy loaded.
        self._tokenizer = None
        self._voice_encoder = None
        self._s3_tokenizer = None

    # -------------------- vLLM required hooks --------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        logits = self.logits_processor(self.speech_head, hidden_states)
        if logits is None:
            return None
        logits = logits.masked_fill(~self._speech_allowed_mask, float("-inf"))
        return logits

    # -------------------- Omni multimodal output plumbing --------------------

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("runtime_additional_information") or []

        speech_tokens_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            st = info.get("speech_tokens")
            if isinstance(st, torch.Tensor):
                speech_tokens_list.append(st)

        if not speech_tokens_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        speech_tokens = torch.cat(speech_tokens_list, dim=0)
        span_len = int(speech_tokens.shape[0])
        hidden = hidden[:span_len]
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"speech_tokens": speech_tokens},
        )

    # -------------------- preprocess / postprocess --------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        device = input_ids.device
        if span_len <= 0:
            return input_ids, input_embeds if input_embeds is not None else self.embed_input_ids(input_ids), {}

        text_list = info_dict.get("text")
        if not isinstance(text_list, list) or not text_list or not text_list[0]:
            raise ValueError("Missing additional_information.text for Chatterbox T3.")

        if span_len > 1:
            # Prefill
            prompt_embeds_cpu = info_dict.get("t3_prompt_embeds")
            is_first_prefill = not isinstance(prompt_embeds_cpu, torch.Tensor) or prompt_embeds_cpu.ndim != 2

            if is_first_prefill:
                prompt_embeds_full = self._build_prompt_embeds(info_dict, device)
                prompt_embeds_cpu = prompt_embeds_full.detach().to("cpu").contiguous()

                info_update: dict[str, Any] = {
                    "t3_prompt_embeds": prompt_embeds_cpu,
                    "t3_prefill_offset": 0,
                }

                # Pass ref_dict to S3Gen via stage connector.
                ref_dict = info_dict.get("_ref_dict")
                if ref_dict is not None:
                    info_update["ref_dict"] = ref_dict

                take = prompt_embeds_cpu[:span_len]
                if int(take.shape[0]) < span_len:
                    pad_n = span_len - int(take.shape[0])
                    pad_rows = torch.zeros(pad_n, take.shape[-1])
                    take = torch.cat([take, pad_rows], dim=0)
                prompt_embeds = take.to(device=device, dtype=torch.bfloat16)
                info_update["t3_prefill_offset"] = span_len
            else:
                offset = int(info_dict.get("t3_prefill_offset", 0) or 0)
                s = max(0, min(offset, int(prompt_embeds_cpu.shape[0])))
                e = max(0, min(offset + span_len, int(prompt_embeds_cpu.shape[0])))
                take = prompt_embeds_cpu[s:e]
                if int(take.shape[0]) < span_len:
                    pad_n = span_len - int(take.shape[0])
                    pad_rows = torch.zeros(pad_n, take.shape[-1])
                    take = torch.cat([take, pad_rows], dim=0)
                prompt_embeds = take.to(device=device, dtype=torch.bfloat16)
                info_update = {"t3_prefill_offset": offset + span_len}

            # Dummy input_ids (in-vocab for vLLM bookkeeping).
            input_ids_out = torch.zeros_like(input_ids)
            # Placeholder speech_tokens for the make_omni_output path.
            info_update["speech_tokens"] = torch.zeros((prompt_embeds.shape[0],), device=device, dtype=torch.long)
            return input_ids_out, prompt_embeds, info_update

        # Decode: span_len == 1
        last_hidden_cpu = info_dict.get("last_t3_hidden")
        if not isinstance(last_hidden_cpu, torch.Tensor):
            raise RuntimeError("Missing `last_t3_hidden` in additional_information; postprocess must run first.")

        last_token_embed = self.speech_emb(input_ids.clamp(0, self.config.speech_vocab_size - 1).long())
        inputs_embeds_out = last_token_embed.reshape(1, -1)

        info_update = {
            "speech_tokens": input_ids.reshape(-1).to(torch.long),
        }
        return input_ids, inputs_embeds_out, info_update

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        last = hidden_states[-1, :].detach().to("cpu").contiguous()
        return {"last_t3_hidden": last}

    # -------------------- Prompt construction --------------------

    def _build_prompt_embeds(self, info_dict: dict[str, Any], device: torch.device) -> torch.Tensor:
        """Build the full prompt embedding sequence for T3 prefill.

        Sequence layout: [cond_enc_output || text_emb || start_speech_token_emb]

        Where cond_enc_output = [speaker_proj(speaker_emb) || speech_emb(cond_prompt)]
        """
        text = info_dict["text"][0]
        ref_audio_path = None
        ref_audio_list = info_dict.get("ref_audio")
        if isinstance(ref_audio_list, list) and ref_audio_list:
            ref_audio_path = ref_audio_list[0]

        tokenizer = self._get_tokenizer()
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        text_token_ids = torch.tensor(text_tokens, dtype=torch.long, device=device)

        text_embedded = self.text_emb(text_token_ids).unsqueeze(0)  # (1, T, H)

        speaker_emb = self._get_speaker_embedding(ref_audio_path, device)  # (1, 256)

        cond_speech_emb = self._get_cond_prompt_speech_emb(ref_audio_path, device)  # (1, plen, H) or None

        cond_output = self.cond_enc(speaker_emb, cond_speech_emb)  # (1, 1+plen, H)

        start_token = torch.tensor([self.config.start_speech_token], dtype=torch.long, device=device)
        start_emb = self.speech_emb(start_token).unsqueeze(0)  # (1, 1, H)

        # Concatenate: [cond || text || start_speech]
        prompt_embeds = torch.cat([cond_output, text_embedded, start_emb], dim=1)  # (1, L, H)

        # Store ref_dict for S3Gen (speaker embedding + conditioning for vocoder).
        info_dict["_ref_dict"] = {
            "speaker_emb": speaker_emb.detach().cpu(),
            "ref_audio_path": ref_audio_path,
        }

        return prompt_embeds.squeeze(0)  # (L, H)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    def _get_speaker_embedding(self, ref_audio_path: str | None, device: torch.device) -> torch.Tensor:
        """Extract speaker embedding from reference audio using VoiceEncoder."""
        if ref_audio_path is None:
            return torch.zeros(1, self.config.speaker_embed_size, device=device)

        voice_encoder = self._ensure_voice_encoder(device)

        import torchaudio

        wav, sr = torchaudio.load(ref_audio_path)
        if sr != S3_SR:
            wav = torchaudio.functional.resample(wav, sr, S3_SR)
        wav = wav.mean(0, keepdim=True)  # mono

        if wav.shape[-1] > ENC_COND_LEN:
            wav = wav[..., :ENC_COND_LEN]

        wav = wav.to(device=device)
        with torch.no_grad():
            emb = voice_encoder(wav)  # (1, 256)
        return emb

    def _get_cond_prompt_speech_emb(self, ref_audio_path: str | None, device: torch.device) -> torch.Tensor | None:
        """Tokenize reference audio and embed as conditioning prompt."""
        if ref_audio_path is None:
            return None

        s3_tokenizer = self._ensure_s3_tokenizer(device)

        import torchaudio

        wav, sr = torchaudio.load(ref_audio_path)
        if sr != S3_SR:
            wav = torchaudio.functional.resample(wav, sr, S3_SR)
        wav = wav.mean(0, keepdim=True)  # mono

        pad_len = S3_TOKEN_HOP - (wav.shape[-1] % S3_TOKEN_HOP)
        if pad_len < S3_TOKEN_HOP:
            wav = torch.nn.functional.pad(wav, (0, pad_len))

        wav = wav.to(device=device)
        with torch.no_grad():
            tokens, _ = s3_tokenizer(wav.unsqueeze(0), max_len=None)  # (1, plen)

        plen = self.config.speech_cond_prompt_len
        if tokens.shape[-1] > plen:
            tokens = tokens[:, :plen]

        cond_emb = self.speech_emb(tokens.clamp(0, self.config.speech_vocab_size - 1))  # (1, plen, H)
        return cond_emb

    def _ensure_voice_encoder(self, device: torch.device):
        """Lazy-load the VoiceEncoder (CAM++) for speaker embedding extraction."""
        if self._voice_encoder is not None:
            return self._voice_encoder

        try:
            from chatterbox.models.voice_encoder import VoiceEncoder

            ve_path = cached_file(self.model_path, "ve.pt")
            if ve_path is None:
                raise FileNotFoundError("ve.pt not found in model checkpoint")
            self._voice_encoder = VoiceEncoder.from_pretrained(ve_path, device=device)
        except ImportError:
            logger.warning(
                "chatterbox package not installed; using dummy VoiceEncoder. "
                "Install chatterbox for proper speaker embedding extraction."
            )
            self._voice_encoder = _DummyVoiceEncoder(self.config.speaker_embed_size, device)
        return self._voice_encoder

    def _ensure_s3_tokenizer(self, device: torch.device):
        """Lazy-load S3Tokenizer for reference audio tokenization."""
        if self._s3_tokenizer is not None:
            return self._s3_tokenizer

        try:
            from chatterbox.models.s3tokenizer import S3Tokenizer

            s3tok_path = cached_file(self.model_path, "s3tokenizer.pt")
            if s3tok_path is None:
                raise FileNotFoundError("s3tokenizer.pt not found in model checkpoint")
            self._s3_tokenizer = S3Tokenizer.from_pretrained(s3tok_path, device=device)
        except ImportError:
            logger.warning(
                "chatterbox package not installed; S3Tokenizer unavailable. Conditioning prompt will be empty."
            )
            self._s3_tokenizer = _DummyS3Tokenizer(device)
        return self._s3_tokenizer

    # -------------------- Weight loading --------------------

    # Map Chatterbox safetensors keys → vLLM parameter names.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "tfmr.": "model.",
            "text_emb.": "text_emb.",
            "speech_emb.": "speech_emb.",
            "speech_head.": "speech_head.",
            "cond_enc.speaker_proj.": "cond_enc.speaker_proj.",
        }
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                continue

            mapped_name = name
            for old_prefix, new_prefix in self.hf_to_vllm_mapper.orig_to_new_prefix.items():
                if name.startswith(old_prefix):
                    mapped_name = new_prefix + name[len(old_prefix) :]
                    break

            if mapped_name not in params_dict:
                continue

            param = params_dict[mapped_name]
            # GPT-2 Conv1D → Linear transpose.
            for conv1d_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_name in mapped_name and mapped_name.endswith(".weight"):
                    loaded_weight = loaded_weight.t()
                    break

            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, loaded_weight)
            else:
                param.data.copy_(loaded_weight)
            loaded_params.add(mapped_name)

        return loaded_params

    # -------------------- Prompt length estimation (for offline inference) --------------------

    @staticmethod
    def estimate_prompt_len(text: str, speech_cond_prompt_len: int = 375) -> int:
        """Rough estimate of prompt token count for placeholder allocation.

        Layout: [1 (speaker) + speech_cond_prompt_len + text_tokens + 1 (start_speech)]
        """
        # Rough: ~1 token per 4 chars for GPT-2 tokenizer.
        text_len = max(1, len(text) // 4 + 10)
        return 1 + speech_cond_prompt_len + text_len + 1


# Fallback stubs when chatterbox package is not installed
class _DummyVoiceEncoder:
    def __init__(self, embed_size: int, device: torch.device):
        self.embed_size = embed_size
        self.device = device

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, self.embed_size, device=self.device)


class _DummyS3Tokenizer:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, wav: torch.Tensor, max_len=None):
        return torch.zeros(1, 375, dtype=torch.long, device=self.device), torch.tensor([375])
