"""Chatterbox S3Gen vocoder for vLLM-Omni.

S3Gen converts discrete speech tokens from T3 into 24 kHz waveforms via:
1. S3Tokenizer embedding lookup
2. UpsampleConformerEncoder (tokens → hidden)
3. CausalConditionalCFM flow matching (hidden → mel, 2 ODE steps)
4. HiFTGenerator vocoder (mel → waveform)

This is the Stage-1 generation model, following the same pattern as
``Qwen3TTSCode2Wav`` — it consumes raw input token IDs, runs inference in a
single forward pass, and returns waveform audio via ``OmniOutput``.

All S3Gen sub-components (tokenizer, conformer, CFM, HiFi-GAN, CAM++ speaker
encoder) are loaded directly from safetensors / checkpoints in ``__init__``,
not through vLLM's standard weight loader.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

S3GEN_SR = 24000


class ChatterboxS3Gen(nn.Module):
    """Stage-1 vocoder for Chatterbox TTS (GenerationModelRunner).

    Consumes speech tokens (from T3) and optional speaker/reference conditioning,
    then decodes waveform audio at 24 kHz.
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.requires_raw_input_tokens = True

        self._s3gen_model = None
        self._logged_stats = False

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def _ensure_model_loaded(self):
        """Lazy-load the full S3Gen pipeline from checkpoint."""
        if self._s3gen_model is not None:
            return self._s3gen_model

        device = self.vllm_config.device_config.device

        try:
            from chatterbox.models.s3gen import S3Token2Wav

            s3gen_path = cached_file(self.model_path, "s3gen.pt")
            if s3gen_path is None:
                raise FileNotFoundError("s3gen.pt not found in model checkpoint")
            ckpt_dir = os.path.dirname(s3gen_path)

            model = S3Token2Wav.from_pretrained(ckpt_dir, device=device)
            model.eval()
            self._s3gen_model = model
        except ImportError:
            raise ImportError(
                "chatterbox package is required for S3Gen vocoder. Install it with: pip install chatterbox-tts"
            )
        return self._s3gen_model

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def _split_request_ids(self, ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        """Split concatenated input_ids into per-request segments."""
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode speech tokens into audio waveform.

        input_ids layout per request: [ctx_frames, *speech_tokens]
        where speech_tokens are the T3-generated tokens (1D, not codebook-major
        since Chatterbox uses a single codebook).
        """
        model = self._ensure_model_loaded()
        device = self._module_device(model) if hasattr(model, "parameters") else self.vllm_config.device_config.device
        sr_tensor = torch.tensor(S3GEN_SR, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))

        audios: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []

        for req_ids in request_ids_list:
            if req_ids.numel() < 2:
                audios.append(empty)
                srs.append(sr_tensor)
                continue

            ctx_frames = int(req_ids[0].item())
            speech_tokens = req_ids[1:]

            if speech_tokens.numel() == 0:
                audios.append(empty)
                srs.append(sr_tensor)
                continue

            if not self._logged_stats:
                self._logged_stats = True
                logger.info(
                    "S3Gen: tokens=%d ctx=%d range=[%d,%d]",
                    speech_tokens.numel(),
                    ctx_frames,
                    int(speech_tokens.min().item()),
                    int(speech_tokens.max().item()),
                )

            # S3Gen expects tokens as (1, T) on the model device.
            tokens_2d = speech_tokens.unsqueeze(0).to(device=device)

            try:
                wav = model.inference(tokens_2d)  # (1, samples) or (samples,)
                if isinstance(wav, torch.Tensor):
                    wav_np = wav.float().detach().cpu().numpy().reshape(-1)
                elif isinstance(wav, np.ndarray):
                    wav_np = wav.astype(np.float32).reshape(-1)
                else:
                    wav_np = np.array(wav, dtype=np.float32).reshape(-1)

                if ctx_frames > 0:
                    # Each token = S3GEN_SR / 25 = 960 samples.
                    samples_per_token = S3GEN_SR // 25
                    cut = ctx_frames * samples_per_token
                    if cut < wav_np.shape[0]:
                        wav_np = wav_np[cut:]
                    else:
                        logger.warning(
                            "Context trim %d >= decoded length %d; returning empty audio.",
                            cut,
                            wav_np.shape[0],
                        )
                        audios.append(empty)
                        srs.append(sr_tensor)
                        continue

                if wav_np.shape[0] > 0:
                    audios.append(torch.from_numpy(wav_np).to(dtype=torch.float32))
                else:
                    audios.append(empty)
            except Exception as e:
                logger.error("S3Gen inference failed: %s", e)
                audios.append(empty)

            srs.append(sr_tensor)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(f"ChatterboxS3Gen expected (audio_tensor, sr) outputs, got {type(model_outputs)}")

        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_tensor,
                "sr": sr,
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # All S3Gen weights are loaded lazily from checkpoint files.
        return set()
