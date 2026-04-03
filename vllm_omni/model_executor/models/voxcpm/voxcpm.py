from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import warnings
import wave
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional, Tuple, Type, Union
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _import_voxcpm_base_model_class():
    """Import upstream ``VoxCPMModel`` from ``VoxCPM/src/voxcpm`` (env, sibling tree, or pip)."""
    env_path = os.environ.get("VLLM_OMNI_VOXCPM_CODE_PATH")
    if env_path:
        env_candidate = Path(env_path).expanduser()
        if env_candidate.exists():
            candidate_str = str(env_candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            try:
                from voxcpm.model.voxcpm import VoxCPMModel

                return VoxCPMModel
            except ImportError:
                pass

    candidates: list[Path] = []
    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root.parent / "VoxCPM" / "src")

    for candidate in candidates:
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            from voxcpm.model.voxcpm import VoxCPMModel

            return VoxCPMModel
        except ImportError:
            continue

    try:
        from voxcpm.model.voxcpm import VoxCPMModel

        return VoxCPMModel
    except ImportError:
        pass

    raise ImportError(
        "Failed to import VoxCPMModel. Install the `voxcpm` package or set "
        "`VLLM_OMNI_VOXCPM_CODE_PATH` to the VoxCPM repository `src` directory "
        "(the parent of the `voxcpm` package that contains `model/` and `modules/`)."
    )


def _import_voxcpm_audio_vae_classes():
    env_path = os.environ.get("VLLM_OMNI_VOXCPM_CODE_PATH")
    if env_path:
        env_candidate = Path(env_path).expanduser()
        if env_candidate.exists():
            candidate_str = str(env_candidate)
            # 强制插入到 sys.path 最顶部
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            # 立刻尝试导入
            try:
                from voxcpm.modules.audiovae import AudioVAE, AudioVAEConfig
                return AudioVAE, AudioVAEConfig
            except ImportError:
                pass

    # 环境变量无效 → 默认仓库路径
    candidates: list[Path] = []
    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root.parent / "VoxCPM" / "src")

    for candidate in candidates:
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            from voxcpm.modules.audiovae import AudioVAE, AudioVAEConfig
            return AudioVAE, AudioVAEConfig
        except ImportError:
            continue

    # 最后尝试 pip 包
    try:
        from voxcpm.modules.audiovae import AudioVAE, AudioVAEConfig
        return AudioVAE, AudioVAEConfig
    except ImportError:
        pass

    raise ImportError(
        "Failed to import VoxCPM AudioVAE. Install the `voxcpm` package or set "
        "`VLLM_OMNI_VOXCPM_CODE_PATH` to the VoxCPM repository `src` directory."
    )


def _make_voxcpm_model_for_omni(base: Type[Any]) -> Type[Any]:
    """Subclass upstream VoxCPMModel: local ``_inference`` + ``latents_only`` prompt-cache generation."""

    from voxcpm.model.utils import get_dtype

    class VoxCPMModelForOmni(base):
        @torch.inference_mode()
        def build_prompt_cache(self, *args: Any, **kwargs: Any):
            with _patch_torchaudio_load_with_soundfile():
                return super().build_prompt_cache(*args, **kwargs)

        @torch.inference_mode()
        def _inference(
            self,
            text: torch.Tensor,
            text_mask: torch.Tensor,
            feat: torch.Tensor,
            feat_mask: torch.Tensor,
            min_len: int = 2,
            max_len: int = 2000,
            inference_timesteps: int = 10,
            cfg_value: float = 2.0,
            streaming: bool = False,
            streaming_prefix_len: int = 3,
        ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
            """Core inference loop (aligned with upstream ``VoxCPMModel._inference``)."""
            B, _, _, _ = feat.shape

            feat_embed = self.feat_encoder(feat)
            feat_embed = self.enc_to_lm_proj(feat_embed)

            if self.config.lm_config.use_mup:
                scale_emb = self.config.lm_config.scale_emb
            else:
                scale_emb = 1.0

            text_embed = self.base_lm.embed_tokens(text) * scale_emb
            combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

            prefix_feat_cond = feat[:, -1, ...]
            pred_feat_seq: List[torch.Tensor] = []

            audio_patch_count = int(feat_mask.sum().item())
            if audio_patch_count > 0:
                context_len = min(streaming_prefix_len - 1, audio_patch_count)
                prompt_context_patches = list(feat[:, -context_len:, :, :].split(1, dim=1))
                pred_feat_seq = prompt_context_patches + pred_feat_seq

            enc_outputs, kv_cache_tuple = self.base_lm(
                inputs_embeds=combined_embed,
                is_causal=True,
            )
            self.base_lm.kv_cache.fill_caches(kv_cache_tuple)

            enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
            lm_hidden = enc_outputs[:, -1, :]

            residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
                inputs_embeds=enc_outputs + feat_mask.unsqueeze(-1) * feat_embed,
                is_causal=True,
            )
            self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
            residual_hidden = residual_enc_outputs[:, -1, :]

            for _i in tqdm(range(max_len)):
                dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)
                dit_hidden_2 = self.res_to_dit_proj(residual_hidden)
                dit_hidden = dit_hidden_1 + dit_hidden_2

                pred_feat = self.feat_decoder(
                    mu=dit_hidden,
                    patch_size=self.patch_size,
                    cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                    n_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                ).transpose(1, 2)

                curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))
                curr_embed = self.enc_to_lm_proj(curr_embed)

                pred_feat_seq.append(pred_feat.unsqueeze(1))
                prefix_feat_cond = pred_feat

                if streaming:
                    pred_feat_chunk = torch.cat(pred_feat_seq[-streaming_prefix_len:], dim=1)
                    feat_pred = rearrange(pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size)
                    yield feat_pred, pred_feat_seq

                stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
                if _i > min_len and stop_flag == 1:
                    break

                lm_hidden = self.base_lm.forward_step(
                    curr_embed[:, 0, :],
                    torch.tensor([self.base_lm.kv_cache.step()], device=curr_embed.device),
                ).clone()

                lm_hidden = self.fsq_layer(lm_hidden)
                residual_hidden = self.residual_lm.forward_step(
                    lm_hidden + curr_embed[:, 0, :],
                    torch.tensor([self.residual_lm.kv_cache.step()], device=curr_embed.device),
                ).clone()

            if not streaming:
                pred_feat_seq_cat = torch.cat(pred_feat_seq, dim=1)
                feat_pred = rearrange(pred_feat_seq_cat, "b t p d -> b d (t p)", b=B, p=self.patch_size)
                yield feat_pred, pred_feat_seq_cat.squeeze(0).cpu()

        @torch.inference_mode()
        def _generate_with_prompt_cache(
            self,
            target_text: str,
            prompt_cache: dict,
            min_len: int = 2,
            max_len: int = 2000,
            inference_timesteps: int = 10,
            cfg_value: float = 2.0,
            retry_badcase: bool = False,
            retry_badcase_max_times: int = 3,
            retry_badcase_ratio_threshold: float = 6.0,
            streaming: bool = False,
            streaming_prefix_len: int = 3,
            latents_only: bool = False,
        ) -> Generator[
            Tuple[Optional[torch.Tensor], torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]],
            None,
            None,
        ]:
            if retry_badcase and streaming:
                warnings.warn(
                    "Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.",
                )
                retry_badcase = False
            if prompt_cache is None:
                prompt_audio_feat = torch.empty(
                    (0, self.patch_size, self.audio_vae.latent_dim),
                    dtype=torch.float32,
                )
                text = target_text
            else:
                prompt_audio_feat = prompt_cache["audio_feat"]
                prompt_text = prompt_cache["prompt_text"]
                text = prompt_text + target_text

            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor(
                        [self.audio_start_token],
                        dtype=torch.int32,
                        device=text_token.device,
                    ),
                ],
                dim=-1,
            )

            target_text_token = torch.LongTensor(self.text_tokenizer(target_text))

            audio_length = prompt_audio_feat.size(0)
            text_length = text_token.shape[0]
            text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
            audio_pad_feat = torch.zeros(
                (text_token.shape[0], self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([text_token, text_pad_token])
            audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat], dim=0)
            text_mask = (
                torch.cat([torch.ones(text_length), torch.zeros(audio_length)])
                .type(torch.int32)
                .to(text_token.device)
            )
            audio_mask = (
                torch.cat([torch.zeros(text_length), torch.ones(audio_length)])
                .type(torch.int32)
                .to(text_token.device)
            )

            text_token = text_token.unsqueeze(0).to(self.device)
            text_mask = text_mask.unsqueeze(0).to(self.device)
            audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
            audio_mask = audio_mask.unsqueeze(0).to(self.device)

            target_text_length = len(self.text_tokenizer(target_text))
            retry_badcase_times = 0
            while retry_badcase_times < retry_badcase_max_times:
                inference_result = self._inference(
                    text_token,
                    text_mask,
                    audio_feat,
                    audio_mask,
                    min_len=min_len,
                    max_len=min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len),
                    inference_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                    streaming=streaming,
                    streaming_prefix_len=streaming_prefix_len,
                )
                if streaming:
                    patch_len = self.patch_size * self.chunk_size
                    for latent_pred, pred_audio_feat in inference_result:
                        if latents_only:
                            decode_audio = None
                            # Third value must be the latent window for VAE (Omni async_chunk).
                            yield (decode_audio, target_text_token, latent_pred)
                        else:
                            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                            decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                            yield (decode_audio, target_text_token, pred_audio_feat)
                    break
                else:
                    latent_pred, pred_audio_feat = next(inference_result)
                    if retry_badcase:
                        if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                            print(
                                f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...",
                                file=sys.stderr,
                            )
                            retry_badcase_times += 1
                            continue
                        break
                    break

            if not streaming:
                if latents_only:
                    decode_audio = None
                else:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    patch_len = self.patch_size * self.chunk_size
                    if audio_mask.sum().item() > 0:
                        decode_audio = decode_audio[..., patch_len * (streaming_prefix_len - 1) :].squeeze(1).cpu()
                    else:
                        decode_audio = decode_audio[..., :].squeeze(1).cpu()
                yield (decode_audio, target_text_token, pred_audio_feat)

    VoxCPMModelForOmni.__name__ = "VoxCPMModelForOmni"
    VoxCPMModelForOmni.__qualname__ = "VoxCPMModelForOmni"
    return VoxCPMModelForOmni


def _import_voxcpm_model_class() -> Type[Any]:
    base = _import_voxcpm_base_model_class()
    return _make_voxcpm_model_for_omni(base)


def _device_to_string(device: torch.device) -> str:
    if device.index is None:
        return device.type
    return f"{device.type}:{device.index}"


def _normalize_dtype_name(dtype: Any) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        mapping = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
        }
        return mapping.get(dtype, str(dtype).removeprefix("torch."))
    dtype_str = str(dtype)
    return dtype_str.removeprefix("torch.")


def _resolve_runtime_device(vllm_config: VllmConfig) -> torch.device:
    try:
        from vllm_omni.platforms import current_omni_platform

        return current_omni_platform.get_torch_device()
    except Exception:
        pass

    device = getattr(getattr(vllm_config, "device_config", None), "device", None)
    if isinstance(device, torch.device):
        return device
    if device:
        return torch.device(device)
    return torch.device("cpu")


def _prepare_runtime_model_dir(
    model_path: str | Path,
    *,
    target_device: torch.device,
    target_dtype: str | None,
) -> str:
    source_dir = Path(model_path)
    config_path = source_dir / "config.json"
    if not config_path.exists():
        return str(source_dir)

    config_dict = json.loads(config_path.read_text())
    desired_device = target_device.type
    desired_dtype = target_dtype or config_dict.get("dtype")

    if config_dict.get("device") == desired_device and config_dict.get("dtype") == desired_dtype:
        return str(source_dir)

    digest = sha256(
        f"{source_dir.resolve()}:{config_path.read_text()}:{desired_device}:{desired_dtype}".encode("utf-8")
    ).hexdigest()[:16]
    runtime_dir = Path(tempfile.gettempdir()) / "vllm_omni_voxcpm_runtime" / digest
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for entry in source_dir.iterdir():
        target = runtime_dir / entry.name
        if entry.name == "config.json" or target.exists():
            continue
        try:
            target.symlink_to(entry, target_is_directory=entry.is_dir())
        except OSError:
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, target)

    patched_config = dict(config_dict)
    patched_config["device"] = desired_device
    if desired_dtype is not None:
        patched_config["dtype"] = desired_dtype
    (runtime_dir / "config.json").write_text(json.dumps(patched_config, indent=2, sort_keys=True))
    return str(runtime_dir)


@contextmanager
def _force_cuda_available_for_npu(device: torch.device):
    if device.type != "npu":
        yield
        return

    with patch("torch.cuda.is_available", return_value=True):
        yield


@contextmanager
def _patch_torchaudio_load_with_soundfile():
    """Use soundfile-backed loading to avoid torchaudio's torchcodec dependency."""
    try:
        import soundfile as sf
        import torchaudio
    except ImportError:
        yield
        return

    def _load_with_soundfile(
        uri: Any,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: str | None = None,
        buffer_size: int = 4096,
        backend: str | None = None,
    ) -> tuple[torch.Tensor, int]:
        del normalize, format, buffer_size, backend

        audio, sample_rate = sf.read(uri, dtype="float32", always_2d=False)
        audio_np = np.asarray(audio, dtype=np.float32)

        start = max(int(frame_offset), 0)
        stop = None if num_frames is None or int(num_frames) < 0 else start + int(num_frames)
        audio_np = audio_np[start:stop]

        if audio_np.ndim == 1:
            audio_np = audio_np[None, :] if channels_first else audio_np[:, None]
        elif audio_np.ndim == 2:
            if channels_first:
                audio_np = audio_np.T
        else:
            raise ValueError(f"Unsupported audio shape from soundfile: {audio_np.shape}")

        return torch.from_numpy(np.ascontiguousarray(audio_np)), int(sample_rate)

    with patch.object(torchaudio, "load", new=_load_with_soundfile):
        yield


class _DirectVoxCPMLatentGenerator:
    def __init__(self, tts_model: Any):
        self.tts_model = tts_model
        self.sample_rate = int(getattr(tts_model, "sample_rate", 24000))

    def generate_latents(
        self,
        *,
        text: str,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
    ) -> torch.Tensor:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("target text must be a non-empty string")
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        if prompt_wav_path is not None and not os.path.exists(prompt_wav_path):
            raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")

        prompt_cache = None
        if prompt_wav_path is not None and prompt_text is not None:
            prompt_cache = self.tts_model.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=prompt_wav_path,
            )

        gen_kw = dict(
            target_text=" ".join(text.split()),
            prompt_cache=prompt_cache,
            min_len=min_len,
            max_len=max_len,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
        )
        try:
            _, _, pred_audio_feat = self.tts_model.generate_with_prompt_cache(
                **gen_kw,
                latents_only=True,
            )
        except TypeError:
            _, _, pred_audio_feat = self.tts_model.generate_with_prompt_cache(**gen_kw)
        return pred_audio_feat.detach().cpu().to(torch.float32)

    def iter_latent_chunks_streaming(
        self,
        *,
        text: str,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        streaming_prefix_len: int = 3,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
    ) -> Generator[Tuple[torch.Tensor, bool], None, None]:
        """Yield (latent_window, is_last_chunk) for Omni async_chunk latent → VAE."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("target text must be a non-empty string")
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        if prompt_wav_path is not None and not os.path.exists(prompt_wav_path):
            raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")

        prompt_cache = None
        if prompt_wav_path is not None and prompt_text is not None:
            prompt_cache = self.tts_model.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=prompt_wav_path,
            )

        gen_kw = dict(
            target_text=" ".join(text.split()),
            prompt_cache=prompt_cache,
            min_len=min_len,
            max_len=max_len,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            streaming_prefix_len=streaming_prefix_len,
            latents_only=True,
        )
        # Upstream ``generate_with_prompt_cache`` forces ``streaming=False`` (see VoxCPM voxcpm.py).
        stream_entry = getattr(self.tts_model, "generate_with_prompt_cache_streaming", None)
        if stream_entry is not None:
            gen = stream_entry(**gen_kw)
        else:
            gen = self.tts_model._generate_with_prompt_cache(streaming=True, **gen_kw)

        it = iter(gen)
        prev = next(it, None)
        while prev is not None:
            cur = next(it, None)
            _, _target_tok, chunk_latent = prev
            if not isinstance(chunk_latent, torch.Tensor):
                chunk_latent = torch.as_tensor(chunk_latent)
            yield chunk_latent, cur is None
            prev = cur


class _DirectVoxCPMAudioVAE:
    def __init__(self, audio_vae: nn.Module, *, patch_size: int = 2):
        self.audio_vae = audio_vae
        self.sample_rate = int(getattr(audio_vae, "sample_rate", 24000))
        self.latent_dim = int(getattr(audio_vae, "latent_dim", 64))
        self.patch_size = int(patch_size)
        self._chunk_size = int(getattr(audio_vae, "chunk_size", 1))
        # Matches VoxCPM ``_generate_with_prompt_cache`` streaming: decode_audio[..., -patch_len:]
        self._stream_audio_patch_samples = max(1, self.patch_size * self._chunk_size)

    def _prepare_latents_for_decode(self, latent_audio_feat: Any) -> torch.Tensor:
        latents = latent_audio_feat
        if not isinstance(latents, torch.Tensor):
            latents = torch.tensor(latents, dtype=torch.float32)
        latents = latents.detach().to(torch.float32)

        if latents.ndim == 3:
            if latents.shape[-1] == self.latent_dim:
                latents = rearrange(latents, "t p d -> 1 d (t p)")
            elif latents.shape[1] == self.latent_dim:
                latents = latents.contiguous()
            else:
                raise ValueError(f"Unsupported latent_audio_feat shape: {tuple(latents.shape)}")
        elif latents.ndim == 2:
            if latents.shape[0] == self.latent_dim:
                latents = latents.unsqueeze(0)
            elif latents.shape[1] == self.latent_dim:
                latents = rearrange(latents, "t d -> 1 d t")
            else:
                raise ValueError(f"Unsupported latent_audio_feat shape: {tuple(latents.shape)}")
        else:
            raise ValueError(f"Unsupported latent_audio_feat ndim: {latents.ndim}")

        return latents

    @torch.no_grad()
    def decode(self, latent_audio_feat: Any, *, trim_streaming_patch: bool = False) -> torch.Tensor:
        latents = self._prepare_latents_for_decode(latent_audio_feat)
        device = next(self.audio_vae.parameters()).device
        raw = self.audio_vae.decode(latents.to(device=device, dtype=torch.float32))
        if isinstance(raw, dict):
            audio = raw.get("audio")
            if audio is None:
                audio = next(v for v in raw.values() if isinstance(v, torch.Tensor))
        else:
            audio = raw
        if audio.dim() == 3:
            stream = audio.squeeze(1)
        elif audio.dim() == 2:
            stream = audio
        else:
            stream = audio.reshape(audio.shape[0], -1)
        if trim_streaming_patch:
            pl = self._stream_audio_patch_samples
            stream = stream[..., -pl:]
        return stream.reshape(-1).detach().cpu().to(torch.float32)


def _load_native_voxcpm_model(
    model_path: str,
    *,
    device: torch.device,
    dtype: str | None,
):
    VoxCPMModel = _import_voxcpm_model_class()
    runtime_model_path = _prepare_runtime_model_dir(model_path, target_device=device, target_dtype=dtype)

    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)

    with _force_cuda_available_for_npu(device):
        tts_model = VoxCPMModel.from_local(
            runtime_model_path,
            optimize=device.type == "cuda",
        )

    return tts_model


def _load_native_voxcpm_latent_generator(
    model_path: str,
    *,
    device: torch.device,
    dtype: str | None,
) -> _DirectVoxCPMLatentGenerator:
    return _DirectVoxCPMLatentGenerator(_load_native_voxcpm_model(model_path, device=device, dtype=dtype))


def _load_native_voxcpm_audio_vae(
    model_path: str,
    *,
    device: torch.device,
) -> _DirectVoxCPMAudioVAE:
    AudioVAE, AudioVAEConfig = _import_voxcpm_audio_vae_classes()
    runtime_model_path = _prepare_runtime_model_dir(model_path, target_device=device, target_dtype="float32")
    config_dict = json.loads((Path(runtime_model_path) / "config.json").read_text())
    audio_vae_config = config_dict.get("audio_vae_config")
    if audio_vae_config is not None:
        audio_vae = AudioVAE(config=AudioVAEConfig(**audio_vae_config))
    else:
        audio_vae = AudioVAE()

    state_dict = torch.load(
        Path(runtime_model_path) / "audiovae.pth",
        map_location="cpu",
        weights_only=True,
    )["state_dict"]
    audio_vae.load_state_dict(state_dict, strict=True)
    audio_vae = audio_vae.to(device=device, dtype=torch.float32).eval()
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)
    patch_size = int(config_dict.get("patch_size", 2))
    return _DirectVoxCPMAudioVAE(audio_vae, patch_size=patch_size)


class VoxCPMForConditionalGeneration(nn.Module):
    input_modalities = "audio"
    _LATENT_STAGES = {"latent_generator", "latent", "ar_dit"}
    _VAE_STAGES = {"vae", "audio_vae"}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.model_stage = getattr(vllm_config.model_config, "model_stage", "latent_generator")
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self.inject_omni_request_id_into_runtime_info = True
        self._pipeline = None
        self._latent_stream_gens: dict[str, Any] = {}
        # When True, AR sampling should emit EOS so vLLM ends the request; when False, emit a
        # non-EOS placeholder so streaming latent/VAE steps can continue (see compute_logits).
        self._ar_emit_stop_token: bool = True

    def _runner_hidden_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        """Device/dtype for tensors consumed by NPU/GPU AR runner (must match logits_indices)."""
        device = _resolve_runtime_device(self.vllm_config)
        mc = getattr(self.vllm_config, "model_config", None)
        dtype = getattr(mc, "dtype", torch.float32) if mc is not None else torch.float32
        return device, dtype

    def _ensure_model_loaded(self):
        if self._pipeline is not None:
            return

        target_device = _resolve_runtime_device(self.vllm_config)
        model_dtype = getattr(self.vllm_config.model_config, "dtype", None)
        normalized_dtype = _normalize_dtype_name(model_dtype)
        if self.model_stage in self._LATENT_STAGES:
            self._pipeline = _load_native_voxcpm_latent_generator(
                self.model_path,
                device=target_device,
                dtype=normalized_dtype,
            )
        elif self.model_stage in self._VAE_STAGES:
            self._pipeline = _load_native_voxcpm_audio_vae(
                self.model_path,
                device=target_device,
            )
        else:
            raise ValueError(
                f"Unsupported VoxCPM model_stage: {self.model_stage}. "
                "pure_voxcpm only supports split-stage latent_generator/vae inference."
            )

        logger.info(
            "Loaded VoxCPM stage '%s' on %s",
            self.model_stage,
            _device_to_string(target_device),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load VoxCPM via its native runtime instead of vLLM's HF weight loader.

        VoxCPM stages are constructed from the original local model directory using
        ``VoxCPMModel.from_local`` / ``AudioVAE`` inside ``_ensure_model_loaded``.
        The standard vLLM weight iterator is therefore not applicable here.
        """
        del weights
        self._ensure_model_loaded()
        return set()

    @staticmethod
    def _extract_val(info: dict[str, Any], key: str, default: Any) -> Any:
        value = info.get(key, default)
        if isinstance(value, list):
            return value[0] if value else default
        return value

    @staticmethod
    def _normalize_audio_samples(samples: Any) -> np.ndarray:
        if isinstance(samples, torch.Tensor):
            return samples.detach().cpu().float().reshape(-1).numpy()
        return np.asarray(samples, dtype=np.float32).reshape(-1)

    @classmethod
    def _normalize_ref_audio(cls, ref_audio: Any) -> tuple[np.ndarray, int]:
        if isinstance(ref_audio, str):
            raise TypeError("String ref_audio should be handled as a path before waveform normalization.")

        if isinstance(ref_audio, dict):
            sr = ref_audio.get("sample_rate") or ref_audio.get("sampling_rate") or ref_audio.get("sr")
            samples = None
            for key in ("audio", "wav", "samples", "array", "waveform"):
                if key in ref_audio and ref_audio[key] is not None:
                    samples = ref_audio[key]
                    break
            if sr is None or samples is None:
                raise ValueError("ref_audio dict must contain waveform data and sample rate.")
            return cls._normalize_audio_samples(samples), int(sr)

        if isinstance(ref_audio, (list, tuple)):
            if len(ref_audio) == 1:
                return cls._normalize_ref_audio(ref_audio[0])
            if len(ref_audio) == 2 and np.isscalar(ref_audio[1]):
                return cls._normalize_audio_samples(ref_audio[0]), int(ref_audio[1])

        raise TypeError(f"Unsupported ref_audio format: {type(ref_audio)!r}")

    @staticmethod
    def _write_temp_prompt_wav(waveform: np.ndarray, sample_rate: int) -> str:
        prompt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        prompt_file.close()

        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        wav = np.clip(wav, -1.0, 1.0)
        pcm16 = (wav * 32767.0).astype(np.int16)
        with wave.open(prompt_file.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm16.tobytes())

        return prompt_file.name

    @classmethod
    def _resolve_prompt_inputs(cls, info: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
        prompt_text = cls._extract_val(info, "prompt_text", None)
        prompt_wav_path = cls._extract_val(info, "prompt_wav_path", None)
        if prompt_wav_path:
            if prompt_text is None:
                prompt_text = cls._extract_val(info, "ref_text", None)
            return prompt_wav_path, prompt_text, None

        ref_audio = cls._extract_val(info, "ref_audio", None)
        ref_text = cls._extract_val(info, "ref_text", None)
        if ref_audio is None or ref_text is None:
            return None, None, None

        if isinstance(ref_audio, str):
            return ref_audio, ref_text, None

        waveform, sample_rate = cls._normalize_ref_audio(ref_audio)
        temp_prompt_wav = cls._write_temp_prompt_wav(waveform, sample_rate)
        return temp_prompt_wav, ref_text, temp_prompt_wav

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def _get_vocab_size(self) -> int:
        model_cfg = getattr(self.vllm_config, "model_config", None)
        if model_cfg is not None:
            getter = getattr(model_cfg, "get_vocab_size", None)
            if callable(getter):
                try:
                    return int(getter())
                except Exception:
                    pass
            hf_cfg = getattr(model_cfg, "hf_text_config", None)
            if hf_cfg is not None and hasattr(hf_cfg, "vocab_size"):
                return int(hf_cfg.vocab_size)
        return 32000

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor:
        del sampling_metadata
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if hidden_states is None:
            dev, dt = self._runner_hidden_device_dtype()
            hidden_states = torch.zeros((0, 1), device=dev, dtype=dt)
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(-1)
        elif hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        vocab_size = self._get_vocab_size()
        num_rows = int(hidden_states.shape[0])
        logits = torch.zeros(
            (num_rows, vocab_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        eos_id = 2 if vocab_size > 2 else 0
        safe_id = 1 if vocab_size > 1 and 1 != eos_id else 0
        emit_stop = getattr(self, "_ar_emit_stop_token", True)
        if num_rows > 0:
            if emit_stop:
                logits[:, eos_id] = 1.0e6
            else:
                logits[:, eos_id] = -1.0e9
                logits[:, safe_id] = 1.0e6
        return logits

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds, kwargs
        self._ensure_model_loaded()
        out_dev, out_dtype = self._runner_hidden_device_dtype()
        if input_ids is not None and input_ids.device.type == out_dev.type:
            out_dev = input_ids.device

        infos = runtime_additional_information or [{}]
        
        sample_rate = int(getattr(self._pipeline, "sample_rate", 24000))
        async_chunk = bool(getattr(self.vllm_config.model_config, "async_chunk", False))
        if self.model_stage in self._VAE_STAGES:
            if all(self._extract_val(info, "latent_audio_feat", None) is None for info in infos):
                self._ar_emit_stop_token = True
                return OmniOutput(
                    text_hidden_states=torch.zeros((0, 1), device=out_dev, dtype=out_dtype),
                    multimodal_outputs={
                        "model_outputs": [torch.zeros((0,), dtype=torch.float32) for _ in infos],
                        "sr": [torch.tensor(sample_rate, dtype=torch.int32) for _ in infos],
                    },
                )
        else:
            texts = [self._extract_val(info, "text", "") for info in infos]
            if all(not text for text in texts):
                mm_empty: dict[str, Any] = {
                    "latent_audio_feat": [torch.zeros((0,), dtype=torch.float32) for _ in infos],
                    "sr": [torch.tensor(sample_rate, dtype=torch.int32) for _ in infos],
                }
                self._ar_emit_stop_token = True
                return OmniOutput(
                    text_hidden_states=torch.zeros((0, 1), device=out_dev, dtype=out_dtype),
                    multimodal_outputs=mm_empty,
                )

        outputs: list[torch.Tensor] = []
        sample_rates: list[torch.Tensor] = []
        last_chunk_flags: list[bool] | None = (
            [] if (self.model_stage in self._LATENT_STAGES and async_chunk) else None
        )
        for info in infos:
            if self.model_stage in self._VAE_STAGES:
                latent_audio_feat = self._extract_val(info, "latent_audio_feat", None)
                print(f"---latent_audio_feat---:{latent_audio_feat.shape}")
                audio_tensor = self._pipeline.decode(
                    latent_audio_feat,
                    trim_streaming_patch=async_chunk,
                )
                outputs.append(audio_tensor.float().cpu())
                sample_rates.append(torch.tensor(sample_rate, dtype=torch.int32))
                continue

            text = self._extract_val(info, "text", "")
            cfg_value = float(self._extract_val(info, "cfg_value", 2.0))
            inference_timesteps = int(self._extract_val(info, "inference_timesteps", 10))
            min_len = int(self._extract_val(info, "min_len", 2))
            max_len = int(self._extract_val(info, "max_len", self._extract_val(info, "max_new_tokens", 4096)))
            retry_badcase = bool(self._extract_val(info, "retry_badcase", True))
            retry_badcase_max_times = int(self._extract_val(info, "retry_badcase_max_times", 3))
            retry_badcase_ratio_threshold = float(self._extract_val(info, "retry_badcase_ratio_threshold", 6.0))
            streaming_prefix_len = int(self._extract_val(info, "streaming_prefix_len", 3))

            req_key = str(info.get("_omni_req_id", "0"))
            prompt_wav_path: str | None = None
            prompt_text: str | None = None
            temp_prompt_wav: str | None = None
            created_temp: str | None = None

            if self.model_stage in self._LATENT_STAGES and async_chunk:
                if req_key not in self._latent_stream_gens:
                    prompt_wav_path, prompt_text, temp_prompt_wav = self._resolve_prompt_inputs(info)
                    created_temp = temp_prompt_wav
                    self._latent_stream_gens[req_key] = self._pipeline.iter_latent_chunks_streaming(
                        text=text,
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=prompt_text,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                        min_len=min_len,
                        max_len=max_len,
                        streaming_prefix_len=streaming_prefix_len,
                        retry_badcase=False,
                        retry_badcase_max_times=retry_badcase_max_times,
                        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                    )
                gen = self._latent_stream_gens[req_key]
                chunk_latent: torch.Tensor | None = None
                try:
                    chunk_latent, is_last = next(gen)
                except StopIteration:
                    self._latent_stream_gens.pop(req_key, None)
                    outputs.append(torch.zeros((0,), dtype=torch.float32))
                    assert last_chunk_flags is not None
                    last_chunk_flags.append(True)
                else:
                    if is_last:
                        self._latent_stream_gens.pop(req_key, None)
                    outputs.append(chunk_latent.detach().float().cpu())
                    assert last_chunk_flags is not None
                    last_chunk_flags.append(bool(is_last))
                finally:
                    if created_temp is not None and os.path.exists(created_temp):
                        os.unlink(created_temp)
                sample_rates.append(torch.tensor(sample_rate, dtype=torch.int32))
                outputs_tensor = torch.stack(outputs)
                if chunk_latent is not None:
                    print(f"---outputs_tensor---:{outputs_tensor.shape},chunk_latent:{chunk_latent.shape}")
                else:
                    print(f"---outputs_tensor---:{outputs_tensor.shape},chunk_latent:StopIteration")
                continue

            prompt_wav_path, prompt_text, temp_prompt_wav = self._resolve_prompt_inputs(info)
            try:
                if self.model_stage in self._LATENT_STAGES:
                    latent_audio_feat = self._pipeline.generate_latents(
                        text=text,
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=prompt_text,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                        min_len=min_len,
                        max_len=max_len,
                        retry_badcase=retry_badcase,
                        retry_badcase_max_times=retry_badcase_max_times,
                        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                    )
                    outputs.append(latent_audio_feat.float().cpu())
                    outputs_tensor = torch.stack(outputs)
                    print(f"---outputs_tensor---:{outputs_tensor.shape},latent_audio_feat:{latent_audio_feat.shape}")
            finally:
                if temp_prompt_wav is not None and os.path.exists(temp_prompt_wav):
                    os.unlink(temp_prompt_wav)

            sample_rates.append(torch.tensor(sample_rate, dtype=torch.int32))

        output_key = "latent_audio_feat" if self.model_stage in self._LATENT_STAGES else "model_outputs"
        mm: dict[str, Any] = {output_key: outputs, "sr": sample_rates}
        if outputs:
            outputs_tensor = torch.stack(outputs)
            if outputs_tensor.ndim == 1:
                text_hidden_states = outputs_tensor.unsqueeze(-1)
            else:
                hidden_dim = outputs_tensor.shape[-1]
                text_hidden_states = outputs_tensor.reshape(-1, hidden_dim)
        else:
            text_hidden_states = torch.zeros((0, 1), device=out_dev, dtype=out_dtype)
        text_hidden_states = text_hidden_states.to(device=out_dev, dtype=out_dtype)

        if self.model_stage in self._LATENT_STAGES and async_chunk and last_chunk_flags:
            self._ar_emit_stop_token = all(last_chunk_flags)
        elif self.model_stage in self._LATENT_STAGES:
            self._ar_emit_stop_token = True
        elif self.model_stage in self._VAE_STAGES:
            self._ar_emit_stop_token = True
        else:
            self._ar_emit_stop_token = True

        return OmniOutput(
            text_hidden_states=text_hidden_states,
            multimodal_outputs=mm,
        )

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        del batch_size, dtype, device
        return {}
