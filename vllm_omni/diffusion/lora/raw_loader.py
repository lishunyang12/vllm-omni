# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict loader for official LTX native/Comfy LoRA safetensors."""

from __future__ import annotations

import os

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper

_LTX_KEY_RENAMES = (
    ("audio_patchify_proj", "audio_proj_in"),
    ("patchify_proj", "proj_in"),
    ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
    ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
    ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
    ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ("audio_prompt_adaln_single", "audio_prompt_adaln"),
    ("prompt_adaln_single", "prompt_adaln"),
    ("audio_adaln_single", "audio_time_embed"),
    ("adaln_single", "time_embed"),
)


def _remap_ltx_lora_key(key: str) -> str:
    if not key.startswith("diffusion_model."):
        raise ValueError(f"Official LTX LoRA key must start with 'diffusion_model.': {key}")
    key = key.removeprefix("diffusion_model.")
    for source, destination in _LTX_KEY_RENAMES:
        key = key.replace(source, destination)
    return key


def load_ltx_native_lora(path: str, *, lora_model_id: int, dtype: torch.dtype) -> tuple[LoRAModel, PEFTHelper]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for source_key in handle.keys():
            if not source_key.endswith((".lora_A.weight", ".lora_B.weight")):
                raise ValueError(f"Unexpected tensor in official LTX LoRA: {source_key}")
            tensors[_remap_ltx_lora_key(source_key)] = handle.get_tensor(source_key)
    if not tensors:
        raise ValueError(f"Official LTX LoRA contains no A/B tensors: {path}")
    suffix_a, suffix_b = ".lora_A.weight", ".lora_B.weight"
    prefixes_a = {key.removesuffix(suffix_a) for key in tensors if key.endswith(suffix_a)}
    prefixes_b = {key.removesuffix(suffix_b) for key in tensors if key.endswith(suffix_b)}
    if prefixes_a != prefixes_b:
        raise ValueError(
            f"Incomplete official LTX LoRA pairs: missing_A={sorted(prefixes_b - prefixes_a)}, "
            f"missing_B={sorted(prefixes_a - prefixes_b)}"
        )
    invalid_pairs = [
        prefix
        for prefix in prefixes_a
        if tensors[f"{prefix}{suffix_a}"].ndim != 2
        or tensors[f"{prefix}{suffix_b}"].ndim != 2
        or tensors[f"{prefix}{suffix_b}"].shape[1] != tensors[f"{prefix}{suffix_a}"].shape[0]
    ]
    if invalid_pairs:
        raise ValueError(f"Invalid official LTX LoRA factor shapes for: {sorted(invalid_pairs)}")
    rank = max(int(tensors[f"{prefix}{suffix_a}"].shape[0]) for prefix in prefixes_a)
    targets = sorted({prefix.rsplit(".", 1)[-1] for prefix in prefixes_a})
    # Official LTX computes strength * (B @ A), without PEFT alpha/rank scaling.
    helper = PEFTHelper(r=rank, lora_alpha=rank, target_modules=targets)
    model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_model_id, tensors=tensors, peft_helper=helper, device="cpu", dtype=dtype
    )
    for lora in model.loras.values():
        lora.optimize()
    return model, helper


LTX25_DISTILLED_LORA_FILENAME = "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"


def resolve_ltx25_distilled_lora(model: str, *, revision: str | None = None) -> str:
    """Resolve the required official LoRA450 from a local or Hub checkpoint."""
    if os.path.isdir(model):
        path = os.path.join(model, LTX25_DISTILLED_LORA_FILENAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Full LTX-2.5 two-stage requires {LTX25_DISTILLED_LORA_FILENAME!r} under {model!r}."
            )
        return path
    return hf_hub_download(repo_id=model, filename=LTX25_DISTILLED_LORA_FILENAME, revision=revision)
