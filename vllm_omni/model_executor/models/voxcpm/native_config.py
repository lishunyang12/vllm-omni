from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def is_native_voxcpm_config_dict(config_dict: Mapping[str, Any]) -> bool:
    return all(
        key in config_dict
        for key in (
            "lm_config",
            "encoder_config",
            "dit_config",
            "patch_size",
            "feat_dim",
        )
    )


def detect_native_voxcpm_model_type(model: str | Path) -> str | None:
    config_path = Path(model) / "config.json"
    if not config_path.exists():
        return None

    try:
        config_dict = json.loads(config_path.read_text())
    except Exception:
        return None

    if is_native_voxcpm_config_dict(config_dict):
        return "voxcpm"

    return None


def _build_hf_compatible_voxcpm_config(config_dict: Mapping[str, Any]) -> dict[str, Any]:
    lm_config = dict(config_dict.get("lm_config", {}) or {})

    return {
        "model_type": "voxcpm",
        "architectures": ["VoxCPMForConditionalGeneration"],
        "bos_token_id": lm_config.get("bos_token_id", 1),
        "eos_token_id": lm_config.get("eos_token_id", 2),
        "vocab_size": lm_config.get("vocab_size", 32000),
        "hidden_size": lm_config.get("hidden_size", 1024),
        "intermediate_size": lm_config.get("intermediate_size", 4096),
        "max_position_embeddings": lm_config.get("max_position_embeddings", 4096),
        "num_attention_heads": lm_config.get("num_attention_heads", 16),
        "num_hidden_layers": lm_config.get("num_hidden_layers", 24),
        "num_key_value_heads": lm_config.get("num_key_value_heads", lm_config.get("num_attention_heads", 16)),
        "rms_norm_eps": lm_config.get("rms_norm_eps", 1e-6),
        "rope_theta": lm_config.get("rope_theta", 10000.0),
        "rope_scaling": lm_config.get("rope_scaling"),
        "lm_config": config_dict.get("lm_config", {}),
        "encoder_config": config_dict.get("encoder_config", {}),
        "dit_config": config_dict.get("dit_config", {}),
        "audio_vae_config": config_dict.get("audio_vae_config"),
        "patch_size": config_dict.get("patch_size", 2),
        "feat_dim": config_dict.get("feat_dim", 64),
        "residual_lm_num_layers": config_dict.get("residual_lm_num_layers", 6),
        "scalar_quantization_latent_dim": config_dict.get("scalar_quantization_latent_dim", 256),
        "scalar_quantization_scale": config_dict.get("scalar_quantization_scale", 9),
        "max_length": config_dict.get("max_length", lm_config.get("max_position_embeddings", 4096)),
        "device": config_dict.get("device", "cuda"),
        "dtype": config_dict.get("dtype", "bfloat16"),
        "dit_mean_mode": config_dict.get("dit_mean_mode", False),
    }


def ensure_hf_compatible_voxcpm_config(model: str | Path) -> str | None:
    model_path = Path(model)
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None

    try:
        config_text = config_path.read_text()
        config_dict = json.loads(config_text)
    except Exception:
        return None

    if not is_native_voxcpm_config_dict(config_dict):
        return None

    rendered = _build_hf_compatible_voxcpm_config(config_dict)
    digest = hashlib.sha256(f"{model_path}:{config_text}".encode()).hexdigest()[:16]
    out_dir = Path(tempfile.gettempdir()) / "vllm_omni_voxcpm_configs" / digest
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "config.json"
    out_path.write_text(json.dumps(rendered, indent=2, sort_keys=True))
    return str(out_dir)
