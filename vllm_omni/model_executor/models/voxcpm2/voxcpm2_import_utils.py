# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic import utilities for VoxCPM2 native package.

Supports three discovery modes:
1. VLLM_OMNI_VOXCPM_CODE_PATH env var (explicit source tree)
2. Sibling directory: ../VoxCPM/src
3. pip-installed voxcpm package (>= 2.0)
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _iter_voxcpm2_src_candidates() -> list[Path]:
    """Yield candidate source directories for VoxCPM2."""
    candidates: list[Path] = []
    env_path = os.environ.get("VLLM_OMNI_VOXCPM_CODE_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root.parent / "VoxCPM" / "src")

    # Deduplicate
    seen: set[str] = set()
    unique: list[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _prepend_src(candidate: Path) -> None:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _import_voxcpm2_attrs(module_name: str, *attr_names: str) -> tuple[Any, ...]:
    """Import attributes from voxcpm package, trying source tree first."""
    last_exc: ImportError | None = None

    # Try source tree candidates
    for candidate in _iter_voxcpm2_src_candidates():
        if not candidate.exists():
            continue
        _prepend_src(candidate)
        try:
            mod = importlib.import_module(module_name)
            return tuple(getattr(mod, name) for name in attr_names)
        except (ImportError, AttributeError) as exc:
            last_exc = ImportError(str(exc))
            continue

    # Fallback: pip-installed package
    try:
        mod = importlib.import_module(module_name)
        return tuple(getattr(mod, name) for name in attr_names)
    except (ImportError, AttributeError) as exc:
        last_exc = ImportError(str(exc))

    raise ImportError(
        f"Could not import {attr_names} from {module_name}. "
        f"Install voxcpm>=2.0: pip install voxcpm. "
        f"Or set VLLM_OMNI_VOXCPM_CODE_PATH to the VoxCPM source tree. "
        f"Last error: {last_exc}"
    )


def import_voxcpm2_core():
    """Import VoxCPM core class for model loading."""
    (VoxCPM,) = _import_voxcpm2_attrs("voxcpm.core", "VoxCPM")
    return VoxCPM


def import_audio_vae_v2():
    """Import AudioVAE V2 for separate VAE stage loading."""
    AudioVAE, AudioVAEConfig = _import_voxcpm2_attrs(
        "voxcpm.modules.audiovae", "AudioVAE", "AudioVAEConfig"
    )
    return AudioVAE, AudioVAEConfig
