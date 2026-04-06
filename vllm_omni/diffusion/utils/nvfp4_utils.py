# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for detecting and configuring NVFP4 (ModelOpt FP4) quantization
from safetensors checkpoint files."""

from __future__ import annotations

import json
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


def detect_nvfp4_from_safetensors(weights_path: str) -> bool:
    """Probe safetensors files at *weights_path* for NVFP4 quantization.

    Detection heuristic: NVFP4 checkpoints store weights as ``uint8``
    (packed FP4 nibbles) with companion ``float8_e4m3fn`` scale tensors.
    If we find ``*.weight`` tensors with dtype ``U8`` alongside
    ``*.weight_scale`` tensors, we classify the checkpoint as NVFP4.

    Returns ``True`` if NVFP4 is detected, ``False`` otherwise.
    """
    safetensors_files = _find_safetensors_files(weights_path)
    if not safetensors_files:
        return False

    # Only probe the first file – the pattern should be consistent.
    metadata = _read_safetensors_header(safetensors_files[0])
    if metadata is None:
        return False

    has_uint8_weight = False
    has_weight_scale = False
    for tensor_name, tensor_info in metadata.items():
        if tensor_name == "__metadata__":
            # Check explicit quant markers in file-level metadata.
            if isinstance(tensor_info, dict):
                quant_type = tensor_info.get("quantization", "").lower()
                if "nvfp4" in quant_type or "fp4" in quant_type:
                    return True
            continue
        dtype = tensor_info.get("dtype", "")
        if tensor_name.endswith(".weight") and dtype == "U8":
            has_uint8_weight = True
        if tensor_name.endswith(".weight_scale"):
            has_weight_scale = True
        if has_uint8_weight and has_weight_scale:
            return True

    return False


def _find_safetensors_files(path: str) -> list[str]:
    """Return sorted list of safetensors files under *path*."""
    p = Path(path)
    if p.is_file() and p.suffix == ".safetensors":
        return [str(p)]
    if p.is_dir():
        files = sorted(str(f) for f in p.glob("*.safetensors"))
        # Also check transformer/ subfolder
        if not files:
            files = sorted(str(f) for f in (p / "transformer").glob("*.safetensors") if f.exists())
        return files
    return []


def _read_safetensors_header(filepath: str) -> dict | None:
    """Read the JSON header from a safetensors file without loading tensors."""
    try:
        with open(filepath, "rb") as f:
            # First 8 bytes = uint64 header length
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return None
            header_len = int.from_bytes(header_len_bytes, byteorder="little")
            # Sanity check – headers are typically < 10 MB
            if header_len > 50 * 1024 * 1024:
                logger.warning("Safetensors header too large (%d bytes), skipping", header_len)
                return None
            header_bytes = f.read(header_len)
            return json.loads(header_bytes)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read safetensors header from %s: %s", filepath, e)
        return None
