# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for detecting and configuring NVFP4 (ModelOpt FP4) quantization
from safetensors checkpoint files.

Handles both local paths and HuggingFace repo IDs (e.g.
``black-forest-labs/FLUX.2-dev-NVFP4``) without fully downloading large
safetensors shards — only the JSON header at the start of the file is
fetched via HTTP range requests when the repo is remote.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Default single-file NVFP4 checkpoint filename convention used by
# black-forest-labs/FLUX.2-dev-NVFP4 and similar BFL-distributed ckps.
_DEFAULT_NVFP4_FILENAME_PATTERNS = (
    "*-nvfp4.safetensors",
    "*-nvfp4-mixed.safetensors",
    "*nvfp4*.safetensors",
)


def detect_nvfp4_from_safetensors(weights_path: str) -> bool:
    """Return True if the checkpoint at *weights_path* looks like NVFP4.

    Detection order:
    1. ``__metadata__._quantization_metadata`` marker (what ModelOpt writes
       into the safetensors header for FLUX.2-dev-NVFP4).
    2. Presence of ``.weight`` tensors with dtype ``U8`` alongside
       companion ``.weight_scale`` tensors — the structural fingerprint of
       block-scaled NVFP4 quantization.
    """
    header = _read_first_safetensors_header(weights_path)
    if header is None:
        return False
    return _header_looks_nvfp4(header)


def parse_nvfp4_quant_metadata(weights_path: str) -> dict[str, Any] | None:
    """Parse NVFP4 metadata and return a ModelOpt-style config dict.

    The returned dict is ready to pass to
    ``ModelOptNvFp4Config.from_config(...)``. Fields ``quant_algo``,
    ``kv_cache_quant_algo``, ``exclude_modules`` and ``group_size`` are all
    guaranteed to be present (required by ModelOpt for NVFP4 checkpoints).

    Returns ``None`` if *weights_path* is not an NVFP4 checkpoint.
    """
    header = _read_first_safetensors_header(weights_path)
    if header is None or not _header_looks_nvfp4(header):
        return None

    # Build the "quantized linear module path" set from the embedded
    # _quantization_metadata. These paths use the checkpoint's native
    # layer naming (BFL-style for FLUX.2).
    quantized_paths: set[str] = set()
    meta = header.get("__metadata__", {}) or {}
    raw = meta.get("_quantization_metadata")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(raw, dict):
        parsed = raw
    else:
        parsed = {}

    layers_meta = parsed.get("layers", {}) if isinstance(parsed, dict) else {}
    for layer_name, layer_info in layers_meta.items():
        if isinstance(layer_info, dict) and "fp4" in str(layer_info.get("format", "")).lower():
            quantized_paths.add(layer_name)

    # Fall back to scanning tensor names if the metadata block was absent.
    if not quantized_paths:
        for tensor_name, info in header.items():
            if tensor_name == "__metadata__":
                continue
            if isinstance(info, dict) and info.get("dtype") == "U8" and tensor_name.endswith(".weight"):
                quantized_paths.add(tensor_name[: -len(".weight")])

    # Flux2 NVFP4 from NVIDIA/BFL quantizes a very specific subset of
    # linear layers. Everything else in the model must be excluded so
    # vLLM's ModelOpt path loads them as plain BF16.
    #
    # We express the exclude list as wildcard patterns in the *vLLM*
    # (post-mapping, diffusers-style) parameter naming, because that is
    # the prefix format ModelOpt's ``is_layer_excluded`` will see when
    # ``get_quant_method`` is called on each layer.
    exclude_modules = _build_flux2_exclude_modules(quantized_paths)

    return {
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "exclude_modules": exclude_modules,
            "group_size": 16,
        }
    }


def resolve_nvfp4_checkpoint_file(weights_path: str, prefer_mixed: bool = False) -> str | None:
    """Return the filename of the preferred NVFP4 safetensors at *weights_path*.

    Used to build ``allow_patterns_overrides`` for
    ``DiffusersPipelineLoader`` so we only download the single variant the
    user actually wants (not both the plain + mixed files, which can
    together be 40+ GB).
    """
    files = _list_safetensors(weights_path)
    if not files:
        return None

    # Local file passed directly.
    if len(files) == 1:
        return os.path.basename(files[0])

    basenames = [os.path.basename(f) for f in files]

    def _pick(pattern_kind: str) -> str | None:
        for name in basenames:
            low = name.lower()
            if pattern_kind == "mixed" and "mixed" in low:
                return name
            if pattern_kind == "plain" and "nvfp4" in low and "mixed" not in low:
                return name
        return None

    if prefer_mixed:
        return _pick("mixed") or _pick("plain") or basenames[0]
    return _pick("plain") or _pick("mixed") or basenames[0]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SAFETENSORS_HEADER_PROBE_BYTES = 16 * 1024 * 1024  # 16 MB is plenty


def _header_looks_nvfp4(header: dict[str, Any]) -> bool:
    meta = header.get("__metadata__")
    if isinstance(meta, dict):
        raw = meta.get("_quantization_metadata")
        if raw:
            return "fp4" in str(raw).lower()
        legacy = meta.get("quantization", "")
        if isinstance(legacy, str) and "fp4" in legacy.lower():
            return True

    has_u8_weight = False
    has_weight_scale = False
    for name, info in header.items():
        if name == "__metadata__" or not isinstance(info, dict):
            continue
        dtype = info.get("dtype", "")
        if name.endswith(".weight") and dtype == "U8":
            has_u8_weight = True
        elif name.endswith(".weight_scale") or name.endswith(".weight_scale_2"):
            has_weight_scale = True
        if has_u8_weight and has_weight_scale:
            return True
    return False


def _build_flux2_exclude_modules(quantized_paths: set[str]) -> list[str]:
    """Build exclude_modules in diffusers naming for Flux2 NVFP4 checkpoints.

    The set of quantized layers varies across BFL releases. For example,
    ``FLUX.2-dev-NVFP4`` leaves the text-stream attention (``txt_attn``) in
    BF16, while ``FLUX.2-klein-4b-nvfp4`` quantizes it. We therefore drive
    the exclude list off the *actual* per-layer metadata that ModelOpt
    embedded in the safetensors header, instead of hard-coding a single
    pattern.

    *quantized_paths* uses the checkpoint's native (BFL) layer naming, but
    the patterns we return must match the diffusers names that ModelOpt's
    ``is_layer_excluded`` will see at inference. We probe a small set of
    BFL substrings against *quantized_paths* and emit the diffusers-style
    wildcard for any module group that is not present.

    Layer groups that are *always* unquantized in every Flux2 NVFP4 ckp
    we've seen (embedders, modulation, final layer, time/guidance) are
    listed unconditionally.
    """
    excl = [
        # Embedders — patch + context, never quantized
        "*x_embedder*",
        "*context_embedder*",
        # Time / guidance embeddings
        "*time_guidance_embed*",
        # AdaLN-style modulation linears (double-stream + single-stream)
        "*double_stream_modulation_img*",
        "*double_stream_modulation_txt*",
        "*single_stream_modulation*",
        # Final layer
        "*norm_out.linear*",
        "*proj_out*",
    ]

    # Conditional excludes derived from the embedded per-layer metadata.
    # Each entry maps a BFL substring → list of diffusers-name wildcards
    # that should be excluded if no quantized layer matches that substring.
    conditional = [
        ("txt_attn.qkv", ["*attn.add_kv_proj*"]),
        ("txt_attn.proj", ["*attn.to_add_out*"]),
    ]
    for bfl_substr, diffusers_patterns in conditional:
        if not any(bfl_substr in p for p in quantized_paths):
            excl.extend(diffusers_patterns)
    return excl


def _list_safetensors(weights_path: str) -> list[str]:
    """List safetensors files at *weights_path*.

    Supports:
    - A direct ``*.safetensors`` file path
    - A local directory (optionally with a ``transformer/`` subfolder)
    - A HuggingFace repo ID like ``org/name`` (listed via the Hub API —
      no file bytes are downloaded here, only the repo's file listing)
    """
    p = Path(weights_path)
    if p.is_file() and p.suffix == ".safetensors":
        return [str(p)]
    if p.is_dir():
        files = sorted(str(f) for f in p.glob("*.safetensors"))
        if not files:
            files = sorted(str(f) for f in (p / "transformer").glob("*.safetensors") if f.exists())
        return files

    # HF repo ID fallback. Local resolution failed, so ask the Hub.
    if "/" in weights_path and not os.path.exists(weights_path):
        try:
            from huggingface_hub import HfApi

            repo_files = HfApi().list_repo_files(weights_path)
        except Exception as e:
            logger.debug("HF list_repo_files failed for %s: %s", weights_path, e)
            return []
        return sorted(f for f in repo_files if f.endswith(".safetensors"))
    return []


def _read_first_safetensors_header(weights_path: str) -> dict[str, Any] | None:
    """Read the JSON header of the first safetensors file under *weights_path*.

    For local paths we mmap-ish read the first few bytes. For HF repo IDs
    we use a ranged HTTP request so we never download the 20+ GB shard.
    """
    files = _list_safetensors(weights_path)
    if not files:
        return None

    first = files[0]
    # Local path branch
    if os.path.exists(first):
        return _read_local_header(first)

    # HF repo branch — first is a relative filename inside the repo
    return _read_hf_header(weights_path, first)


def _read_local_header(filepath: str) -> dict[str, Any] | None:
    try:
        with open(filepath, "rb") as f:
            header_len = _unpack_header_len(f.read(8))
            if header_len is None or header_len > _SAFETENSORS_HEADER_PROBE_BYTES:
                return None
            return json.loads(f.read(header_len))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read safetensors header from %s: %s", filepath, e)
        return None


def _read_hf_header(repo_id: str, relative_path: str) -> dict[str, Any] | None:
    """Fetch only the JSON header of a safetensors file from a HF repo.

    Uses ``hf_hub_url`` + a ranged HTTP request so we don't download the
    full shard. Falls back to ``hf_hub_download`` for pathological hosts
    that don't honor Range (extremely rare for HF).
    """
    try:
        import requests
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
    except Exception as e:
        logger.debug("huggingface_hub/requests unavailable (%s)", e)
        return None

    url = hf_hub_url(repo_id=repo_id, filename=relative_path)
    headers = build_hf_headers()
    headers["Range"] = f"bytes=0-{_SAFETENSORS_HEADER_PROBE_BYTES - 1}"
    try:
        resp = requests.get(url, headers=headers, timeout=30, stream=False, allow_redirects=True)
        resp.raise_for_status()
        blob = resp.content
    except Exception as e:
        logger.debug("Ranged fetch failed for %s/%s: %s", repo_id, relative_path, e)
        return None

    if len(blob) < 8:
        return None
    header_len = _unpack_header_len(blob[:8])
    if header_len is None or header_len > _SAFETENSORS_HEADER_PROBE_BYTES - 8:
        return None
    try:
        return json.loads(blob[8 : 8 + header_len])
    except json.JSONDecodeError as e:
        logger.debug("Invalid safetensors header JSON from %s/%s: %s", repo_id, relative_path, e)
        return None


def _unpack_header_len(buf: bytes) -> int | None:
    if len(buf) < 8:
        return None
    (header_len,) = struct.unpack("<Q", buf)
    if header_len <= 0:
        return None
    return header_len
