# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry of developer-curated skip-softmax calibrations hosted in-repo (the TeaCache
_MODEL_COEFFICIENTS analog). Users load the raw checkpoint and pass target_sparsity; the
calibration is resolved here. No entry + a target_sparsity request -> the backend raises
(use skip_softmax_threshold for the calibration-free path)."""

import json
import os

from vllm_omni.diffusion.attention.sparse_attention.config import parse_sparse_attention_config

_CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrations")

# Keyed by a distinctive lowercase substring of the model path (sibling models share a
# pipeline class -- Wan A14B vs 5B are both "WanPipeline" -- so a path key is used).
# Multi-transformer models list one calibration file per expert.
_CALIBRATIONS: dict[str, dict[str, str]] = {
    "wan2.2-t2v-a14b": {
        "transformer": "wan2_2_a14b/transformer.json",
        "transformer_2": "wan2_2_a14b/transformer_2.json",
    },
    "hunyuanvideo-1.5": {
        "transformer": "hunyuan_video_15/transformer.json",
    },
    "cosmos3-super": {
        "transformer": "cosmos3_super/transformer.json",
    },
}


def _load(rel_path: str) -> dict:
    with open(os.path.join(_CALIB_DIR, rel_path)) as f:
        return parse_sparse_attention_config(json.load(f))


def resolve_calibration(model: str | None) -> dict | None:
    """Parsed calibration for a model, or None. Multi-expert models return
    ``{"by_expert": {expert_attr: parsed}}``; single-transformer models return the parsed
    dict directly. Match is a case-insensitive substring of the model path."""
    if not model:
        return None
    m = model.lower()
    for key, files in _CALIBRATIONS.items():
        if key in m:
            parsed = {expert: _load(rel) for expert, rel in files.items()}
            return next(iter(parsed.values())) if len(parsed) == 1 else {"by_expert": parsed}
    return None
