# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skip-softmax sparse-attention config subsystem: hosted per-model calibrations
(registry + data) and their per-layer resolution. The trtllm kernel backend consumes
only the resolved {a, b}; it knows nothing about hosting, files, or module matching."""

from vllm_omni.diffusion.attention.sparse_attention.apply import (
    apply_to_pipeline,
    is_ignored,
    layer_match_names,
    resolve_layer_calibration,
    select_expert,
)
from vllm_omni.diffusion.attention.sparse_attention.config import parse_sparse_attention_config
from vllm_omni.diffusion.attention.sparse_attention.registry import resolve_calibration

__all__ = [
    "resolve_calibration",
    "resolve_layer_calibration",
    "apply_to_pipeline",
    "parse_sparse_attention_config",
    "is_ignored",
    "layer_match_names",
    "select_expert",
]
