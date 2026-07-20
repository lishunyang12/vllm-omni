# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the skip-softmax sparse-attention config subsystem.

The module-name matching in ``apply.py`` was the source of a silent per-expert no-op
(skip never engaged because layer prefixes did not carry ``transformer.``). These tests
pin the matching contract in isolation so that class of bug cannot regress.
"""

import pytest

from vllm_omni.diffusion.attention.sparse_attention import (
    is_ignored,
    parse_sparse_attention_config,
    resolve_calibration,
    resolve_layer_calibration,
    select_expert,
)
from vllm_omni.diffusion.attention.sparse_attention.apply import layer_match_names


# --- pattern matching primitives ---------------------------------------------------


def test_layer_match_names_strips_expert_prefix():
    names = layer_match_names("transformer_2.blocks.5.attn1")
    assert "transformer_2.blocks.5.attn1" in names  # full name
    assert "blocks.5.attn1" in names                # relative (for relative ignore patterns)


def test_layer_match_names_strips_orig_mod():
    # torch.compile wraps submodules as `._orig_mod.`; matching must see through it.
    names = layer_match_names("transformer.blocks.5._orig_mod.attn1")
    assert "transformer.blocks.5.attn1" in names or "blocks.5.attn1" in names


def test_is_ignored_matches_relative_pattern():
    # ignore pattern is relative (no expert prefix); full module name still matches.
    assert is_ignored("transformer.blocks.5.attn2", ["blocks.*.attn2"])
    assert not is_ignored("transformer.blocks.5.attn1", ["blocks.*.attn2"])


def test_is_ignored_empty_patterns():
    assert not is_ignored("transformer.blocks.5.attn1", [])
    assert not is_ignored("transformer.blocks.5.attn1", None)


def test_is_ignored_ancestor_match_with_attn_suffix():
    # Our skippable unit is the inner Attention layer, one level below the ModelOpt-
    # calibrated module: full name `transformer.blocks.0.attn2.attn`, ignore `blocks.0.attn2`.
    assert is_ignored("transformer.blocks.0.attn2.attn", ["blocks.0.attn2"])
    # boundary-safe: blocks.1 must NOT match blocks.11
    assert not is_ignored("transformer.blocks.11.attn2.attn", ["blocks.1.attn2"])
    # a non-ignored self-attn layer is not ignored
    assert not is_ignored("transformer.blocks.5.attn1.attn", ["blocks.0.attn2", "blocks.1.attn2"])


def test_select_expert_longest_prefix_wins():
    keys = ("transformer", "transformer_2")
    assert select_expert("transformer_2.blocks.5.attn1", keys) == "transformer_2"
    assert select_expert("transformer.blocks.5.attn1", keys) == "transformer"


def test_select_expert_no_match():
    assert select_expert("blocks.5.attn1", ("transformer", "transformer_2")) is None


# --- per-layer resolution (the bug's blast radius) ---------------------------------


def _two_expert_calib():
    return {
        "by_expert": {
            "transformer": {"a": 2142.7, "b": 4.28, "target_sparsity": 0.45,
                            "formula": "a*exp(b*target_sparsity)", "ignore": ["blocks.*.attn2"]},
            "transformer_2": {"a": 314.68, "b": 6.17, "target_sparsity": 0.45,
                              "formula": "a*exp(b*target_sparsity)", "ignore": ["blocks.*.attn2"]},
        }
    }


def test_resolve_routes_to_correct_expert():
    calib = _two_expert_calib()
    hi = resolve_layer_calibration("transformer.blocks.5.attn1", calib)
    lo = resolve_layer_calibration("transformer_2.blocks.5.attn1", calib)
    assert hi["a"] == pytest.approx(2142.7)   # high-noise expert
    assert lo["a"] == pytest.approx(314.68)   # low-noise expert


def test_resolve_ignored_layer_is_dense():
    # cross-attn (attn2) is in the ignore set -> stays dense (None).
    assert resolve_layer_calibration("transformer.blocks.5.attn2", _two_expert_calib()) is None


def test_resolve_single_transformer():
    calib = {"a": 104.76, "b": 7.81, "target_sparsity": 0.45,
             "formula": "a*exp(b*target_sparsity)", "ignore": ["refiner_blocks.*"]}
    assert resolve_layer_calibration("transformer_blocks.5.attn", calib)["a"] == pytest.approx(104.76)
    assert resolve_layer_calibration("refiner_blocks.0.attn", calib) is None


def test_resolve_none_calibration():
    assert resolve_layer_calibration("transformer.blocks.5.attn1", None) is None


# --- config parsing ----------------------------------------------------------------


def test_parse_045_config_groups_schema():
    sac = {
        "config_groups": {
            "group_0": {
                "algorithm": "skip_softmax",
                "ignore": ["blocks.0.attn1"],
                "target_sparsity": 0.45,
                "threshold_scale_factor": {
                    "formula": "a * exp(b * target_sparsity)",
                    "coefficients": {"a": 2142.7, "b": 4.28},
                },
            }
        }
    }
    parsed = parse_sparse_attention_config(sac)
    assert parsed["a"] == pytest.approx(2142.7)
    assert parsed["b"] == pytest.approx(4.28)
    assert parsed["target_sparsity"] == pytest.approx(0.45)
    assert parsed["ignore"] == ["blocks.0.attn1"]


def test_parse_unsupported_formula_raises():
    sac = {"config_groups": {"group_0": {"algorithm": "skip_softmax",
            "threshold_scale_factor": {"formula": "a * b ** target_sparsity",
                                       "coefficients": {"a": 1.0, "b": 2.0}}}}}
    with pytest.raises(ValueError, match="unsupported skip-softmax formula"):
        parse_sparse_attention_config(sac)


def test_parse_empty_returns_none():
    assert parse_sparse_attention_config(None) is None
    assert parse_sparse_attention_config({}) is None


# --- registry (hosted data) --------------------------------------------------------


def test_registry_resolves_wan_a14b_two_experts():
    calib = resolve_calibration("/models/Wan2.2-T2V-A14B-Diffusers")
    assert set(calib["by_expert"]) == {"transformer", "transformer_2"}
    assert calib["by_expert"]["transformer"]["a"] == pytest.approx(2142.7334, rel=1e-4)
    assert calib["by_expert"]["transformer_2"]["a"] == pytest.approx(314.6824, rel=1e-4)


def test_registry_unknown_model_is_none():
    assert resolve_calibration("/models/some-uncalibrated-model") is None
    assert resolve_calibration(None) is None


# --- post-build applier walk (the fix for the silent per-expert no-op) -------------


class _FakeImpl:
    """Stands in for TrtllmGenImpl: records the stamped (a, b)."""

    def __init__(self):
        self.stamped = None

    def set_layer_calibration(self, a, b):
        self.stamped = (a, b)


def _fake_pipeline():
    """A minimal nn.Module tree mirroring Wan A14B: two experts, each with a self-attn
    (attn1) and cross-attn (attn2) whose inner Attention layer holds the impl at `.attention`."""
    import torch.nn as nn

    def _attn_layer():
        layer = nn.Module()
        layer.attention = _FakeImpl()
        return layer

    def _wan_attn():
        wa = nn.Module()
        wa.attn = _attn_layer()  # inner Attention lives at `.attn`
        return wa

    def _transformer():
        tf = nn.Module()
        blocks = nn.ModuleList()
        for _ in range(3):
            blk = nn.Module()
            blk.attn1 = _wan_attn()
            blk.attn2 = _wan_attn()
            blocks.append(blk)
        tf.blocks = blocks
        return tf

    pipe = nn.Module()
    pipe.transformer = _transformer()
    pipe.transformer_2 = _transformer()
    return pipe


def test_apply_to_pipeline_routes_and_ignores():
    from vllm_omni.diffusion.attention.sparse_attention import apply_to_pipeline

    pipe = _fake_pipeline()
    # ignore all cross-attn (attn2) + block-0 self-attn endpoints
    calib = {
        "by_expert": {
            "transformer": {"a": 2142.7, "b": 4.28, "target_sparsity": 0.5,
                            "formula": "a*exp(b*target_sparsity)",
                            "ignore": ["blocks.*.attn2", "blocks.0.attn1"]},
            "transformer_2": {"a": 314.68, "b": 6.17, "target_sparsity": 0.5,
                              "formula": "a*exp(b*target_sparsity)",
                              "ignore": ["blocks.*.attn2", "blocks.0.attn1"]},
        }
    }
    stamped = apply_to_pipeline(pipe, calib)

    # 3 blocks x 2 experts, self-attn only, minus block-0 endpoints (2) = 4 stamped
    assert stamped == 4

    def impl(expert, blk, attn):
        return getattr(getattr(pipe, expert).blocks[blk], attn).attn.attention

    # high-noise expert self-attn (non-endpoint) -> transformer curve
    assert impl("transformer", 1, "attn1").stamped == pytest.approx((2142.7, 4.28))
    # low-noise expert self-attn -> transformer_2 curve (routing by name)
    assert impl("transformer_2", 1, "attn1").stamped == pytest.approx((314.68, 6.17))
    # cross-attn stays dense (never stamped)
    assert impl("transformer", 1, "attn2").stamped is None
    # block-0 self-attn endpoint stays dense
    assert impl("transformer", 0, "attn1").stamped is None
