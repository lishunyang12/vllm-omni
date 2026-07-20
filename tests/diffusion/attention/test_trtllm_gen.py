# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the trtllm diffusion attention backend (BF16 + Skip-Softmax).

Logic tests (skip-factor resolution, timestep gate) run on CPU. Kernel tests
(BF16 parity vs SDPA, skip, GQA, unsupported-head-dim raises) require a Blackwell
SM100+ GPU with flashinfer and are skipped otherwise.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends import trtllm_gen as tg
from vllm_omni.diffusion.attention.backends.trtllm_gen import TrtllmGenImpl


def _has_trtllm_gen() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 10:
        return False
    try:
        import flashinfer  # noqa: F401
    except Exception:
        return False
    return tg.HAS_FLASHINFER


requires_trtllm_gen = pytest.mark.skipif(
    not _has_trtllm_gen(), reason="requires Blackwell SM100+ GPU with flashinfer"
)


# ----------------------------- logic (CPU) -----------------------------

def _impl(**bk):
    return TrtllmGenImpl(8, 128, 1.0 / math.sqrt(128), causal=False, num_kv_heads=8, backend_kwargs=bk)


def test_skip_config_pure_resolution():
    # SkipSoftmaxConfig is pure: resolve_factor takes the timestep explicitly, no
    # ForwardContext needed. Covers enabled/gated and the dense guard window.
    from vllm_omni.diffusion.attention.backends.trtllm_gen import SkipSoftmaxConfig

    assert not SkipSoftmaxConfig().enabled
    assert SkipSoftmaxConfig().resolve_factor(4096, timestep=0.3) is None

    cfg = SkipSoftmaxConfig.from_backend_kwargs(
        {"skip_softmax_threshold": 0.01, "disabled_until_timestep": 0.6}
    )
    assert cfg.enabled and cfg.gated
    assert cfg.resolve_factor(4096, timestep=0.9) is None  # early/noisy -> dense guard
    assert cfg.resolve_factor(4096, timestep=0.3) == pytest.approx(0.01 * 4096)  # late -> skip
    assert cfg.resolve_factor(4096, timestep=None) == pytest.approx(0.01 * 4096)  # no guard


def test_skip_factor_none_without_curve():
    # target_sparsity set but no calibrated (a, b) and no threshold -> no skip
    assert _impl(target_sparsity=0.5)._resolve_skip_factor(4096) is None


def _calibrated(a, b, **bk):
    # a, b are stamped post-build by the applier (set_layer_calibration), not passed as
    # backend_kwargs -- they are per-layer/per-expert and only resolvable by module name.
    impl = _impl(**bk)
    impl.set_layer_calibration(a, b)
    return impl


def test_skip_factor_from_curve():
    assert _calibrated(100.0, 0.0, target_sparsity=0.5)._resolve_skip_factor(4096) == pytest.approx(100.0)
    f2 = _calibrated(2.0, 1.0, target_sparsity=0.5)._resolve_skip_factor(4096)
    assert f2 == pytest.approx(2.0 * math.exp(0.5))


def test_skip_factor_none_until_stamped():
    # A calibrated layer that was never stamped (e.g. an ignored/cross-attn layer the
    # applier skipped) stays dense even with target_sparsity set -- no wrong-threshold guess.
    assert _impl(target_sparsity=0.5)._resolve_skip_factor(4096) is None


def test_skip_factor_from_threshold_no_calibration():
    # skip_softmax_threshold enables skip with no calibration: the kernel uses
    # threshold = factor / seqlen, so factor = threshold * seqlen.
    impl = _impl(skip_softmax_threshold=0.01)
    assert impl._resolve_skip_factor(4096) == pytest.approx(0.01 * 4096)
    assert impl._resolve_skip_factor(8192) == pytest.approx(0.01 * 8192)
    # direct threshold takes priority over the calibrated curve
    impl2 = _calibrated(100.0, 0.0, skip_softmax_threshold=0.02, target_sparsity=0.5)
    assert impl2._resolve_skip_factor(4096) == pytest.approx(0.02 * 4096)


def test_skip_timestep_gate(monkeypatch):
    from vllm_omni.diffusion import forward_context as fc

    impl = _calibrated(100.0, 0.0, target_sparsity=0.5, disabled_until_timestep=0.6)
    ctx = fc.ForwardContext(denoise_timestep=0.9)  # early / noisy
    monkeypatch.setattr(fc, "_forward_context", ctx)
    assert impl._resolve_skip_factor(4096) is None  # in the dense guard window
    ctx.denoise_timestep = 0.3  # late / clean
    assert impl._resolve_skip_factor(4096) == pytest.approx(100.0)


def test_denoise_progress_mixin(monkeypatch):
    # The mixin publishes both step idx and normalized timestep in one call,
    # reading self.scheduler; timestep is normalized by num_train_timesteps.
    import types

    from vllm_omni.diffusion import forward_context as fc

    class _Pipe(fc.DenoiseProgressMixin):
        scheduler = types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=1000))

    ctx = fc.ForwardContext()
    monkeypatch.setattr(fc, "_forward_context", ctx)
    _Pipe().record_denoise_step(3, 250)  # raw t=250 on a 1000-step schedule
    assert ctx.denoise_step_idx == 3
    assert ctx.denoise_timestep == pytest.approx(0.25)
    # timestep is optional: step idx still published, timestep left untouched.
    _Pipe().record_denoise_step(4)
    assert ctx.denoise_step_idx == 4 and ctx.denoise_timestep == pytest.approx(0.25)


# Per-layer ignore matching and per-expert routing now live in
# sparse_attention.apply (tested in test_sparse_attention.py). The backend impl no longer
# matches by name -- it only holds the resolved curve stamped onto it.


def test_propagate_calibration_045_schema():
    # The 0.45 ModelOpt schema nests calibration under config_groups.group_0. data.py parses
    # it and stashes the whole curve on spec.extra['skip_calibration'] for the applier; a,b
    # are NOT flattened onto the spec (they are resolved per-layer at apply time).
    import types

    from vllm_omni.diffusion.data import (
        AttentionConfig,
        AttentionSpec,
        OmniDiffusionConfig,
        TransformerConfig,
    )

    cfg = AttentionConfig(default=AttentionSpec(backend="TRTLLM", extra={}))
    tf = TransformerConfig.from_dict({
        "sparse_attention_config": {"config_groups": {"group_0": {
            "algorithm": "skip_softmax",
            "targets": ["WanAttention"],
            "ignore": ["blocks.0.attn2", "blocks.1.attn2"],
            "threshold_scale_factor": {"formula": "a * exp(b * target_sparsity)",
                                       "coefficients": {"a": 2142.7, "b": 4.28}},
            "target_sparsity": 0.5,
        }}}
    })
    OmniDiffusionConfig._propagate_skip_softmax_calibration(
        types.SimpleNamespace(diffusion_attention_config=cfg, model="/x"), tf
    )
    spec, _ = cfg.resolve_with_source("self")
    cal = spec.extra["skip_calibration"]
    assert cal["a"] == pytest.approx(2142.7)
    assert cal["b"] == pytest.approx(4.28)
    assert cal["ignore"] == ["blocks.0.attn2", "blocks.1.attn2"]
    assert spec.extra["target_sparsity"] == pytest.approx(0.5)


def test_target_sparsity_without_calibration_raises():
    # target_sparsity on a model with no checkpoint config and no hosted calibration errors;
    # skip_softmax_threshold on the same model is fine (calibration-free).
    import types

    from vllm_omni.diffusion.data import (
        AttentionConfig,
        AttentionSpec,
        OmniDiffusionConfig,
        TransformerConfig,
    )

    def _cfg(extra):
        return types.SimpleNamespace(
            diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TRTLLM", extra=extra)),
            model="/models/no-such-calibration",
        )

    with pytest.raises(ValueError, match="no skip-softmax calibration"):
        OmniDiffusionConfig._propagate_skip_softmax_calibration(
            _cfg({"target_sparsity": 0.5}), TransformerConfig.from_dict({})
        )
    # calibration-free threshold path must NOT raise
    OmniDiffusionConfig._propagate_skip_softmax_calibration(
        _cfg({"skip_softmax_threshold": 0.02}), TransformerConfig.from_dict({})
    )


# --------------------------- kernel (B300) ----------------------------

def _sdpa_ref(q, k, v, scale):
    qp, kp, vp = (x.permute(0, 2, 1, 3) for x in (q, k, v))
    return F.scaled_dot_product_attention(qp, kp, vp, is_causal=False, scale=scale).permute(0, 2, 1, 3).float()


@requires_trtllm_gen
def test_bf16_dense_matches_sdpa():
    torch.manual_seed(0)
    B, S, H, D = 2, 256, 8, 128
    scale = 1.0 / math.sqrt(D)
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = _impl().forward_cuda(q, k, v, None).float()
    ref = _sdpa_ref(q, k, v, scale)
    rel = (out - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.01, f"BF16 dense rel err {rel:.4f} too high"


@requires_trtllm_gen
def test_skip_tiny_threshold_matches_dense():
    torch.manual_seed(0)
    B, S, H, D = 2, 512, 8, 128
    scale = 1.0 / math.sqrt(D)
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    # tiny factor -> skip enabled but ~nothing skipped -> ~= dense
    out = _impl(skip_softmax_threshold=1e-9).forward_cuda(q, k, v, None).float()
    ref = _sdpa_ref(q, k, v, scale)
    rel = (out - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.02, f"skip(tiny) rel err {rel:.4f} should be ~dense"


@requires_trtllm_gen
def test_skip_threshold_engages():
    """A large skip_softmax_threshold (calibration-free) actually skips: output
    diverges from dense and stays finite (no NaN). Random data is unstructured,
    so a high raw threshold (>=2) is needed to force skipping here."""
    torch.manual_seed(0)
    B, S, H, D = 2, 512, 8, 128
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    dense = _impl().forward_cuda(q, k, v, None).float()
    skipped = _impl(skip_softmax_threshold=2.0).forward_cuda(q, k, v, None).float()
    assert torch.isfinite(skipped).all()
    assert (skipped - dense).abs().mean() / dense.abs().mean() > 0.02


@requires_trtllm_gen
def test_gqa_matches_sdpa():
    # GQA (num_q_heads != num_kv_heads) is handled by the kernel natively (not a
    # fallback). Cosmos-shaped: 64 query heads, 8 KV heads, head_dim 128.
    torch.manual_seed(0)
    B, S, Hq, Hkv, D = 1, 256, 64, 8, 128
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
    out = TrtllmGenImpl(Hq, D, scale, causal=False, num_kv_heads=Hkv,
                        backend_kwargs={}).forward_cuda(q, k, v, None).float()
    qp, kp, vp = (x.permute(0, 2, 1, 3) for x in (q, k, v))
    ref = F.scaled_dot_product_attention(
        qp, kp, vp, is_causal=False, scale=scale, enable_gqa=True).permute(0, 2, 1, 3).float()
    rel = (out - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.01, f"GQA rel err {rel:.4f} vs SDPA(GQA) too high"


@requires_trtllm_gen
def test_fp8_attention_matches_bf16():
    # FP8 attention (extra.sage) quantizes Q/K/V to e4m3 and runs the FMHA in FP8;
    # output stays close to BF16 (quant noise only), finite, same shape.
    torch.manual_seed(0)
    B, S, H, D = 1, 512, 8, 128
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    bf16 = _impl().forward_cuda(q, k, v, None).float()
    fp8 = TrtllmGenImpl(H, D, 1.0 / math.sqrt(D), causal=False, num_kv_heads=H,
                        backend_kwargs={"sage": "per_tensor"}).forward_cuda(q, k, v, None).float()
    assert torch.isfinite(fp8).all() and fp8.shape == bf16.shape
    rel = (fp8 - bf16).abs().mean() / bf16.abs().mean()
    assert rel < 0.15, f"FP8 attn rel err {rel:.4f} too high"


@requires_trtllm_gen
def test_unsupported_head_dim_raises():
    # No silent SDPA fallback: an unsupported head_dim raises from the kernel itself
    # (trtllm_ragged_attention_deepseek asserts the supported head dims).
    q = k = v = torch.randn(1, 32, 8, 64, device="cuda", dtype=torch.bfloat16)  # head_dim 64
    with pytest.raises(Exception, match="[Uu]nsupported head dim"):
        _impl().forward_cuda(q, k, v, None)


def test_skip_end_to_end_config_path(monkeypatch):
    """Full skip flow: checkpoint calibration -> AttentionSpec.extra -> backend_kwargs
    -> resolved factor, gated by the model timestep."""
    import types

    from vllm_omni.diffusion import forward_context as fc
    from vllm_omni.diffusion.data import (
        AttentionConfig,
        AttentionSpec,
        OmniDiffusionConfig,
        TransformerConfig,
    )

    # user config: trtllm + skip operating point (no explicit a,b)
    cfg = AttentionConfig(
        default=AttentionSpec(
            backend="TRTLLM",
            extra={"target_sparsity": 0.65, "disabled_until_timestep": 0.86},
        )
    )
    # checkpoint carries the ModelOpt calibration curve
    tf = TransformerConfig.from_dict(
        {"sparse_attention_config": {"threshold_scale_factor": {"prefill": {"a": 7.93, "b": 8.61}}}}
    )
    OmniDiffusionConfig._propagate_skip_softmax_calibration(
        types.SimpleNamespace(diffusion_attention_config=cfg, model="/x"), tf
    )
    spec, _ = cfg.resolve_with_source("self")
    cal = spec.extra["skip_calibration"]
    assert cal["a"] == 7.93 and cal["b"] == 8.61

    # backend built from the resolved extra (as layer.py does), then the applier stamps
    # this layer's curve (as apply_to_pipeline does by module name).
    impl = TrtllmGenImpl(8, 128, 1.0 / math.sqrt(128), causal=False, num_kv_heads=8, backend_kwargs=spec.extra)
    impl.set_layer_calibration(cal["a"], cal["b"])
    expected = 7.93 * math.exp(8.61 * 0.65)

    ctx = fc.ForwardContext(denoise_timestep=0.3)  # late -> skip active
    monkeypatch.setattr(fc, "_forward_context", ctx)
    assert impl._resolve_skip_factor(4096) == pytest.approx(expected)
