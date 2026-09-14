# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.attention import layer
from vllm_omni.diffusion.attention.backends.fastvideo_vsa import FastVideoVSABackend, FastVideoVSAImpl
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.data import AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _SpecializedImpl(FastVideoVSAImpl):
    pass


class _SpecializedBackend(FastVideoVSABackend):
    @staticmethod
    def get_impl_cls():
        return _SpecializedImpl


@pytest.fixture
def local_attention(monkeypatch):
    monkeypatch.setattr(layer, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(layer, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_specialization_preserves_selected_backend_and_options(mocker, local_attention, explicit, override):
    spec = AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=7) if explicit else None
    select = mocker.patch.object(layer, "get_attn_backend_for_role", return_value=(FastVideoVSABackend, spec))
    attention = layer.Attention(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        role="video.self",
        role_category="self",
        qkv_layout="BSND",
        backend_overrides={"FASTVIDEO_VSA": _SpecializedBackend} if override else None,
    )

    assert attention.attn_backend is (_SpecializedBackend if override else FastVideoVSABackend)
    assert type(attention.attention) is (_SpecializedImpl if override else FastVideoVSAImpl)
    assert attention.attention.topk == (7 if explicit else 64)
    assert attention.attention.qkv_layout == "BSND"
    assert attention.backend_pref == "FASTVIDEO_VSA"
    assert attention.backend_explicit is explicit
    assert attention.attn_spec is spec
    assert isinstance(attention.parallel_strategy, NoParallelAttention)
    select.assert_called_once_with(
        role="video.self", head_size=8, attention_config=None, role_category="self", allow_trtllm_default=False
    )


def test_unselected_specialization_does_not_replace_dense_backend(mocker, local_attention):
    mocker.patch.object(layer, "get_attn_backend_for_role", return_value=(SDPABackend, None))
    attention = layer.Attention(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_overrides={"FASTVIDEO_VSA": _SpecializedBackend},
    )
    assert attention.attn_backend is SDPABackend
    assert type(attention.attention) is SDPABackend.get_impl_cls()


def test_specialization_does_not_bypass_selection_errors(mocker, local_attention):
    mocker.patch.object(layer, "get_attn_backend_for_role", side_effect=ImportError("optional kernel unavailable"))
    with pytest.raises(ImportError, match="optional kernel unavailable"):
        layer.Attention(
            num_heads=2,
            head_size=8,
            softmax_scale=8**-0.5,
            causal=False,
            backend_overrides={"FASTVIDEO_VSA": _SpecializedBackend},
        )


def test_specialization_must_extend_selected_backend(mocker, local_attention):
    mocker.patch.object(layer, "get_attn_backend_for_role", return_value=(FastVideoVSABackend, None))
    with pytest.raises(TypeError, match="must subclass the selected attention backend"):
        layer.Attention(
            num_heads=2,
            head_size=8,
            softmax_scale=8**-0.5,
            causal=False,
            backend_overrides={"FASTVIDEO_VSA": SDPABackend},
        )
