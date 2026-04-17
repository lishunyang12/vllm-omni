# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.layers.rope import RotaryEmbedding

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_cuda_rope_accepts_3d_query(monkeypatch):
    import vllm.vllm_flash_attn.layers.rotary as rotary

    def fake_apply_rotary_emb(x, cos, sin, interleaved=False):
        assert x.shape == (1, 4, 2, 8)
        assert cos.shape == (4, 4)
        assert sin.shape == (4, 4)
        return x + 1

    monkeypatch.setattr(rotary, "apply_rotary_emb", fake_apply_rotary_emb)

    rope = RotaryEmbedding(is_neox_style=False)
    x = torch.zeros(4, 2, 8)
    cos = torch.zeros(4, 4)
    sin = torch.zeros(4, 4)

    out = rope.forward_cuda(x, cos, sin)

    assert out.shape == x.shape
    assert torch.equal(out, torch.ones_like(x))
