# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest
import torch
from vllm.platforms.interface import DeviceCapability

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.envs import PACKAGES_CHECKER
from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _blackwell_sm120(monkeypatch: pytest.MonkeyPatch, *, cudnn_version: int = 90500) -> None:
    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(12, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: cudnn_version)


def _install_dummy_flashinfer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))


def test_auto_selects_cudnn_for_supported_blackwell_head_size(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=128)

    assert path == DiffusionAttentionBackendEnum.CUDNN_ATTN.get_path()


def test_auto_routes_incompatible_head_size_to_sdpa(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)
    _install_dummy_flashinfer(monkeypatch)

    # 320 is outside cuDNN FMHA (<=256, multiple of 8) and FlashInfer {64,128,256}.
    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=320)

    assert path == DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()


def test_auto_routes_non_multiple_head_size_to_sdpa_without_flashinfer(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)
    monkeypatch.setitem(sys.modules, "flashinfer", None)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=12)

    assert path == DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()


def test_auto_selects_flashinfer_when_cudnn_too_old(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch, cudnn_version=90499)
    _install_dummy_flashinfer(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=128)

    assert path == DiffusionAttentionBackendEnum.FLASHINFER_ATTN.get_path()


def test_auto_skips_flashinfer_for_unsupported_head_size_when_cudnn_too_old(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch, cudnn_version=90499)
    _install_dummy_flashinfer(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls(None, head_size=72)

    assert path == DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()


def test_explicit_cudnn_raises_for_incompatible_head_size(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)

    with pytest.raises(ValueError, match="head_size=12 is unsupported"):
        CudaOmniPlatform.get_diffusion_attn_backend_cls("CUDNN_ATTN", head_size=12)


def test_explicit_cudnn_accepts_supported_head_size(monkeypatch: pytest.MonkeyPatch):
    _blackwell_sm120(monkeypatch)

    path = CudaOmniPlatform.get_diffusion_attn_backend_cls("CUDNN_ATTN", head_size=72)

    assert path == DiffusionAttentionBackendEnum.CUDNN_ATTN.get_path()
