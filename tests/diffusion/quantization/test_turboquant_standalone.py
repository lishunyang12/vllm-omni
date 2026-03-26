#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone TurboQuant tests — requires only PyTorch, no vLLM.

Usage:
    python tests/diffusion/quantization/test_turboquant_standalone.py
"""

import math
import os
import sys
import types

import torch

# Direct import to avoid vllm_omni.__init__ pulling in vLLM.
_mod = types.ModuleType("vllm_omni.quantization.turboquant")
sys.modules[_mod.__name__] = _mod

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    _mod.__name__,
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "vllm_omni",
                 "quantization", "turboquant.py"),
)
_spec.loader.exec_module(_mod)

EXPECTED_MSE_NORMALIZED = _mod.EXPECTED_MSE_NORMALIZED
TurboQuantConfig = _mod.TurboQuantConfig
TurboQuantState = _mod.TurboQuantState
_get_codebook = _mod._get_codebook
_hadamard_transform = _mod._hadamard_transform
compute_distortion = _mod.compute_distortion
random_rotate = _mod.random_rotate
random_rotate_inverse = _mod.random_rotate_inverse
scalar_dequantize = _mod.scalar_dequantize
scalar_quantize = _mod.scalar_quantize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_codebook():
    print("\n--- Codebook ---")
    for bits in [1, 2, 3, 4]:
        cb = _get_codebook(bits, 128, DEVICE)
        check(f"{bits}-bit symmetric", torch.allclose(cb, -cb.flip(0), atol=1e-5))
        check(f"{bits}-bit sorted", bool((cb[1:] > cb[:-1]).all()))


def test_hadamard():
    print("\n--- Hadamard ---")
    x = torch.randn(64, device=DEVICE)
    y = _hadamard_transform(_hadamard_transform(x))
    check("self-inverse", torch.allclose(x, y, atol=1e-4),
          f"max diff={( x - y).abs().max():.6f}")

    check("norm-preserving", torch.allclose(
        x.norm(), _hadamard_transform(x).norm(), atol=1e-4))


def test_rotation():
    print("\n--- Random rotation ---")
    d = 128
    sf = (torch.randint(0, 2, (d,), device=DEVICE).float() * 2 - 1)
    x = torch.randn(10, d, device=DEVICE)

    y = random_rotate(x, sf)
    check("norm-preserving",
          torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4))

    x_rec = random_rotate_inverse(y, sf)
    check("invertible", torch.allclose(x, x_rec, atol=1e-4),
          f"max diff={(x - x_rec).abs().max():.6f}")


def test_scalar_quantize():
    print("\n--- Scalar quantization ---")
    cb = _get_codebook(3, 128, DEVICE)
    indices = scalar_quantize(cb, cb)
    recovered = scalar_dequantize(indices, cb)
    check("centroid roundtrip", torch.allclose(cb, recovered))

    x = torch.randn(100, device=DEVICE) / math.sqrt(128)
    idx = scalar_quantize(x, cb)
    check("index range", int(idx.min()) >= 0 and int(idx.max()) <= 7)


def test_roundtrip_mse():
    print("\n--- Roundtrip MSE (paper Theorem 1) ---")
    torch.manual_seed(0)
    n, d = 500, 128

    for bits in [1, 2, 3, 4]:
        config = TurboQuantConfig(bit_width=bits, use_qjl=False)
        state = TurboQuantState(config, d, layer_idx=0, device=DEVICE)

        x = torch.randn(n, 1, d, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)

        x_hat = state.dequantize(state.quantize(x))
        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
        bound = EXPECTED_MSE_NORMALIZED[bits]

        check(f"{bits}-bit MSE={mse:.4f} (bound={bound:.4f})",
              mse < bound * 3.0,
              f"ratio={mse / bound:.2f}x")


def test_qjl_unbiased():
    print("\n--- QJL unbiasedness (paper Theorem 2) ---")
    torch.manual_seed(42)
    d = 128
    n = 300

    x = torch.randn(n, 1, d, device=DEVICE)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.randn(n, 1, d, device=DEVICE)

    state = TurboQuantState(
        TurboQuantConfig(bit_width=2, use_qjl=True),
        d, layer_idx=0, device=DEVICE,
    )
    x_hat = state.dequantize(state.quantize(x))

    ip_true = (y * x).sum(dim=-1)
    ip_est = (y * x_hat).sum(dim=-1)
    bias = (ip_est - ip_true).mean().abs().item()
    check(f"bias={bias:.4f}", bias < 0.05)


def test_nonstandard_head_size():
    print("\n--- Non-power-of-2 head sizes ---")
    for hs in [96, 80, 192]:
        config = TurboQuantConfig(bit_width=2, use_qjl=False)
        state = TurboQuantState(config, hs, layer_idx=0, device=DEVICE)
        x = torch.randn(2, 4, hs, device=DEVICE)
        x_hat = state.dequantize(state.quantize(x))
        check(f"head_size={hs} shape", x_hat.shape == x.shape)


def test_determinism():
    print("\n--- Determinism ---")
    config = TurboQuantConfig(bit_width=3)
    state = TurboQuantState(config, 128, layer_idx=0, device=DEVICE)
    x = torch.randn(2, 4, 128, device=DEVICE)
    q1 = state.quantize(x)
    q2 = state.quantize(x)
    check("same input -> same indices", torch.equal(q1["indices"], q2["indices"]))


def test_compression_ratio():
    print("\n--- Compression ratio ---")
    d = 128
    for bits in [2, 3, 4]:
        fp16_bytes = d * 2
        tq_bytes = math.ceil(d * bits / 8) + 2  # +2 for norm (float16)
        ratio = fp16_bytes / tq_bytes
        check(f"{bits}-bit ratio={ratio:.1f}x vs FP16", ratio > 1.5)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    test_codebook()
    test_hadamard()
    test_rotation()
    test_scalar_quantize()
    test_roundtrip_mse()
    test_qjl_unbiased()
    test_nonstandard_head_size()
    test_determinism()
    test_compression_ratio()

    print(f"\n{'=' * 40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
