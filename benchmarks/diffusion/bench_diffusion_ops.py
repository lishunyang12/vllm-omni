# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import gc
import io
import math
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

DEFAULT_DEVICE = "cuda"
WARMUP_ITERS = 10
BENCH_ITERS = 30
EPS = 1e-6


def _update_bench_config(warmup: int, iters: int):
    global WARMUP_ITERS, BENCH_ITERS
    WARMUP_ITERS = warmup
    BENCH_ITERS = iters


_FA3_FUNC = None
_FA2_FUNC = None
_SAGE_FUNC = None


def _detect_fa3():
    global _FA3_FUNC
    if _FA3_FUNC is not None:
        return _FA3_FUNC
    try:
        from fa3_fwd_interface import flash_attn_func

        _FA3_FUNC = flash_attn_func
        return _FA3_FUNC
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from flash_attn_interface import flash_attn_func

        _FA3_FUNC = flash_attn_func
        return _FA3_FUNC
    except (ImportError, ModuleNotFoundError):
        pass
    return None


def _detect_fa2():
    global _FA2_FUNC
    if _FA2_FUNC is not None:
        return _FA2_FUNC
    try:
        from flash_attn import flash_attn_func

        _FA2_FUNC = flash_attn_func
        return _FA2_FUNC
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from flash_attn.flash_attn_interface import flash_attn_func

        _FA2_FUNC = flash_attn_func
        return _FA2_FUNC
    except (ImportError, ModuleNotFoundError):
        pass
    return None


def _detect_sage():
    global _SAGE_FUNC
    if _SAGE_FUNC is not None:
        return _SAGE_FUNC
    try:
        from sageattention import sageattn

        _SAGE_FUNC = sageattn
        return _SAGE_FUNC
    except (ImportError, ModuleNotFoundError):
        pass
    return None


def _try_import_flashinfer_norm():
    try:
        import flashinfer.norm as norm

        return norm
    except (ImportError, ModuleNotFoundError):
        return None


def _try_import_flaggems():
    try:
        from flag_gems.fused.fused_add_rms_norm import fused_add_rms_norm
        from flag_gems.ops.layernorm import layer_norm
        from flag_gems.ops.rms_norm import rms_norm

        return rms_norm, layer_norm, fused_add_rms_norm
    except (ImportError, ModuleNotFoundError):
        return None


def _try_import_quack():
    try:
        from quack.rmsnorm import layernorm_fwd, rmsnorm_fwd

        return rmsnorm_fwd, layernorm_fwd
    except (ImportError, ModuleNotFoundError):
        return None


def _try_import_triton_norm():
    try:
        from vllm_omni.diffusion.kernels.triton.norm import norm_infer

        return norm_infer
    except (ImportError, ModuleNotFoundError):
        return None


DIFFUSION_ATTN_SHAPES: list[dict[str, Any]] = [
    {
        "shape_id": "qwen_image_1024",
        "model": "qwen-image",
        "seq_len": 4096,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "1024x1024, 28 blocks",
    },
    {
        "shape_id": "qwen_image_2048",
        "model": "qwen-image",
        "seq_len": 16384,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "2048x2048, 28 blocks",
    },
    {
        "shape_id": "qwen_image_edit",
        "model": "qwen-image-edit",
        "seq_len": 8308,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "edit mode, dual-stream concat",
    },
    {
        "shape_id": "flux_4608",
        "model": "flux",
        "seq_len": 4608,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "1024x1024, joint attn (img+txt)",
    },
    {
        "shape_id": "flux2_4608",
        "model": "flux2",
        "seq_len": 4608,
        "num_heads": 48,
        "num_kv_heads": 48,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "1024x1024, 48 heads",
    },
    {
        "shape_id": "sd3_4096",
        "model": "sd3.5",
        "seq_len": 4096,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 64,
        "gpu_config": "1 GPU",
        "notes": "1024x1024, head_dim=64",
    },
    {
        "shape_id": "glm_image_4096",
        "model": "glm-image",
        "seq_len": 4096,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "1024x1024",
    },
    {
        "shape_id": "zimage_4096",
        "model": "z-image",
        "seq_len": 4096,
        "num_heads": 30,
        "num_kv_heads": 6,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "GQA (n_kv=6), hidden=3840",
    },
    {
        "shape_id": "zimage_4128",
        "model": "z-image",
        "seq_len": 4128,
        "num_heads": 30,
        "num_kv_heads": 6,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "GQA with padding",
    },
    {
        "shape_id": "wan_t2v_17850",
        "model": "wan-t2v",
        "seq_len": 17850,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "480p 81f, self-attn",
    },
    {
        "shape_id": "wan_t2v_44100",
        "model": "wan-t2v",
        "seq_len": 44100,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "720p 81f, self-attn",
    },
    {
        "shape_id": "wan_t2v_cross_512",
        "model": "wan-t2v",
        "seq_len": 512,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "cross-attn kv side (text enc)",
    },
    {
        "shape_id": "wan_t2v_cross_q17850_kv512",
        "model": "wan-t2v",
        "seq_len": 17850,
        "kv_seq_len": 512,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "cross-attn Q=visual, KV=text",
    },
    {
        "shape_id": "hunyuan_video_27030",
        "model": "hunyuan-video",
        "seq_len": 27030,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "480p, 54 layers",
    },
    {
        "shape_id": "ltx2_4096",
        "model": "ltx2",
        "seq_len": 4096,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 64,
        "gpu_config": "1 GPU",
        "notes": "head_dim=64, 32 heads",
    },
    {
        "shape_id": "text_enc_77",
        "model": "text-encoder",
        "seq_len": 77,
        "num_heads": 12,
        "num_kv_heads": 12,
        "head_dim": 64,
        "gpu_config": "1 GPU",
        "notes": "CLIP text encoder",
    },
    {
        "shape_id": "text_enc_512",
        "model": "text-encoder",
        "seq_len": 512,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 128,
        "gpu_config": "1 GPU",
        "notes": "T5/Qwen text encoder",
    },
]

DIFFUSION_NORM_SHAPES: list[dict[str, Any]] = [
    {
        "shape_id": "qwen_ln_4096x3072",
        "model": "qwen-image",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 4096, 3072],
    },
    {
        "shape_id": "qwen_ln_26x3072",
        "model": "qwen-image",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 26, 3072],
    },
    {
        "shape_id": "qwen_ln_6x3072",
        "model": "qwen-image",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 6, 3072],
    },
    {
        "shape_id": "qwen_rms_26x3584",
        "model": "qwen-image",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 26, 3584],
    },
    {
        "shape_id": "qwen_rms_6x3584",
        "model": "qwen-image",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 6, 3584],
    },
    {
        "shape_id": "qwen_edit_ln_189x3072",
        "model": "qwen-image-edit",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 189, 3072],
    },
    {
        "shape_id": "qwen_edit_ln_192x3072",
        "model": "qwen-image-edit",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 192, 3072],
    },
    {
        "shape_id": "qwen_edit_ln_8308x3072",
        "model": "qwen-image-edit",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 8308, 3072],
    },
    {
        "shape_id": "qwen_edit_rms_189x3584",
        "model": "qwen-image-edit",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 189, 3584],
    },
    {
        "shape_id": "qwen_edit_rms_192x3584",
        "model": "qwen-image-edit",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 192, 3584],
    },
    {
        "shape_id": "flux_ln_77x768",
        "model": "flux",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 77, 768],
    },
    {
        "shape_id": "flux_ln_512x3072",
        "model": "flux",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 512, 3072],
    },
    {
        "shape_id": "flux_ln_4096x3072",
        "model": "flux",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 4096, 3072],
    },
    {
        "shape_id": "flux_ln_4608x3072",
        "model": "flux",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 4608, 3072],
    },
    {
        "shape_id": "flux_rms_512x4096",
        "model": "flux",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 512, 4096],
    },
    {
        "shape_id": "flux2_ln_512x6144",
        "model": "flux2",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 512, 6144],
    },
    {
        "shape_id": "flux2_ln_4096x6144",
        "model": "flux2",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 4096, 6144],
    },
    {
        "shape_id": "flux2_ln_4608x6144",
        "model": "flux2",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 4608, 6144],
    },
    {
        "shape_id": "flux2_rms_4608x48x128",
        "model": "flux2",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 4608, 48, 128],
    },
    {
        "shape_id": "zimage_ln_4128x3840",
        "model": "z-image",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 4128, 3840],
    },
    {
        "shape_id": "zimage_rms_32x3840",
        "model": "z-image",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 32, 3840],
    },
    {
        "shape_id": "zimage_rms_4096x3840",
        "model": "z-image",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 4096, 3840],
    },
    {
        "shape_id": "zimage_rms_4128x3840",
        "model": "z-image",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 4128, 3840],
    },
    {
        "shape_id": "zimage_rms_512x2560",
        "model": "z-image",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 512, 2560],
    },
    {
        "shape_id": "zimage_rms_512x32x128",
        "model": "z-image",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 512, 32, 128],
    },
    {
        "shape_id": "zimage_rms_512x8x128",
        "model": "z-image",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 512, 8, 128],
    },
    {
        "shape_id": "wan_ti2v_ln_17850x3072",
        "model": "wan-t2v",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 17850, 3072],
    },
    {
        "shape_id": "wan_ti2v_rms_17850x3072",
        "model": "wan-t2v",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 17850, 3072],
    },
    {
        "shape_id": "wan_ti2v_rms_512x3072",
        "model": "wan-t2v",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 512, 3072],
    },
    {
        "shape_id": "wan_ti2v_rms_512x4096",
        "model": "wan-t2v",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 512, 4096],
    },
    {
        "shape_id": "wan_fused_add_17850x3072",
        "model": "wan-t2v",
        "gpu_config": "1 GPU",
        "op": "fused_add_rmsnorm",
        "input_shape": [17850, 3072],
    },
    {
        "shape_id": "hunyuan_ln_46x768",
        "model": "hunyuan-video",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 46, 768],
    },
    {
        "shape_id": "hunyuan_ln_45x3072",
        "model": "hunyuan-video",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 45, 3072],
    },
    {
        "shape_id": "hunyuan_ln_27030x3072",
        "model": "hunyuan-video",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 27030, 3072],
    },
    {
        "shape_id": "hunyuan_ln_27075x3072",
        "model": "hunyuan-video",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 27075, 3072],
    },
    {
        "shape_id": "hunyuan_rms_140x4096",
        "model": "hunyuan-video",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 140, 4096],
    },
    {
        "shape_id": "hunyuan_rms_45x24x128",
        "model": "hunyuan-video",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 45, 24, 128],
    },
    {
        "shape_id": "hunyuan_rms_27030x24x128",
        "model": "hunyuan-video",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 27030, 24, 128],
    },
    {
        "shape_id": "hunyuan_rms_27075x24x128",
        "model": "hunyuan-video",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 27075, 24, 128],
    },
    {
        "shape_id": "hunyuan_fused_add_140x4096",
        "model": "hunyuan-video",
        "gpu_config": "1 GPU",
        "op": "fused_add_rmsnorm",
        "input_shape": [140, 4096],
    },
    {
        "shape_id": "hunyuan_fused_add_27030x3072",
        "model": "hunyuan-video",
        "gpu_config": "1 GPU",
        "op": "fused_add_rmsnorm",
        "input_shape": [27030, 3072],
    },
    {
        "shape_id": "sd3_ln_4096x1536",
        "model": "sd3.5",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 4096, 1536],
    },
    {
        "shape_id": "sd3_ln_77x768",
        "model": "sd3.5",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 77, 768],
    },
    {
        "shape_id": "ltx2_ln_4096x2048",
        "model": "ltx2",
        "gpu_config": "1 GPU",
        "op": "layernorm",
        "input_shape": [1, 4096, 2048],
    },
    {
        "shape_id": "ltx2_rms_4096x2048",
        "model": "ltx2",
        "gpu_config": "1 GPU",
        "op": "rmsnorm",
        "input_shape": [1, 4096, 2048],
    },
]

GRID_SEQ_LENS = [512, 1024, 4096, 8192, 16384, 27030, 44100]
GRID_HEADS = [12, 24, 32, 48]
GRID_HEAD_DIMS = [64, 128]


def benchmark_fn(
    fn: Callable,
    reset_fn: Callable | None = None,
    warmup: int | None = None,
    iters: int | None = None,
) -> tuple[float, float, float]:
    if warmup is None:
        warmup = WARMUP_ITERS
    if iters is None:
        iters = BENCH_ITERS
    for _ in range(warmup):
        if reset_fn is not None:
            reset_fn()
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        if reset_fn is not None:
            reset_fn()
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2], max(times), min(times)


def geometric_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def effective_rows(input_shape: list[int]) -> int:
    rows = 1
    for d in input_shape[:-1]:
        rows *= d
    return rows


def _expand_kv(k: torch.Tensor, v: torch.Tensor, num_heads: int):
    nkv = k.shape[2]
    if nkv == num_heads:
        return k, v
    r = num_heads // nkv
    return k.repeat_interleave(r, dim=2), v.repeat_interleave(r, dim=2)


def build_attn_providers(
    dtype: torch.dtype,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int = 1,
    kv_seq_len: int | None = None,
) -> dict[str, Callable]:
    if kv_seq_len is None:
        kv_seq_len = seq_len
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=DEFAULT_DEVICE, dtype=dtype)
    k = torch.randn(batch_size, kv_seq_len, num_kv_heads, head_dim, device=DEFAULT_DEVICE, dtype=dtype)
    v = torch.randn(batch_size, kv_seq_len, num_kv_heads, head_dim, device=DEFAULT_DEVICE, dtype=dtype)
    scale = head_dim**-0.5
    providers: dict[str, Callable] = {}

    ke, ve = _expand_kv(k, v, num_heads)
    qs = q.permute(0, 2, 1, 3).contiguous()
    ks = ke.permute(0, 2, 1, 3).contiguous()
    vs = ve.permute(0, 2, 1, 3).contiguous()
    providers["sdpa"] = lambda: F.scaled_dot_product_attention(qs, ks, vs, is_causal=False, scale=scale)

    fa3 = _detect_fa3()
    if fa3 is not None:

        def _fa3():
            out = fa3(q, k, v, causal=False, softmax_scale=scale)
            return out[0] if isinstance(out, tuple) else out

        providers["fa3"] = _fa3

    fa2 = _detect_fa2()
    if fa2 is not None:
        providers["fa2"] = lambda: fa2(q, k, v, causal=False, softmax_scale=scale)

    sage = _detect_sage()
    if sage is not None:
        providers["sage_attn"] = lambda: sage(
            q,
            k,
            v,
            tensor_layout="NHD",
            is_causal=False,
            sm_scale=scale,
        )

    return providers


def build_rmsnorm_providers(
    dtype: torch.dtype,
    batch_size: int,
    hidden_size: int,
) -> dict[str, Callable]:
    x = torch.randn(batch_size, hidden_size, device=DEFAULT_DEVICE, dtype=dtype)
    w = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=dtype)
    providers: dict[str, Callable] = {}

    providers["pytorch"] = lambda: F.rms_norm(x, (hidden_size,), w, EPS)

    fi = _try_import_flashinfer_norm()
    if fi is not None:
        fi_out = torch.empty_like(x)
        providers["flashinfer"] = lambda: fi.rmsnorm(x, w, eps=EPS, out=fi_out)

    fg = _try_import_flaggems()
    if fg is not None:
        fg_rms, _, _ = fg
        providers["flaggems"] = lambda: fg_rms(x, (hidden_size,), w, EPS)

    qk = _try_import_quack()
    if qk is not None:
        qk_rms, _ = qk
        providers["quack"] = lambda: qk_rms(x, w, eps=EPS)

    triton_norm = _try_import_triton_norm()
    if triton_norm is not None:
        t_out = torch.empty_like(x)
        providers["triton_norm_infer"] = lambda: triton_norm(x, w, bias=None, eps=EPS, is_rms_norm=True, out=t_out)

    return providers


def build_layernorm_providers(
    dtype: torch.dtype,
    batch_size: int,
    hidden_size: int,
) -> dict[str, Callable]:
    x = torch.randn(batch_size, hidden_size, device=DEFAULT_DEVICE, dtype=dtype)
    w = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=dtype)
    b = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=dtype)
    providers: dict[str, Callable] = {}

    providers["pytorch"] = lambda: F.layer_norm(x, (hidden_size,), w, b, EPS)

    fi = _try_import_flashinfer_norm()
    if fi is not None:
        fi_w = w.to(torch.float32)
        fi_b = b.to(torch.float32)
        providers["flashinfer"] = lambda: fi.layernorm(x, fi_w, fi_b, EPS)

    fg = _try_import_flaggems()
    if fg is not None:
        _, fg_ln, _ = fg
        providers["flaggems"] = lambda: fg_ln(x, (hidden_size,), w, b)[0]

    qk = _try_import_quack()
    if qk is not None:
        _, qk_ln = qk
        qk_w = w.to(torch.float32)
        qk_b = b.to(torch.float32)
        providers["quack"] = lambda: qk_ln(x, qk_w, qk_b, EPS)

    triton_norm = _try_import_triton_norm()
    if triton_norm is not None:
        t_out = torch.empty_like(x)
        providers["triton_norm_infer"] = lambda: triton_norm(x, w, b, eps=EPS, is_rms_norm=False, out=t_out)

    return providers


def build_fused_add_rmsnorm_providers(
    dtype: torch.dtype,
    batch_size: int,
    hidden_size: int,
) -> dict[str, tuple[Callable, Callable]]:
    base_x = torch.randn(batch_size, hidden_size, device=DEFAULT_DEVICE, dtype=dtype)
    base_res = torch.randn_like(base_x)
    w = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=dtype)

    x = base_x.clone()
    res = base_res.clone()

    def reset():
        x.copy_(base_x)
        res.copy_(base_res)

    providers: dict[str, tuple[Callable, Callable]] = {}

    providers["pytorch"] = (
        lambda: F.rms_norm(x + res, (hidden_size,), w, EPS),
        reset,
    )

    fi = _try_import_flashinfer_norm()
    if fi is not None:
        providers["flashinfer"] = (
            lambda: fi.fused_add_rmsnorm(x, res, w, eps=EPS),
            reset,
        )

    fg = _try_import_flaggems()
    if fg is not None:
        _, _, fg_fused = fg
        providers["flaggems"] = (
            lambda: fg_fused(x, res, (hidden_size,), w, EPS),
            reset,
        )

    qk = _try_import_quack()
    if qk is not None:
        qk_rms, _ = qk
        providers["quack"] = (
            lambda: qk_rms(x, w, residual=res, eps=EPS),
            reset,
        )

    return providers


def _safe_benchmark(
    op: str,
    provider: str,
    fn: Callable,
    reset_fn: Callable | None = None,
) -> dict[str, Any] | None:
    try:
        median, mx, mn = benchmark_fn(fn, reset_fn)
        return {"op": op, "provider": provider, "median_us": median, "max_us": mx, "min_us": mn, "status": "ok"}
    except Exception as e:
        print(f"    {provider}: FAILED ({e})")
        return None


def run_attention_shapes(
    shapes: list[dict[str, Any]],
    dtypes: list[torch.dtype],
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        sid = shape["shape_id"]
        for dtype in dtypes:
            ds = "bf16" if dtype == torch.bfloat16 else "fp16"
            print(f"  [attention] {sid} ({ds})")
            try:
                providers = build_attn_providers(
                    dtype,
                    shape["seq_len"],
                    shape["num_heads"],
                    shape["num_kv_heads"],
                    shape["head_dim"],
                    batch_size,
                    kv_seq_len=shape.get("kv_seq_len"),
                )
            except Exception as e:
                print(f"    SKIP: {e}")
                continue

            latencies: dict[str, float] = {}
            for pname, fn in providers.items():
                r = _safe_benchmark("attention", pname, fn)
                if r:
                    latencies[pname] = r["median_us"]

            if not latencies:
                continue
            winner = min(latencies, key=latencies.get)
            all_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(latencies.items(), key=lambda x: x[1]))
            kv_sl = shape.get("kv_seq_len", shape["seq_len"])
            rows.append(
                {
                    "op": "attention",
                    "shape_id": sid,
                    "model": shape["model"],
                    "gpu_config": shape.get("gpu_config", "1 GPU"),
                    "dtype": ds,
                    "input_shape": (
                        f"q={shape['seq_len']}, kv={kv_sl}, "
                        f"h={shape['num_heads']}, kvh={shape['num_kv_heads']}, "
                        f"d={shape['head_dim']}"
                    ),
                    "winner": winner,
                    "winner_latency_us": round(latencies[winner], 2),
                    "all_providers": all_str,
                }
            )
            del providers
            gc.collect()
            torch.cuda.empty_cache()
    return rows


def run_norm_shapes(
    shapes: list[dict[str, Any]],
    dtypes: list[torch.dtype],
    ops_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        op = shape["op"]
        if ops_filter and op not in ops_filter:
            continue
        sid = shape["shape_id"]
        input_shape = shape["input_shape"]
        bs = effective_rows(input_shape)
        hs = input_shape[-1]

        for dtype in dtypes:
            ds = "bf16" if dtype == torch.bfloat16 else "fp16"
            print(f"  [{op}] {sid} ({ds})")

            latencies: dict[str, float] = {}

            if op == "rmsnorm":
                providers = build_rmsnorm_providers(dtype, bs, hs)
                for pname, fn in providers.items():
                    r = _safe_benchmark(op, pname, fn)
                    if r:
                        latencies[pname] = r["median_us"]

            elif op == "layernorm":
                providers = build_layernorm_providers(dtype, bs, hs)
                for pname, fn in providers.items():
                    r = _safe_benchmark(op, pname, fn)
                    if r:
                        latencies[pname] = r["median_us"]

            elif op == "fused_add_rmsnorm":
                fused_providers = build_fused_add_rmsnorm_providers(dtype, bs, hs)
                for pname, (fn, reset) in fused_providers.items():
                    r = _safe_benchmark(op, pname, fn, reset)
                    if r:
                        latencies[pname] = r["median_us"]
            else:
                continue

            if not latencies:
                continue
            winner = min(latencies, key=latencies.get)
            all_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(latencies.items(), key=lambda x: x[1]))
            rows.append(
                {
                    "op": op,
                    "shape_id": sid,
                    "model": shape["model"],
                    "gpu_config": shape.get("gpu_config", "1 GPU"),
                    "dtype": ds,
                    "input_shape": str(input_shape),
                    "winner": winner,
                    "winner_latency_us": round(latencies[winner], 2),
                    "all_providers": all_str,
                }
            )
            gc.collect()
            torch.cuda.empty_cache()
    return rows


def run_attention_grid(dtypes: list[torch.dtype], batch_size: int = 1) -> list[dict[str, Any]]:
    shapes = []
    for sl in GRID_SEQ_LENS:
        for nh in GRID_HEADS:
            for hd in GRID_HEAD_DIMS:
                shapes.append(
                    {
                        "shape_id": f"grid_{sl}x{nh}x{hd}",
                        "model": "grid",
                        "seq_len": sl,
                        "num_heads": nh,
                        "num_kv_heads": nh,
                        "head_dim": hd,
                        "gpu_config": "1 GPU",
                    }
                )
    return run_attention_shapes(shapes, dtypes, batch_size)


def check_attention_accuracy(shapes: list[dict[str, Any]], dtypes: list[torch.dtype]):
    print("=" * 60)
    print("Attention accuracy check (max abs diff vs SDPA reference)")
    print("=" * 60)
    for shape in shapes[:5]:
        for dtype in dtypes:
            ds = "bf16" if dtype == torch.bfloat16 else "fp16"
            sl, nh, nkv, hd = shape["seq_len"], shape["num_heads"], shape["num_kv_heads"], shape["head_dim"]
            kv_sl = shape.get("kv_seq_len", sl)
            q = torch.randn(1, sl, nh, hd, device=DEFAULT_DEVICE, dtype=dtype)
            k = torch.randn(1, kv_sl, nkv, hd, device=DEFAULT_DEVICE, dtype=dtype)
            v = torch.randn(1, kv_sl, nkv, hd, device=DEFAULT_DEVICE, dtype=dtype)
            scale = hd**-0.5

            ke, ve = _expand_kv(k, v, nh)
            ref = F.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3),
                ke.permute(0, 2, 1, 3),
                ve.permute(0, 2, 1, 3),
                is_causal=False,
                scale=scale,
            ).permute(0, 2, 1, 3)
            torch.cuda.synchronize()

            diffs = []
            for name, detect in [("fa3", _detect_fa3), ("fa2", _detect_fa2), ("sage_attn", _detect_sage)]:
                fn = detect()
                if fn is None:
                    continue
                try:
                    if name == "sage_attn":
                        out = fn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=scale)
                    else:
                        out = fn(q, k, v, causal=False, softmax_scale=scale)
                        out = out[0] if isinstance(out, tuple) else out
                    d = (out.float() - ref.float()).abs().max().item()
                    diffs.append(f"{name}={d:.6f}")
                except Exception as e:
                    diffs.append(f"{name}=SKIP({e.__class__.__name__})")

            if diffs:
                print(f"  {shape['shape_id']} ({ds}): {', '.join(diffs)}")
    print()


def format_markdown(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()

    for op_name in sorted(set(r["op"] for r in rows)):
        op_rows = [r for r in rows if r["op"] == op_name]
        for ds in sorted(set(r["dtype"] for r in op_rows)):
            dr = [r for r in op_rows if r["dtype"] == ds]
            buf.write(f"### {op_name} ({ds})\n\n")

            wins = Counter(r["winner"] for r in dr)
            buf.write("Winner | Win Count\n-------|----------\n")
            for name, count in wins.most_common():
                buf.write(f"{name} | {count}\n")
            buf.write("\n")

            buf.write("Shape ID | Model | GPU Config | Input Shape | Winner | Winner Latency (us) | All Providers\n")
            buf.write("---------|-------|------------|-------------|--------|---------------------|---------------\n")
            for r in dr:
                buf.write(
                    f"{r['shape_id']} | {r['model']} | {r['gpu_config']} | "
                    f"{r['input_shape']} | {r['winner']} | {r['winner_latency_us']} | "
                    f"{r['all_providers']}\n"
                )
            buf.write("\n")
    return buf.getvalue()


def write_csv(rows: list[dict[str, Any]], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def write_markdown(rows: list[dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_markdown(rows), encoding="utf-8")
    print(f"Wrote {path}")


ALL_OPS = ["attention", "layernorm", "rmsnorm", "fused_add_rmsnorm"]


def main():
    parser = argparse.ArgumentParser(description="Unified diffusion ops benchmark")
    parser.add_argument("--ops", nargs="+", default=ALL_OPS, choices=ALL_OPS)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--dtypes", nargs="+", default=["bf16", "fp16"], choices=["bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    _update_bench_config(args.warmup, args.iters)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    dtypes = [dtype_map[d] for d in args.dtypes]

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Ops: {args.ops}")
    print(f"Dtypes: {args.dtypes}")

    print("\nAvailable attention providers: ", end="")
    attn_avail = ["sdpa"]
    if _detect_fa3():
        attn_avail.append("fa3")
    if _detect_fa2():
        attn_avail.append("fa2")
    if _detect_sage():
        attn_avail.append("sage_attn")
    print(", ".join(attn_avail))

    print("Available norm providers: pytorch", end="")
    if _try_import_flashinfer_norm():
        print(", flashinfer", end="")
    if _try_import_flaggems():
        print(", flaggems", end="")
    if _try_import_quack():
        print(", quack", end="")
    if _try_import_triton_norm():
        print(", triton_norm_infer", end="")
    print("\n")

    if args.accuracy and "attention" in args.ops:
        attn_shapes = DIFFUSION_ATTN_SHAPES
        if args.models:
            attn_shapes = [s for s in attn_shapes if s["model"] in args.models]
        check_attention_accuracy(attn_shapes, dtypes)

    all_rows: list[dict[str, Any]] = []

    if "attention" in args.ops:
        print("=" * 60)
        print("Attention benchmark")
        print("=" * 60)
        if args.grid:
            all_rows += run_attention_grid(dtypes, args.batch_size)
        else:
            attn_shapes = DIFFUSION_ATTN_SHAPES
            if args.models:
                attn_shapes = [s for s in attn_shapes if s["model"] in args.models]
            all_rows += run_attention_shapes(attn_shapes, dtypes, args.batch_size)

    norm_ops = [o for o in args.ops if o != "attention"]
    if norm_ops:
        print("=" * 60)
        print("Norm benchmark")
        print("=" * 60)
        norm_shapes = DIFFUSION_NORM_SHAPES
        if args.models:
            norm_shapes = [s for s in norm_shapes if s["model"] in args.models]
        all_rows += run_norm_shapes(norm_shapes, dtypes, norm_ops)

    if not all_rows:
        print("No results collected.")
        return

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(format_markdown(all_rows))

    if args.output_dir:
        out = Path(args.output_dir)
        write_csv(all_rows, out / "diffusion_ops.csv")
        write_markdown(all_rows, out / "diffusion_ops_summary.md")


if __name__ == "__main__":
    main()
