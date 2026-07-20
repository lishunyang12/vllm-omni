# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""trtllm diffusion attention backend (FlashInfer trtllm-gen FMHA) with Skip-Softmax.

BF16 dense attention on Blackwell (SM100+), optional Skip-Softmax on top. GQA is
native; no SDPA fallback (unsupported head dims raise from the kernel). FP8-SAGE
is a planned follow-up.
"""

import math
from dataclasses import dataclass, replace

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.utils.ragged import varlen_cu_seqlens

logger = init_logger(__name__)


@dataclass(frozen=True)
class SkipSoftmaxConfig:
    """Skip-Softmax operating point for the trtllm backend (pure policy, no I/O).

    Factor sources (priority): threshold (calibration-free, all layers) > ignore_threshold
    (manual threshold on the calibration's skippable layers only -- ignore list respected)
    > calibrated curve (factor = a*exp(b*target_sparsity)). disabled_until_timestep keeps
    early/noisy steps dense: skip is active only when timestep t (1->0) drops to
    t <= disabled_until_timestep.
    """

    threshold: float | None = None
    ignore_threshold: float | None = None
    target_sparsity: float | None = None
    disabled_until_timestep: float = 0.0
    a: float | None = None
    b: float | None = None

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: dict | None) -> "SkipSoftmaxConfig":
        # a, b are NOT read here: for the calibrated path they are per-layer and stamped
        # post-build by set_layer_calibration(). backend_kwargs carries only the global
        # operating point (threshold / target_sparsity / timestep guard).
        bk = backend_kwargs or {}
        return cls(
            threshold=bk.get("skip_softmax_threshold"),
            ignore_threshold=bk.get("skip_threshold_calibrated"),
            target_sparsity=bk.get("target_sparsity"),
            disabled_until_timestep=float(bk.get("disabled_until_timestep", 0.0)),
        )

    @property
    def enabled(self) -> bool:
        # ignore_threshold and the calibrated curve both require a stamped layer (a is set);
        # ignored/cross-attn layers (a is None) stay dense -- that's how the ignore list is
        # honored for the manual-threshold path.
        return (
            self.threshold is not None
            or (self.ignore_threshold is not None and self.a is not None)
            or (self.target_sparsity is not None and self.a is not None and self.b is not None)
        )

    @property
    def gated(self) -> bool:
        return self.disabled_until_timestep > 0.0

    def resolve_factor(self, seqlen: int, timestep: float | None) -> float | None:
        """Effective factor for this step (None = dense). timestep is the normalized
        denoise timestep (None -> guard cannot fire, skip applies)."""
        if self.threshold is not None:
            factor = self.threshold * seqlen
        elif self.ignore_threshold is not None and self.a is not None:
            # Manual threshold on the calibration's skippable layers (ignore list respected:
            # ignored layers are never stamped, so a is None -> they stay dense). Fixes the
            # Triton->trtllm-gen under-skip by setting the effective threshold directly.
            factor = self.ignore_threshold * seqlen
        elif self.target_sparsity is not None and self.a is not None and self.b is not None:
            factor = self.a * math.exp(self.b * self.target_sparsity)
        else:
            return None
        if self.gated and timestep is not None and timestep > self.disabled_until_timestep:
            return None  # still in the dense guard window
        return factor

try:
    from flashinfer.prefill import trtllm_ragged_attention_deepseek

    HAS_FLASHINFER = True
except Exception as e:  # pragma: no cover - import guard
    HAS_FLASHINFER = False
    logger.warning(
        "FlashInfer is unavailable; TRTLLM backend will not work. Reason: %s",
        e,
    )

# FMHA kernel scratch buffer: zero-initialized (kernel semaphores start at 0),
# cached per-device. Size from vLLM's workspace env (default 394 MB).
def _workspace_bytes() -> int:
    import vllm.envs as envs

    return getattr(envs, "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", 394 * 1024 * 1024)


def _to_fp8_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize to e4m3 with one per-tensor amax scale (dequant: real = fp8 * scale)."""
    amax = x.abs().amax().clamp(min=1e-4)
    scale = amax / 448.0  # e4m3 max magnitude
    xq = (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return xq, float(scale)


class TrtllmGenBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_name() -> str:
        return "TRTLLM"

    @staticmethod
    def get_impl_cls() -> type["TrtllmGenImpl"]:
        return TrtllmGenImpl


class TrtllmGenImpl(AttentionImpl):
    # Cached device scratch buffer, lazily allocated and shared across layers.
    _workspace: torch.Tensor | None = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        self.skip = SkipSoftmaxConfig.from_backend_kwargs(backend_kwargs)
        self._warned_missing_timestep = False
        # FP8 attention (SAGE family): quantize Q/K/V to e4m3 and let the trtllm-gen FMHA
        # run in FP8. Per-tensor scales (the flashinfer-validated path); the kernel folds
        # them via bmm1/bmm2 scales. `extra.sage` truthy -> on. Composable with Skip-Softmax.
        self.fp8_attn = bool((backend_kwargs or {}).get("sage"))
        # For the calibrated path, the per-layer curve (a, b) is NOT known here: which
        # expert a Wan layer belongs to, and whether it is on the ignore list, are only
        # decidable from the full module name, which this layer does not carry at build
        # time. So a, b are stamped post-build by sparse_attention.apply_to_pipeline via
        # set_layer_calibration(). Until stamped, a calibrated layer stays dense (safe:
        # no wrong-threshold guessing). The calibration-free threshold path needs no stamp.

    def set_layer_calibration(self, a: float, b: float) -> None:
        """Install this layer's calibrated skip curve (post-build applier, by module name).

        Called only for skippable layers -- ignored/cross-attention layers are never
        stamped and therefore stay dense. Idempotent last-write-wins.
        """
        self.skip = replace(self.skip, a=a, b=b)

    def _resolve_skip_factor(self, seqlen: int) -> float | None:
        # Dense unless skip is enabled (threshold set, or target_sparsity + stamped a,b).
        # Then read the denoise timestep off the ForwardContext and delegate to the policy.
        if not self.skip.enabled:
            return None

        # Only a timestep guard needs the ForwardContext; ungated skip must not require one.
        timestep = None
        if self.skip.gated:
            from vllm_omni.diffusion.forward_context import get_forward_context

            timestep = getattr(get_forward_context(), "denoise_timestep", None)
            if timestep is None and not self._warned_missing_timestep:
                logger.warning(
                    "TRTLLM skip: disabled_until_timestep=%s set but no denoise_timestep on "
                    "ForwardContext; timestep guard inactive (skip applied every step).",
                    self.skip.disabled_until_timestep,
                )
                self._warned_missing_timestep = True
        return self.skip.resolve_factor(seqlen, timestep)

    @classmethod
    def _get_workspace(cls, device: torch.device) -> torch.Tensor:
        nbytes = _workspace_bytes()
        ws = cls._workspace
        if ws is None or ws.device != device or ws.numel() < nbytes:
            # Zero-initialized: the kernel's semaphore/counter region must start at 0.
            ws = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            cls._workspace = ws
        return ws

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if not HAS_FLASHINFER:
            raise ImportError(
                "TRTLLM backend requires flashinfer. Install it or select "
                "another backend via --diffusion-attention-backend."
            )

        # Input layout is (B, S, H, D).
        batch, q_len, num_q_heads, head_dim = query.shape
        kv_len, num_kv_heads = key.shape[1], key.shape[2]

        # No pre-check: unsupported head dims raise from the kernel's own assert; GQA is native.
        device = query.device
        # Ragged pack: (B, S, H, D) -> (B*S, H, D) with uniform per-sample lengths.
        q = query.reshape(batch * q_len, num_q_heads, head_dim).contiguous()
        k = key.reshape(batch * kv_len, num_kv_heads, head_dim).contiguous()
        v = value.reshape(batch * kv_len, num_kv_heads, head_dim).contiguous()

        seq_lens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)
        cu_seq_lens_q = varlen_cu_seqlens(batch, q_len, device)
        cu_seq_lens_kv = varlen_cu_seqlens(batch, kv_len, device)
        workspace = self._get_workspace(device)

        # BF16 dense (softmax_scale rides bmm1); Skip-Softmax layers on below.
        bmm1_scale = self.softmax_scale
        bmm2_scale = 1.0

        # FP8 attention: quantize Q/K/V to e4m3 (per-tensor amax) and fold the scales into
        # bmm1/bmm2. Q@K^T runs in FP8 -> real scores = sq*sk*(Qf@Kf); softmax rides bmm1;
        # P@V -> real = sv*(P@Vf). Kept simple/per-tensor (flashinfer-validated); the
        # per-block SAGE layout is a follow-up.
        if self.fp8_attn:
            q, sq = _to_fp8_per_tensor(q)
            k, sk = _to_fp8_per_tensor(k)
            v, sv = _to_fp8_per_tensor(v)
            bmm1_scale = sq * sk * self.softmax_scale
            bmm2_scale = sv

        _skip_factor = self._resolve_skip_factor(kv_len)

        # Keyword args guard against positional drift across flashinfer versions.
        out = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=workspace,
            seq_lens=seq_lens,
            max_q_len=q_len,
            max_kv_len=kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            o_sf_scale=-1.0,  # no NVFP4 output scale (bf16 out)
            batch_size=batch,
            window_left=-1,
            cum_seq_lens_q=cu_seq_lens_q,
            cum_seq_lens_kv=cu_seq_lens_kv,
            enable_pdl=False,
            is_causal=self.causal,  # DiT is bidirectional -> False
            return_lse=False,
            skip_softmax_threshold_scale_factor=_skip_factor,
        )
        return out.reshape(batch, q_len, num_q_heads, head_dim)
