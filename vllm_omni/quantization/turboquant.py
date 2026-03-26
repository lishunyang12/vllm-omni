# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant: online vector quantization for KV cache compression.

Implements the TurboQuant algorithm (https://arxiv.org/abs/2504.19874)
for sub-4-bit KV cache quantization with near-optimal distortion.

Two-stage approach:
  1. Random rotation + Lloyd-Max scalar quantization (b-1 bits)
  2. QJL 1-bit transform on residual for unbiased inner products

Adapted from:
  - https://github.com/tonbistudio/turboquant-pytorch
  - https://github.com/TheTom/turboquant_plus
  - https://github.com/Blaizzy/mlx-vlm/pull/858
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Precomputed Lloyd-Max codebooks (Gaussian approximation N(0, 1/d)).
#
# Normalized centroids: multiply by 1/sqrt(d) to get actual values.
# Solved via Lloyd-Max algorithm on the Gaussian PDF.
# ---------------------------------------------------------------------------

CODEBOOKS_NORMALIZED: dict[int, list[float]] = {
    1: [-0.7979, 0.7979],  # ±sqrt(2/pi)
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [
        -2.1520, -1.3440, -0.7560, -0.2451,
        0.2451, 0.7560, 1.3440, 2.1520,
    ],
    4: [
        -2.7326, -2.0690, -1.6180, -1.2562,
        -0.9424, -0.6568, -0.3880, -0.1284,
        0.1284, 0.3880, 0.6568, 0.9424,
        1.2562, 1.6180, 2.0690, 2.7326,
    ],
}

EXPECTED_MSE_NORMALIZED: dict[int, float] = {
    1: 0.3634,
    2: 0.1175,
    3: 0.03454,
    4: 0.009497,
}


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache quantization.

    Supports fractional bit-widths (2.5, 3.5) via mixed-precision
    channel splitting:
      - 2.5-bit: 50% channels at 3-bit + 50% at 2-bit
      - 3.5-bit: 25% channels at 4-bit + 75% at 3-bit
    """

    bit_width: float = 3
    use_qjl: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        valid = {1, 2, 2.5, 3, 3.5, 4}
        if self.bit_width not in valid:
            raise ValueError(
                f"bit_width must be one of {sorted(valid)}, got {self.bit_width}"
            )

    @property
    def is_fractional(self) -> bool:
        return self.bit_width != int(self.bit_width)

    @property
    def channel_split(self) -> tuple[tuple[int, float], tuple[int, float]] | None:
        """Return ((hi_bits, hi_ratio), (lo_bits, lo_ratio)) for fractional.

        E.g. 3.5 -> ((4, 0.25), (3, 0.75)) meaning 25% at 4-bit, 75% at 3-bit.
        """
        if not self.is_fractional:
            return None
        if self.bit_width == 2.5:
            return ((3, 0.5), (2, 0.5))
        if self.bit_width == 3.5:
            return ((4, 0.25), (3, 0.75))
        return None


# ---------------------------------------------------------------------------
# Random rotation via QR decomposition (Haar-distributed orthogonal matrix).
#
# All three reference implementations use this approach, NOT Hadamard.
# O(d^2) storage and compute per vector, but d=128 is small enough.
# ---------------------------------------------------------------------------


def _generate_rotation_matrix(
    d: int, seed: int, device: torch.device
) -> Tensor:
    """Generate a Haar-distributed random rotation matrix via QR."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    # Fix sign ambiguity in QR to get proper rotation
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def _get_codebook(bit_width: int, dim: int, device: torch.device) -> Tensor:
    """Return Lloyd-Max codebook scaled to dimension d."""
    normalized = CODEBOOKS_NORMALIZED[bit_width]
    scale = 1.0 / math.sqrt(dim)
    return torch.tensor(
        [c * scale for c in normalized],
        device=device,
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Core TurboQuant state — one per attention layer
# ---------------------------------------------------------------------------


class TurboQuantState:
    """Stores per-layer random state for TurboQuant.

    Create one per attention layer. The rotation matrix is generated
    deterministically from seed + layer_idx.

    For fractional bit-widths (2.5, 3.5), channels are split into
    two groups quantized at different bit-widths.
    """

    def __init__(
        self,
        config: TurboQuantConfig,
        head_size: int,
        layer_idx: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.head_size = head_size
        self.device = device

        # Deterministic rotation matrix per layer (d x d)
        self.Pi = _generate_rotation_matrix(
            head_size, config.seed + layer_idx, device
        )
        self.PiT = self.Pi.T.contiguous()

        # QJL projection matrix
        if config.use_qjl:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(config.seed + layer_idx + 10000)
            self.S = torch.randn(
                head_size, head_size, generator=gen
            ).to(device)
        else:
            self.S = None

        # Setup codebooks for integer or fractional bit-widths
        if config.is_fractional:
            split = config.channel_split
            (hi_bits, hi_ratio), (lo_bits, lo_ratio) = split
            self.hi_bits = hi_bits
            self.lo_bits = lo_bits
            self.hi_channels = int(head_size * hi_ratio)
            self.lo_channels = head_size - self.hi_channels
            self.codebook_hi = _get_codebook(hi_bits, head_size, device)
            self.codebook_lo = _get_codebook(lo_bits, head_size, device)
            self.boundaries_hi = (self.codebook_hi[:-1] + self.codebook_hi[1:]) / 2.0
            self.boundaries_lo = (self.codebook_lo[:-1] + self.codebook_lo[1:]) / 2.0
            self.mse_bits = None  # not used for fractional
        else:
            mse_bits = int(config.bit_width) - 1 if config.use_qjl else int(config.bit_width)
            mse_bits = max(mse_bits, 1)
            self.mse_bits = mse_bits
            self.codebook = _get_codebook(mse_bits, head_size, device)
            self.boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2.0
            self.hi_bits = None

    @torch.no_grad()
    def quantize(self, x: Tensor) -> dict[str, Tensor]:
        """Quantize KV head vectors."""
        orig_shape = x.shape
        flat = x.reshape(-1, self.head_size).float()

        # Extract norms, normalize to unit sphere
        norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_norm = flat / (norms + 1e-8)

        # Rotate
        rotated = flat_norm @ self.PiT

        if self.config.is_fractional:
            return self._quantize_fractional(rotated, norms, orig_shape, x.device)
        else:
            return self._quantize_integer(rotated, norms, orig_shape, x.device)

    def _quantize_integer(self, rotated: Tensor, norms: Tensor,
                          orig_shape: tuple, device: torch.device) -> dict:
        indices = torch.bucketize(rotated.contiguous(), self.boundaries).to(torch.uint8)
        packed = pack_indices(indices.reshape(-1), self.mse_bits)

        result: dict[str, Tensor] = {
            "packed": packed,
            "n_elements": torch.tensor(indices.numel(), device=device),
            "orig_shape": torch.tensor(orig_shape, device=device),
            "norms": norms.squeeze(-1).to(torch.float16).reshape(orig_shape[:-1]),
        }

        if self.config.use_qjl and self.S is not None:
            reconstructed = self.codebook[indices.long()]
            residual_rotated = rotated - reconstructed
            residual = residual_rotated @ self.Pi
            r_norm = torch.norm(residual, dim=-1)
            projected = residual @ self.S.T
            signs = (projected >= 0).to(torch.int8) * 2 - 1
            result["qjl_signs"] = signs.reshape(
                orig_shape[:-1] + (self.head_size,)
            )
            result["qjl_norms"] = r_norm.to(torch.float16).reshape(
                orig_shape[:-1]
            )
        return result

    def _quantize_fractional(self, rotated: Tensor, norms: Tensor,
                             orig_shape: tuple, device: torch.device) -> dict:
        """Split channels and quantize each group at different bit-widths."""
        hi_part = rotated[:, :self.hi_channels]
        lo_part = rotated[:, self.hi_channels:]

        hi_idx = torch.bucketize(hi_part.contiguous(), self.boundaries_hi).to(torch.uint8)
        lo_idx = torch.bucketize(lo_part.contiguous(), self.boundaries_lo).to(torch.uint8)

        hi_packed = pack_indices(hi_idx.reshape(-1), self.hi_bits)
        lo_packed = pack_indices(lo_idx.reshape(-1), self.lo_bits)

        return {
            "hi_packed": hi_packed,
            "lo_packed": lo_packed,
            "hi_n": torch.tensor(hi_idx.numel(), device=device),
            "lo_n": torch.tensor(lo_idx.numel(), device=device),
            "orig_shape": torch.tensor(orig_shape, device=device),
            "norms": norms.squeeze(-1).to(torch.float16).reshape(orig_shape[:-1]),
        }

    @torch.no_grad()
    def dequantize(self, state: dict[str, Tensor]) -> Tensor:
        """Dequantize back to full-precision vectors."""
        orig_shape = tuple(state["orig_shape"].tolist())
        norms = state["norms"].float()
        flat_norms = norms.reshape(-1)

        if self.config.is_fractional:
            x_hat = self._dequantize_fractional(state, flat_norms)
        else:
            x_hat = self._dequantize_integer(state, flat_norms)

        return x_hat.to(torch.bfloat16).reshape(orig_shape)

    def _dequantize_integer(self, state: dict, flat_norms: Tensor) -> Tensor:
        packed = state["packed"]
        n_elements = state["n_elements"].item()

        indices = unpack_indices(packed, self.mse_bits, n_elements)
        flat_indices = indices.reshape(-1, self.head_size).long()

        reconstructed = self.codebook[flat_indices]
        x_hat = reconstructed @ self.Pi  # unrotate

        if (self.config.use_qjl and self.S is not None
                and "qjl_signs" in state):
            signs = state["qjl_signs"].reshape(-1, self.head_size).float()
            r_norms = state["qjl_norms"].float().reshape(-1)
            scale = math.sqrt(math.pi / 2.0) / self.head_size
            qjl_recon = scale * r_norms.unsqueeze(-1) * (signs @ self.S)
            x_hat = x_hat + qjl_recon

        x_hat = x_hat * flat_norms.unsqueeze(-1)
        return x_hat

    def _dequantize_fractional(self, state: dict, flat_norms: Tensor) -> Tensor:
        hi_indices = unpack_indices(
            state["hi_packed"], self.hi_bits, state["hi_n"].item()
        ).reshape(-1, self.hi_channels).long()
        lo_indices = unpack_indices(
            state["lo_packed"], self.lo_bits, state["lo_n"].item()
        ).reshape(-1, self.lo_channels).long()

        hi_recon = self.codebook_hi[hi_indices]
        lo_recon = self.codebook_lo[lo_indices]

        # Concatenate channels and unrotate
        rotated_recon = torch.cat([hi_recon, lo_recon], dim=-1)
        x_hat = rotated_recon @ self.Pi
        x_hat = x_hat * flat_norms.unsqueeze(-1)
        return x_hat


# ---------------------------------------------------------------------------
# Asymmetric attention: compute scores from compressed K, decompress V only.
# Adapted from turboquant-pytorch V2 compressor.
# ---------------------------------------------------------------------------


@torch.no_grad()
def turboquant_asymmetric_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    k_state: TurboQuantState,
    v_state: TurboQuantState,
    scale: float,
) -> Tensor:
    """Compute attention with TurboQuant-compressed K/V.

    1. Compress K (store k_mse fp16 + QJL signs for unbiased scores)
    2. Compress V (store indices + norms, much smaller than bf16)
    3. Delete original K, V
    4. Compute attention scores from compressed K (asymmetric)
    5. Decompress V and apply attention weights

    Args:
        query: (B, S_q, H, D) — queries
        key: (B, S_kv, H_kv, D) — keys
        value: (B, S_kv, H_kv, D) — values
        k_state: TurboQuantState for keys (should have use_qjl=True)
        v_state: TurboQuantState for values (MSE-only)
        scale: attention scale factor (1/sqrt(d))

    Returns:
        (B, S_q, H, D) attention output
    """
    B, S_q, H, D = query.shape
    _, S_kv, H_kv, _ = key.shape
    n_rep = H // H_kv  # GQA repeat factor

    # --- Compress K: store dequantized MSE + QJL metadata ---
    k_comp = k_state.quantize(key)
    k_mse = k_state.dequantize(k_comp).float()  # (B, S_kv, H_kv, D)

    # --- Compress V: only indices + norms (smaller than bf16) ---
    v_comp = v_state.quantize(value)

    # Free original K, V
    del key, value

    # --- Asymmetric attention scores ---
    # Expand KV heads for GQA
    if n_rep > 1:
        k_mse = k_mse.unsqueeze(3).expand(B, S_kv, H_kv, n_rep, D)
        k_mse = k_mse.reshape(B, S_kv, H, D)

    # score = Q @ K_mse^T  (standard part)
    q_f = query.float()  # (B, S_q, H, D)
    scores = torch.einsum("bqhd,bkhd->bhqk", q_f, k_mse) * scale

    # QJL correction for unbiased inner product
    if "qjl_signs" in k_comp and k_state.S is not None:
        signs = k_comp["qjl_signs"].float()  # (B, S_kv, H_kv, D)
        r_norms = k_comp["qjl_norms"].float()  # (B, S_kv, H_kv)
        m = D
        correction_scale = math.sqrt(math.pi / 2.0) / m * scale

        # Project queries through QJL matrix
        q_flat = q_f.reshape(-1, D)
        q_proj = (q_flat @ k_state.S.T).reshape(B, S_q, H, D)

        if n_rep > 1:
            signs = signs.unsqueeze(3).expand(B, S_kv, H_kv, n_rep, D)
            signs = signs.reshape(B, S_kv, H, D)
            r_norms = r_norms.unsqueeze(3).expand(B, S_kv, H_kv, n_rep)
            r_norms = r_norms.reshape(B, S_kv, H)

        # qjl_ip[b,h,q,k] = sum_d(q_proj[b,q,h,d] * signs[b,k,h,d])
        qjl_ip = torch.einsum("bqhd,bkhd->bhqk", q_proj, signs)
        scores = scores + correction_scale * qjl_ip * r_norms.permute(0, 2, 1).unsqueeze(2)

    del k_mse, k_comp

    # --- Softmax ---
    attn_weights = torch.softmax(scores, dim=-1).to(torch.bfloat16)
    del scores

    # --- Decompress V and apply weights ---
    v_deq = v_state.dequantize(v_comp)  # (B, S_kv, H_kv, D)
    del v_comp

    if n_rep > 1:
        v_deq = v_deq.unsqueeze(3).expand(B, S_kv, H_kv, n_rep, D)
        v_deq = v_deq.reshape(B, S_kv, H, D)

    # output[b,q,h,d] = sum_k(attn_weights[b,h,q,k] * v_deq[b,k,h,d])
    output = torch.einsum("bhqk,bkhd->bqhd", attn_weights.float(), v_deq.float())

    return output.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Bit packing: store sub-byte indices in packed uint8 tensors
# ---------------------------------------------------------------------------


def pack_indices(indices: Tensor, bits: int) -> Tensor:
    """Pack b-bit indices into uint8 tensor.

    For 4-bit: 2 values per byte.  For 2-bit: 4 values per byte.
    For 3-bit: pack into uint32 then view as uint8.
    """
    flat = indices.reshape(-1).to(torch.int32)
    n = flat.numel()

    if bits == 4:
        # Nibble packing: 2 values per byte
        pad = (2 - n % 2) % 2
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        even = flat[0::2].to(torch.uint8)
        odd = flat[1::2].to(torch.uint8)
        packed = even | (odd << 4)
        return packed

    if bits == 2:
        # 4 values per byte
        pad = (4 - n % 4) % 4
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        packed = (flat[0::4].to(torch.uint8)
                  | (flat[1::4].to(torch.uint8) << 2)
                  | (flat[2::4].to(torch.uint8) << 4)
                  | (flat[3::4].to(torch.uint8) << 6))
        return packed

    if bits == 3:
        # Pack into uint32: 10 values per 32 bits (30 used, 2 wasted)
        group = 10
        pad = (group - n % group) % group
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        flat = flat.reshape(-1, group)
        packed32 = torch.zeros(flat.shape[0], dtype=torch.int32,
                               device=flat.device)
        for i in range(group):
            packed32 |= (flat[:, i] << (i * 3))
        return packed32.view(torch.uint8)

    if bits == 1:
        # 8 values per byte
        pad = (8 - n % 8) % 8
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        packed = torch.zeros(flat.numel() // 8, dtype=torch.uint8,
                             device=flat.device)
        for i in range(8):
            packed |= (flat[i::8].to(torch.uint8) << i)
        return packed

    return indices.to(torch.uint8)


def unpack_indices(packed: Tensor, bits: int, n_elements: int) -> Tensor:
    """Unpack bit-packed tensor back to indices."""
    if bits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        unpacked = torch.stack([low, high], dim=-1).reshape(-1)
        return unpacked[:n_elements]

    if bits == 2:
        b0 = packed & 0x03
        b1 = (packed >> 2) & 0x03
        b2 = (packed >> 4) & 0x03
        b3 = (packed >> 6) & 0x03
        unpacked = torch.stack([b0, b1, b2, b3], dim=-1).reshape(-1)
        return unpacked[:n_elements]

    if bits == 3:
        packed32 = packed.view(torch.int32)
        parts = []
        for i in range(10):
            parts.append((packed32 >> (i * 3)) & 0x07)
        unpacked = torch.stack(parts, dim=-1).reshape(-1)
        return unpacked[:n_elements]

    if bits == 1:
        parts = []
        for i in range(8):
            parts.append((packed >> i) & 0x01)
        unpacked = torch.stack(parts, dim=-1).reshape(-1)
        return unpacked[:n_elements]

    return packed.long()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def compute_distortion(
    original: Tensor,
    reconstructed: Tensor,
) -> dict[str, float]:
    """Compute MSE and inner-product distortion metrics."""
    mse = (original - reconstructed).pow(2).sum(dim=-1).mean().item()
    d = original.shape[-1]
    ip_orig = (original * original).sum(dim=-1)
    ip_recon = (original * reconstructed).sum(dim=-1)
    ip_distortion = (ip_orig - ip_recon).pow(2).mean().item()

    return {
        "mse": mse,
        "inner_product_distortion": ip_distortion,
        "mse_per_dim": mse / d,
    }
