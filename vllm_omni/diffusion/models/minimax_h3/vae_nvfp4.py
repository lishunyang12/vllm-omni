# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load the released FastH3 packed NVFP4 decoder into the native H3 VAE."""

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.layers.nvfp4_linear import PackedNVFP4Linear, require_nvfp4_linear
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

_FORMAT = "fastvideo_h3_decoder_nvfp4_packed_v1"
_LINEAR_TENSORS = ("packed_weight", "weight_inv_scale", "weight_global_sf", "bias", "input_amax")


def released_decoder_pixels(decoded: torch.Tensor) -> torch.Tensor:
    """Apply the release's FP32 pixel normalization before output quantization."""
    pixels = decoded.float()
    shape = (1, 3) + (1,) * (pixels.ndim - 2)
    mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(shape)
    std = pixels.new_tensor((0.229, 0.224, 0.225)).view(shape)
    return (pixels * std + mean).clamp_(0, 1)


def resolve_video_decoder_checkpoint(source: str | Mapping[str, str] | None, device: torch.device) -> Path | None:
    if source is None:
        return None
    require_nvfp4_linear(device)
    if isinstance(source, str):
        path = Path(source)
    elif isinstance(source, Mapping):
        unknown = set(source) - {"model", "revision", "subfolder"}
        if unknown or not isinstance(source.get("model"), str):
            raise ValueError("video_vae_decoder requires model and optional revision/subfolder")
        subfolder = source.get("subfolder", "")
        filename = str(Path(subfolder) / "decoder-nvfp4.safetensors")
        model = source["model"]
        root = Path(model)
        if not root.is_dir():
            root = Path(download_weights_from_hf_specific(model, None, [filename], revision=source.get("revision")))
        path = root / filename
    else:
        raise TypeError("video_vae_decoder must be a local file or a model/revision/subfolder object")
    if not path.is_file():
        raise ValueError(f"NVFP4 decoder checkpoint not found: {path}")
    return path


class _QKVProjection(nn.Module):
    """Preserve each exported projection's global scale and native head packing."""

    def __init__(self, q: nn.Module, k: nn.Module, v: nn.Module, head_dim: int):
        super().__init__()
        self.q, self.k, self.v = q, k, v
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projections = [layer(x).unflatten(-1, (-1, self.head_dim)) for layer in (self.q, self.k, self.v)]
        return torch.cat(projections, dim=-1).flatten(-2)


class _SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Released FastVideo FFN stores [value, gate]; native H3 stores [gate, value].
        value, gate = x.chunk(2, dim=-1)
        return value * F.silu(gate)


class FastH3NVFP4Decoder(nn.Module):
    """Released decoder precision/geometry with the native attention and RoPE."""

    def __init__(self, original: nn.Module, depth: int) -> None:
        super().__init__()
        if original.spatial_parallel or original.t_causal:
            raise ValueError("FastH3 NVFP4 decoder requires non-causal, tile-parallel VAE execution")
        if not 1 <= depth <= len(original.transformer_blocks):
            raise ValueError(f"NVFP4 decoder needs {depth} blocks; host VAE has {len(original.transformer_blocks)}")
        self.proj_in = original.x_embedder
        self.proj_out = original.proj_out
        self.norm_out = original.norm_out
        self.register_tokens = original.register_tokens
        self.rope = original.pos_embed
        self.transformer_blocks = nn.ModuleList(list(original.transformer_blocks)[:depth])
        self.patch_size = original.config.patch_size
        self.patch_size_t = original.config.patch_size_t
        self.out_channels = self.proj_out.out_features // (self.patch_size**2 * self.patch_size_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, frames, height, width = x.shape
        hidden = x.permute(0, 2, 3, 4, 1).flatten(1, 3)
        # Unlike native H3's FP32 projection overrides, the released NVFP4
        # decoder runs both boundary projections under BF16 autocast.
        hidden = self.proj_in(hidden)
        patches = hidden.shape[1]
        registers = self.register_tokens.expand(batch, -1, -1)
        hidden = torch.cat((hidden, registers, torch.zeros_like(hidden[:, :1])), dim=1)
        grids = [
            2.0 * (torch.arange(0.5, n, device=x.device, dtype=torch.float32) / n) - 1.0
            for n in (frames, height, width)
        ]
        positions = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
        positions = positions.unsqueeze(0).expand(batch, -1, -1)
        suffix = positions.new_zeros((batch, registers.shape[1] + 1, 3))
        rope = self.rope(torch.cat((positions, suffix), dim=1))
        for block in self.transformer_blocks:
            hidden = block(hidden, rope)
        hidden = self.proj_out(self.norm_out(hidden))[:, :patches]
        hidden = hidden.reshape(
            batch, frames, height, width, self.out_channels, self.patch_size_t, self.patch_size, self.patch_size
        )
        return hidden.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            batch, self.out_channels, frames * self.patch_size_t, height * self.patch_size, width * self.patch_size
        )


def install_nvfp4_video_decoder(vae: nn.Module, path: Path) -> None:
    """Replace decoder weights on CPU before the VAE moves to its target GPU."""
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        header = checkpoint.metadata() or {}
        if header.get("format") != _FORMAT:
            raise ValueError(f"Expected {_FORMAT}; use the released decoder-nvfp4.safetensors file")
        metadata = json.loads(header.get("nvfp4_metadata", "{}"))
        for key, expected in (("block_size", 16), ("act_scale", "unit"), ("rotation_group", None), ("skip_blocks", [])):
            if key not in metadata or metadata[key] != expected:
                raise ValueError(f"Unsupported NVFP4 decoder {key}: {metadata.get(key)!r}")
        depth = metadata.get("student_layers")
        if type(depth) is not int or metadata.get("num_linears") != depth * 6:
            raise ValueError("NVFP4 decoder metadata must specify six quantized linears per decoder block")
        decoder = FastH3NVFP4Decoder(vae.decoder, depth)
        tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}

    def linear(prefix: str) -> PackedNVFP4Linear:
        try:
            return PackedNVFP4Linear(*(tensors[f"{prefix}.{name}"] for name in _LINEAR_TENSORS))
        except KeyError as exc:
            raise ValueError(f"Missing packed NVFP4 tensor {exc.args[0]}") from exc

    renamed: dict[str, str] = {}
    for i, block in enumerate(decoder.transformer_blocks):
        prefix = f"decoder.transformer_blocks.{i}"
        block.attn.to_qkv = _QKVProjection(
            *(linear(f"{prefix}.attn.to_{part}") for part in ("q", "k", "v")),
            head_dim=block.attn.dim_head,
        )
        block.attn.to_out = linear(f"{prefix}.attn.to_out.0")
        block.ff = nn.Sequential(linear(f"{prefix}.ff.net.0.proj"), _SwiGLU(), linear(f"{prefix}.ff.net.2"))
        for part in ("q", "k", "v"):
            renamed[f"{prefix}.attn.to_{part}."] = f"{prefix}.attn.to_qkv.{part}."
        renamed[f"{prefix}.attn.to_out.0."] = f"{prefix}.attn.to_out."
        renamed[f"{prefix}.ff.net.0.proj."] = f"{prefix}.ff.0."
        renamed[f"{prefix}.ff.net.2."] = f"{prefix}.ff.2."
    mapped = {}
    for name, tensor in tensors.items():
        target = name
        for old, new in renamed.items():
            if name.startswith(old):
                target = new + name[len(old) :]
                break
        mapped[target] = tensor
    # Load only the two components owned by the released checkpoint. Strict
    # loading rejects extra/missing weights and architecture mismatches.
    components = nn.ModuleDict({"decoder": decoder, "post_quant_conv": vae.post_quant_conv})
    components.load_state_dict(mapped, strict=True)
    vae.decoder = decoder
