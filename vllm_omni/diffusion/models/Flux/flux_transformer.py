# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from math import prod
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.base import CachedTransformer
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm


class FLUXTimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1000
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim
        )

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_states.dtype)
        )
        return timesteps_emb


class FLUXCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        
        # QKV projections
        self.to_q = ReplicatedLinear(dim, dim)
        self.to_k = ReplicatedLinear(dim, dim)
        self.to_v = ReplicatedLinear(dim, dim)
        
        # QK normalization
        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        
        # Output projection
        self.to_out = ReplicatedLinear(dim, dim, bias=out_bias)
        
        # Attention layer
        self.attn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor = None,
    ):
        batch_size = hidden_states.shape[0]
        
        # Project to Q, K, V
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        
        # Reshape for multi-head attention
        query = query.unflatten(-1, (self.num_heads, -1))
        key = key.unflatten(-1, (self.num_heads, -1))
        value = value.unflatten(-1, (self.num_heads, -1))
        
        # Apply QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)
        
        # Compute attention
        attention_output = self.attn(
            query,
            key,
            value,
            attn_metadata=AttentionMetadata() if encoder_hidden_states_mask is None else None
        )
        
        # Reshape back
        attention_output = attention_output.flatten(2, 3)
        attention_output = attention_output.to(query.dtype)
        
        # Output projection
        attention_output, _ = self.to_out(attention_output)
        
        return attention_output


class FLUXTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        
        # Self-attention
        self.norm1 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn1 = FLUXCrossAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            qk_norm=qk_norm,
            eps=eps,
        )
        
        # Cross-attention
        self.norm2 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn2 = FLUXCrossAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            qk_norm=qk_norm,
            eps=eps,
        )
        
        # Feed-forward network
        self.norm3 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate"
        )
        
        # Modulation layers
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

    def _modulate(self, x, shift, scale, gate):
        return x * (1 + scale) + shift, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
    ):
        # Get modulation parameters
        mod_params = self.modulation(temb)
        mod1, mod2, mod3 = mod_params.chunk(3, dim=-1)
        
        # Split each modulation into shift, scale, gate
        shift1, scale1, gate1 = mod1.chunk(3, dim=-1)
        shift2, scale2, gate2 = mod2.chunk(3, dim=-1)
        shift3, scale3, gate3 = mod3.chunk(3, dim=-1)
        
        # Self-attention with modulation
        modulated1, gate1_out = self.norm1(hidden_states, (shift1, scale1, gate1))
        attn_output1 = self.attn1(
            modulated1,
            modulated1,
            encoder_hidden_states_mask
        )
        hidden_states = hidden_states + gate1_out * attn_output1
        
        # Cross-attention with modulation
        modulated2, gate2_out = self.norm2(hidden_states, (shift2, scale2, gate2))
        attn_output2 = self.attn2(
            modulated2,
            encoder_hidden_states,
            encoder_hidden_states_mask
        )
        hidden_states = hidden_states + gate2_out * attn_output2
        
        # Feed-forward with modulation
        modulated3, gate3_out = self.norm3(hidden_states, (shift3, scale3, gate3))
        ff_output = self.ff(modulated3)
        hidden_states = hidden_states + gate3_out * ff_output
        
        return hidden_states


class FLUXTransformer2DModel(CachedTransformer):
    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: int | None = None,
        num_layers: int = 40,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 2048,  # T5 hidden size
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        model_config = od_config.tf_model_config
        num_layers = model_config.num_layers
        self.parallel_config = od_config.parallel_config
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        
        # Time and text embeddings
        self.time_text_embed = FLUXTimestepEmbeddings(embedding_dim=self.inner_dim)
        
        # Input projections
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            FLUXTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=self.inner_dim,
                qk_norm=True,
                eps=1e-6,
            )
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True
        )
        
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        # Handle sequence parallel
        if self.parallel_config.sequence_parallel_size > 1:
            hidden_states = torch.chunk(
                hidden_states,
                get_sequence_parallel_world_size(),
                dim=-2
            )[get_sequence_parallel_rank()]
            
        # Project inputs
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        
        # Time embeddings
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)
        temb = self.time_text_embed(timestep, hidden_states)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
            )
        
        # Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        
        # Gather sequence parallel output
        if self.parallel_config.sequence_parallel_size > 1:
            output = get_sp_group().all_gather(output, dim=-2)
            
        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        
        # Load buffers for beta and eps
        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer
                
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                
        return loaded_params