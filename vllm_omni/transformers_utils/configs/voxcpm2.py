# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class VoxCPM2Config(PretrainedConfig):
    model_type = "voxcpm2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        architecture: str = "voxcpm2",
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        intermediate_size: int = 4096,
        max_position_embeddings: int = 8192,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 28,
        num_key_value_heads: int = 16,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        lm_config: dict | None = None,
        encoder_config: dict | None = None,
        dit_config: dict | None = None,
        audio_vae_config: dict | None = None,
        patch_size: int = 4,
        feat_dim: int = 64,
        residual_lm_num_layers: int = 8,
        residual_lm_no_rope: bool = True,
        scalar_quantization_latent_dim: int = 512,
        scalar_quantization_scale: int = 9,
        max_length: int = 8192,
        device: str = "cuda",
        dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.architecture = architecture
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.lm_config = lm_config or {}
        self.encoder_config = encoder_config or {}
        self.dit_config = dit_config or {}
        self.audio_vae_config = audio_vae_config

        self.patch_size = patch_size
        self.feat_dim = feat_dim
        self.residual_lm_num_layers = residual_lm_num_layers
        self.residual_lm_no_rope = residual_lm_no_rope
        self.scalar_quantization_latent_dim = scalar_quantization_latent_dim
        self.scalar_quantization_scale = scalar_quantization_scale
        self.max_length = max_length
        self.device = device
        self.dtype = dtype


AutoConfig.register("voxcpm2", VoxCPM2Config)

__all__ = ["VoxCPM2Config"]
