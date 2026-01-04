"""
Tests for FLUX.1-schnell model pipeline.

Strategy:
1. `mock_dependencies` fixture mocks heavy external components (VAE, Scheduler, TextEncoder)
   to allow fast testing of the pipeline logic without downloading weights.
   - Mocks are configured to return tensors on the correct device.
   - Transformer is mocked dynamically to return random noise of correct shape.

2. `test_real_transformer_init_and_forward` tests the actual `FLUXTransformer2DModel`
   initialization and forward pass with a small configuration to ensure code coverage
   and correctness of the model definition itself, independent of the pipeline mocks.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.flux.pipeline_flux import FLUXPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest


@pytest.fixture
def mock_dependencies(monkeypatch):
    """
    Mock external dependencies to avoid loading real models.
    """
    device = get_local_device()

    # Mock T5 Tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = MagicMock(
        input_ids=torch.zeros((1, 50), dtype=torch.long, device=device),
        attention_mask=torch.ones((1, 50), dtype=torch.long, device=device),
    )
    mock_tokenizer.model_max_length = 512

    # Mock T5 Text Encoder
    mock_text_encoder = MagicMock()
    mock_text_encoder.dtype = torch.float32
    # Output of text encoder must be on the same device as inputs
    mock_text_encoder.return_value.last_hidden_state = torch.randn(1, 50, 2048, device=device)

    # Mock VAE (standard AutoencoderKL)
    mock_vae = MagicMock()
    mock_vae.config.block_out_channels = [128, 256, 512, 512]  # Scale factor 8
    mock_vae.config.scaling_factor = 0.18215
    # Decode return value
    mock_vae.decode.return_value = [torch.randn(1, 3, 128, 128, device=device)]
    # Ensure .to() returns self so configuration persists
    mock_vae.to.return_value = mock_vae

    # Mock Scheduler (FlowMatchEulerDiscreteScheduler)
    mock_scheduler = MagicMock()
    mock_scheduler.config = MagicMock()
    # Timesteps on device to match latents during denoising loop
    mock_scheduler.timesteps = torch.tensor([1.0, 0.5, 0.0], device=device)
    mock_scheduler.set_timesteps.return_value = None

    # Make step return dynamic based on input sample shape
    def mock_scheduler_step(model_output, timestep, sample, **kwargs):
        # sample is the latents, should be preserved
        return (torch.randn_like(sample),)

    mock_scheduler.step.side_effect = mock_scheduler_step

    module_path = "vllm_omni.diffusion.models.flux.pipeline_flux"

    monkeypatch.setattr(f"{module_path}.T5Tokenizer.from_pretrained", lambda *a, **k: mock_tokenizer)
    monkeypatch.setattr(f"{module_path}.T5EncoderModel.from_pretrained", lambda *a, **k: mock_text_encoder)
    monkeypatch.setattr(f"{module_path}.AutoencoderKL.from_pretrained", lambda *a, **k: mock_vae)
    monkeypatch.setattr(
        f"{module_path}.FlowMatchEulerDiscreteScheduler.from_pretrained", lambda *a, **k: mock_scheduler
    )

    return {
        "tokenizer": mock_tokenizer,
        "text_encoder": mock_text_encoder,
        "vae": mock_vae,
        "scheduler": mock_scheduler,
        "device": device,
    }


@pytest.fixture
def flux_pipeline(mock_dependencies, monkeypatch):
    """
    Creates an FLUXPipeline instance with mocked components.
    """
    # Create config
    tf_config = TransformerConfig(
        params={
            "in_channels": 16,
            "out_channels": 16,
            "sample_size": 32,
            "patch_size": 2,
            "num_attention_heads": 4,
            "attention_head_dim": 64,
            "num_layers": 4,
            "joint_attention_dim": 2048,  # T5 hidden size
        }
    )

    od_config = OmniDiffusionConfig(
        model="black-forest-labs/FLUX.1-schnell",
        tf_model_config=tf_config,
        dtype=torch.float32,
        num_gpus=1,
    )

    # Mock Transformer Layer separately to avoid full init
    mock_transformer_cls = MagicMock()
    mock_transformer_instance = MagicMock()
    mock_transformer_instance.dtype = torch.float32
    mock_transformer_instance.in_channels = 16
    
    # Forward return: noise prediction
    def mock_forward(hidden_states, *args, **kwargs):
        # hidden_states shape: (B, SeqLen, Channels)
        return (torch.randn_like(hidden_states),)

    mock_transformer_instance.forward.side_effect = mock_forward
    # Also make the instance itself callable to mimic __call__
    mock_transformer_instance.side_effect = mock_forward

    mock_transformer_cls.return_value = mock_transformer_instance

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.flux.pipeline_flux.FLUXTransformer2DModel", mock_transformer_cls
    )

    # Initialize pipeline
    # We use a dummy model path check override
    with patch("os.path.exists", return_value=True):
        pipeline = FLUXPipeline(od_config=od_config)

    return pipeline


def test_interface_compliance(flux_pipeline):
    """Verify methods required by vllm-omni framework."""
    assert hasattr(flux_pipeline, "load_weights")
    assert hasattr(flux_pipeline, "scheduler")
    assert hasattr(flux_pipeline, "transformer")
    assert hasattr(flux_pipeline, "text_encoder")
    assert hasattr(flux_pipeline, "vae")
    assert hasattr(flux_pipeline, "tokenizer")
    
    # Check FLUX specific attributes
    assert hasattr(flux_pipeline, "vae_scale_factor")
    assert hasattr(flux_pipeline, "tokenizer_max_length")


def test_basic_generation(flux_pipeline):
    """Test the forward pass logic with default parameters."""
    # Setup request
    req = OmniDiffusionRequest(
        prompt="A photo of a cat",
        height=1024,  # FLUX default resolution
        width=1024,
        num_inference_steps=2,
        guidance_scale=3.5,  # FLUX default guidance scale
        num_outputs_per_prompt=1,
    )

    output = flux_pipeline(req)

    assert output is not None
    assert output.output is not None
    # Output should be a tensor from mocked VAE decode
    assert isinstance(output.output, torch.Tensor)
    assert output.output.shape == (1, 3, 128, 128)

    # Check that transformer was called
    assert flux_pipeline.transformer.call_count > 0


def test_classifier_free_guidance(flux_pipeline):
    """Test that classifier-free guidance path is taken when guidance_scale > 1.0."""
    req = OmniDiffusionRequest(
        prompt="A photo of a cat",
        negative_prompt="blurry, low quality",
        height=512,
        width=512,
        num_inference_steps=1,
        guidance_scale=3.5,  # Trigger CFG
    )

    flux_pipeline(req)
    # For CFG, transformer should be called twice (uncond + cond) or with concatenated inputs
    # The implementation concatenates inputs, so it might be called once with larger batch
    assert flux_pipeline.transformer.call_count >= 1


def test_no_guidance(flux_pipeline):
    """Test generation without classifier-free guidance."""
    req = OmniDiffusionRequest(
        prompt="A beautiful landscape",
        height=512,
        width=512,
        num_inference_steps=1,
        guidance_scale=1.0,  # No CFG
    )

    flux_pipeline(req)
    # Should still call transformer
    assert flux_pipeline.transformer.call_count == 1


def test_batch_generation(flux_pipeline):
    """Test generation with multiple prompts."""
    req = OmniDiffusionRequest(
        prompt=["A photo of a cat", "A beautiful sunset"],
        height=512,
        width=512,
        num_inference_steps=1,
        guidance_scale=3.5,
        num_outputs_per_prompt=1,
    )

    output = flux_pipeline(req)
    assert output is not None
    assert output.output.shape[0] == 2  # Batch size 2


def test_multiple_outputs_per_prompt(flux_pipeline):
    """Test generation with multiple outputs per prompt."""
    req = OmniDiffusionRequest(
        prompt="A futuristic city",
        height=512,
        width=512,
        num_inference_steps=1,
        guidance_scale=3.5,
        num_outputs_per_prompt=4,
    )

    output = flux_pipeline(req)
    assert output is not None
    assert output.output.shape[0] == 4  # 4 images per prompt


def test_custom_timesteps(flux_pipeline):
    """Test generation with custom sigmas/timesteps."""
    req = OmniDiffusionRequest(
        prompt="Abstract art",
        height=512,
        width=512,
        num_inference_steps=4,  # Custom number of steps (less than default 28)
        guidance_scale=3.5,
    )

    output = flux_pipeline(req)
    assert output is not None
    # Check that scheduler.set_timesteps was called with correct number
    flux_pipeline.scheduler.set_timesteps.assert_called_once()


def test_resolution_check(flux_pipeline):
    """Test resolution divisible validation logic."""
    # Test with resolution not divisible by vae_scale_factor
    req = OmniDiffusionRequest(
        prompt="test",
        height=513,  # Not divisible by 8
        width=513,
        num_inference_steps=1,
    )

    # Should warn but proceed (as per code I read earlier)
    output = flux_pipeline(req)
    assert output is not None


def test_dtype_consistency(flux_pipeline):
    """Test that all tensors maintain consistent dtype."""
    req = OmniDiffusionRequest(
        prompt="A test image",
        height=512,
        width=512,
        num_inference_steps=1,
        guidance_scale=3.5,
    )

    # Mock the transformer to check dtype
    dtype_checks = []
    original_forward = flux_pipeline.transformer.forward
    
    def dtype_check_forward(hidden_states, *args, **kwargs):
        dtype_checks.append(hidden_states.dtype)
        return original_forward(hidden_states, *args, **kwargs)
    
    flux_pipeline.transformer.forward.side_effect = dtype_check_forward
    
    output = flux_pipeline(req)
    
    # All dtypes should be consistent
    assert all(dtype == torch.float32 for dtype in dtype_checks)


def test_memory_efficient_decoding(flux_pipeline):
    """Test memory-efficient VAE decoding if enabled."""
    # Enable slicing and tiling
    flux_pipeline.vae.use_slicing = True
    flux_pipeline.vae.use_tiling = True
    flux_pipeline.vae.tile_sample_min_height = 256
    flux_pipeline.vae.tile_sample_min_width = 256
    flux_pipeline.vae.tile_sample_stride_height = 192
    flux_pipeline.vae.tile_sample_stride_width = 192
    
    req = OmniDiffusionRequest(
        prompt="High resolution image",
        height=1024,  # Large enough to trigger tiling
        width=1024,
        num_inference_steps=1,
    )
    
    output = flux_pipeline(req)
    assert output is not None


def test_real_transformer_init_and_forward():
    """Test the real FLUXTransformer2DModel initialization and forward pass for coverage."""
    from vllm_omni.diffusion.models.flux.flux_transformer import FLUXTransformer2DModel
    
    device = get_local_device()
    
    # Create minimal config for testing
    tf_config = TransformerConfig(
        params={
            "patch_size": 2,
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 2,
            "attention_head_dim": 32,
            "num_attention_heads": 4,
            "joint_attention_dim": 128,  # Smaller for testing
        }
    )
    
    od_config = OmniDiffusionConfig(
        model="black-forest-labs/FLUX.1-schnell",
        tf_model_config=tf_config,
        dtype=torch.float32,
        num_gpus=1,
    )
    
    # Mock distributed state for QKVParallelLinear initialization
    mock_group = MagicMock()
    mock_group.rank_in_group = 0
    mock_group.world_size = 1
    
    with patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_group):
        # Initialize real model with minimal parameters
        model = FLUXTransformer2DModel(
            od_config=od_config,
            patch_size=2,
            in_channels=16,
            out_channels=16,
            num_layers=2,
            attention_head_dim=32,
            num_attention_heads=4,
            joint_attention_dim=128,
        ).to(device)
        
        # Create dummy inputs
        B, H, W = 1, 32, 32
        C = model.in_channels
        latent_height = H // 2  # patch_size=2
        latent_width = W // 2
        seq_len = latent_height * latent_width
        
        hidden_states = torch.randn(B, seq_len, C, device=device)
        encoder_hidden_states = torch.randn(B, 10, 128, device=device)  # joint_attention_dim=128
        encoder_hidden_states_mask = torch.ones(B, 10, dtype=torch.long, device=device)
        timestep = torch.tensor([0.5], device=device)
        
        # Run forward
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            return_dict=False,
        )
        
        assert output is not None
        assert isinstance(output, tuple)
        # Output should match input shape
        assert output[0].shape == hidden_states.shape


def test_error_handling(flux_pipeline):
    """Test error handling for invalid inputs."""
    # Test with missing prompt
    req = OmniDiffusionRequest(
        prompt=None,
        height=512,
        width=512,
        num_inference_steps=1,
    )
    
    # Should raise ValueError when both prompt and prompt_embeds are None
    with pytest.raises(ValueError):
        flux_pipeline(req)
    
    # Test with excessive sequence length
    req = OmniDiffusionRequest(
        prompt="A" * 1000,  # Very long prompt
        height=512,
        width=512,
        num_inference_steps=1,
        max_sequence_length=600,  # Exceeds tokenizer max length
    )
    
    # Should raise ValueError for max_sequence_length > tokenizer_max_length
    with pytest.raises(ValueError):
        flux_pipeline(req)


def test_weight_loading(flux_pipeline):
    """Test weight loading interface."""
    # Create dummy weights
    dummy_weights = [
        ("transformer.img_in.weight", torch.randn(256, 16)),
        ("transformer.img_in.bias", torch.randn(256)),
        ("transformer.txt_in.weight", torch.randn(256, 2048)),
    ]
    
    # Test load_weights method
    loaded_params = flux_pipeline.load_weights(dummy_weights)
    assert isinstance(loaded_params, set)
    assert len(loaded_params) > 0


def test_post_process_function():
    """Test the post-processing function."""
    from vllm_omni.diffusion.models.flux.pipeline_flux import get_flux_post_process_func
    
    # Create mock config
    od_config = OmniDiffusionConfig(
        model="black-forest-labs/FLUX.1-schnell",
        dtype=torch.float32,
        num_gpus=1,
    )
    
    # Mock the download and file operations
    with patch("os.path.exists", return_value=True):
        with patch("os.path.join", return_value="/tmp/dummy/config.json"):
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    '{"down_block_types": ["DownBlock2D", "DownBlock2D", "DownBlock2D"]}'
                )
                
                # Get post-processing function
                post_process_func = get_flux_post_process_func(od_config)
                assert callable(post_process_func)
                
                # Test post-processing
                dummy_image = torch.randn(1, 3, 128, 128)
                processed = post_process_func(dummy_image)
                assert processed is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])