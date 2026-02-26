"""Minimal HuggingFace-compatible config for Chatterbox Turbo.

Chatterbox does not ship a standard ``config.json``.  This class provides the
parameters that vLLM needs to instantiate the T3 backbone (GPT-2-medium) and
S3Gen vocoder so that ``hf_overrides`` in the stage YAML can force the
architecture name.
"""

from transformers import PretrainedConfig


class ChatterboxTurboConfig(PretrainedConfig):
    model_type = "chatterbox_turbo"

    def __init__(
        self,
        # T3 GPT-2-medium backbone
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        vocab_size: int = 50276,
        speech_vocab_size: int = 6563,
        max_position_embeddings: int = 8196,
        # Special tokens
        start_speech_token: int = 6561,
        stop_speech_token: int = 6562,
        # Conditioning
        speaker_embed_size: int = 256,
        speech_cond_prompt_len: int = 375,
        use_perceiver_resampler: bool = False,
        emotion_adv: bool = False,
        # S3Gen
        s3gen_sample_rate: int = 24000,
        s3_token_rate: int = 25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.speech_vocab_size = speech_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.start_speech_token = start_speech_token
        self.stop_speech_token = stop_speech_token
        self.speaker_embed_size = speaker_embed_size
        self.speech_cond_prompt_len = speech_cond_prompt_len
        self.use_perceiver_resampler = use_perceiver_resampler
        self.emotion_adv = emotion_adv
        self.s3gen_sample_rate = s3gen_sample_rate
        self.s3_token_rate = s3_token_rate
