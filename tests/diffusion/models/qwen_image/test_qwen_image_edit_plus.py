# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import (
    QwenImageEditPlusPipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeProcessorOutput(SimpleNamespace):
    def to(self, _device: str):
        return self


class _FakeProcessor:
    def __call__(self, *, text, images, padding, return_tensors):
        assert padding is True
        assert return_tensors == "pt"
        assert len(text) == 1
        assert len(images) == 2
        return _FakeProcessorOutput(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            attention_mask=torch.tensor([[1, 1, 1, 0]]),
            pixel_values=torch.tensor([1.0]),
            image_grid_thw=torch.tensor([[1, 1, 1]]),
        )


class _FakeTextEncoder:
    dtype = torch.float32

    def __init__(self) -> None:
        self.model_calls = []

    def __call__(self, *args, **kwargs):
        raise AssertionError("full CausalLM forward should not be used for prompt encoding")

    def model(self, **kwargs):
        self.model_calls.append(kwargs)
        return SimpleNamespace(
            last_hidden_state=torch.tensor(
                [
                    [
                        [1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                        [3.0, 3.0, 3.0],
                        [4.0, 4.0, 4.0],
                    ]
                ]
            )
        )


def test_qwen_image_edit_plus_prompt_encoding_skips_lm_head():
    pipeline = QwenImageEditPlusPipeline.__new__(QwenImageEditPlusPipeline)
    pipeline.device = "cpu"
    pipeline.text_encoder = _FakeTextEncoder()
    pipeline.processor = _FakeProcessor()
    pipeline.prompt_template_encode = "{}"
    pipeline.prompt_template_encode_start_idx = 1

    prompt_embeds, attention_mask = pipeline._get_qwen_prompt_embeds(
        prompt="combine",
        image=[object(), object()],
    )

    assert len(pipeline.text_encoder.model_calls) == 1
    model_call = pipeline.text_encoder.model_calls[0]
    assert model_call["output_hidden_states"] is False
    assert model_call["return_dict"] is True
    assert tuple(prompt_embeds.shape) == (1, 2, 3)
    assert tuple(attention_mask.shape) == (1, 2)
    assert torch.equal(prompt_embeds[0], torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))
    assert torch.equal(attention_mask[0], torch.tensor([1, 1]))


def test_qwen_image_edit_plus_encode_prompt_applies_max_sequence_length():
    pipeline = QwenImageEditPlusPipeline.__new__(QwenImageEditPlusPipeline)
    pipeline._get_qwen_prompt_embeds = lambda *args, **kwargs: (
        torch.arange(24, dtype=torch.float32).view(1, 4, 6),
        torch.tensor([[1, 1, 1, 1]]),
    )

    prompt_embeds, attention_mask = pipeline.encode_prompt(
        prompt="combine",
        image=[object()],
        max_sequence_length=2,
    )

    assert tuple(prompt_embeds.shape) == (1, 2, 6)
    assert tuple(attention_mask.shape) == (1, 2)
    assert torch.equal(attention_mask[0], torch.tensor([1, 1]))
