import torch


def validate_qwen_prompt_sequence_lengths(
    attention_mask: torch.Tensor,
    *,
    drop_idx: int,
    max_sequence_length: int,
    supported_max_sequence_length: int,
    prompt_name: str = "prompt",
) -> None:
    sequence_lengths = torch.clamp(attention_mask.sum(dim=1) - drop_idx, min=0)
    too_long = torch.nonzero(sequence_lengths > max_sequence_length, as_tuple=False)
    if too_long.numel() == 0:
        return

    batch_idx = int(too_long[0].item())
    actual_length = int(sequence_lengths[batch_idx].item())
    prompt_ref = f"`{prompt_name}` at batch index {batch_idx}" if attention_mask.shape[0] > 1 else f"`{prompt_name}`"
    raise ValueError(
        f"{prompt_ref} is too long after applying the Qwen prompt template: got {actual_length} tokens, but "
        f"`max_sequence_length` is {max_sequence_length}. Shorten the prompt or increase "
        f"`max_sequence_length` up to {supported_max_sequence_length}."
    )
