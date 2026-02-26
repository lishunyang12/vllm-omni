"""Stage input processor for Chatterbox TTS: T3 -> S3Gen."""

from typing import Any

import torch


def _extract_speech_token(pooling_output: dict[str, Any]) -> int | None:
    """Extract the latest speech token from T3 pooling output."""
    speech_tokens = pooling_output.get("speech_tokens")
    if not isinstance(speech_tokens, torch.Tensor) or speech_tokens.numel() == 0:
        return None
    token = int(speech_tokens.reshape(-1)[-1].item())
    return token


def t3_to_s3gen_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """Buffer T3 speech tokens and send chunks to S3Gen.

    Follows the same pattern as ``qwen3_tts.talker2code2wav_async_chunk``.
    Chatterbox uses a single codebook (not multi-quantizer), so each token
    is a scalar rather than a frame vector.

    Chunks are sent when ``chunk_size`` tokens accumulate (25 tokens = 1 second
    at 25 tokens/sec).  Each chunk includes ``left_context_size`` overlap tokens.
    """
    if not isinstance(pooling_output, dict):
        return None

    request_id = request.external_req_id

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size = int(cfg.get("codec_left_context_frames", 25))
    if chunk_size <= 0 or left_context_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size}"
        )

    finished = bool(request.is_finished())

    token = _extract_speech_token(pooling_output)
    if token is not None:
        # Store as single-element list for compatibility with the code_prompt_token_ids interface.
        transfer_manager.code_prompt_token_ids[request_id].append([token])

    length = len(transfer_manager.code_prompt_token_ids[request_id])
    chunk_length = length % chunk_size

    if chunk_length != 0 and not finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size

    if length <= 0:
        return {
            "code_predictor_codes": [],
            "finished": torch.tensor(bool(finished), dtype=torch.bool),
        }

    end_index = min(length, left_context_size + context_length)
    ctx_frames = max(0, int(end_index - context_length))
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Flatten: each element is [token], so just concatenate.
    flat_tokens = [frame[0] for frame in window_frames]

    # Build final prompt_token_ids: [ctx_frames, *flat_tokens]
    return {
        "code_predictor_codes": [int(ctx_frames)] + flat_tokens,
        "finished": torch.tensor(bool(finished), dtype=torch.bool),
    }
