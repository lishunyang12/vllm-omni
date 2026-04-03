from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def latent2vae(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data

    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    source_outputs = stage_list[source_stage_id].engine_outputs
    if source_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    vae_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        output = source_output.outputs[0]
        multimodal_output = getattr(output, "multimodal_output", None)
        if not isinstance(multimodal_output, dict) or "latent_audio_feat" not in multimodal_output:
            raise ValueError(
                "VoxCPM latent stage output missing 'latent_audio_feat'. "
                f"request_id={getattr(source_output, 'request_id', None)}"
            )

        additional_information = {
            "latent_audio_feat": multimodal_output["latent_audio_feat"],
        }
        if "sr" in multimodal_output:
            additional_information["sample_rate"] = [int(multimodal_output["sr"])]

        vae_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return vae_inputs


def latent2vae_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Stage-0 latent → stage-1 VAE under ``async_chunk`` (connector payload)."""
    del transfer_manager
    finished_request = bool(is_finished)
    if callable(getattr(request, "is_finished", None)):
        finished_request = finished_request or bool(request.is_finished())
    if not isinstance(pooling_output, dict):
        if finished_request:
            return {
                "code_predictor_codes": [0],
                "finished": True,
            }
        return None

    latent = pooling_output.get("latent_audio_feat")
    if isinstance(latent, torch.Tensor) and latent.numel() == 0:
        latent = None

    if latent is None:
        if finished_request:
            return {
                "code_predictor_codes": [0],
                "finished": True,
            }
        return None

    sr = pooling_output.get("sr")
    out: dict[str, Any] = {
        "code_predictor_codes": [0],
        "latent_audio_feat": latent.detach().cpu().contiguous()
        if isinstance(latent, torch.Tensor)
        else latent,
        "finished": finished_request,
    }
    if isinstance(sr, torch.Tensor):
        out["sr"] = sr.detach().cpu().contiguous()
    return out
