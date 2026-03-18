"""Offline split-stage VoxCPM inference example for vLLM Omni."""

import os
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import Omni

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE_CONFIG = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm.yaml"


def _build_prompt(args) -> dict[str, Any]:
    additional_information: dict[str, list[Any]] = {
        "text": [args.text],
        "cfg_value": [args.cfg_value],
        "inference_timesteps": [args.inference_timesteps],
        "min_len": [args.min_len],
        "max_new_tokens": [args.max_new_tokens],
    }

    if args.ref_audio:
        additional_information["ref_audio"] = [args.ref_audio]
    if args.ref_text:
        additional_information["ref_text"] = [args.ref_text]

    return {
        "prompt_token_ids": [1],
        "additional_information": additional_information,
    }


def _extract_audio_tensor(mm: dict[str, Any]) -> torch.Tensor:
    audio = mm.get("audio", mm.get("model_outputs"))
    if audio is None:
        raise ValueError("No audio output found in multimodal output.")
    if isinstance(audio, list):
        audio = torch.cat(audio, dim=-1)
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio)
    return audio.float().cpu().reshape(-1)


def _extract_sample_rate(mm: dict[str, Any]) -> int:
    sr_raw = mm.get("sr", 24000)
    if isinstance(sr_raw, list) and sr_raw:
        sr_raw = sr_raw[-1]
    if hasattr(sr_raw, "item"):
        return int(sr_raw.item())
    return int(sr_raw)


def _save_wav(mm: dict[str, Any], output_dir: Path, request_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"output_{request_id}.wav"
    sf.write(output_path, _extract_audio_tensor(mm).numpy(), _extract_sample_rate(mm), format="WAV")
    return output_path


def parse_args():
    parser = FlexibleArgumentParser(description="Offline split-stage VoxCPM inference with vLLM Omni")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VOXCPM_MODEL"),
        help="Local VoxCPM model directory. Defaults to $VOXCPM_MODEL.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is a split-stage VoxCPM synthesis example running on vLLM Omni.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Optional reference audio path for voice cloning.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of the reference audio.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=str(DEFAULT_STAGE_CONFIG),
        help="Stage config YAML path. Defaults to the split-stage VoxCPM config.",
    )
    parser.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
        help="Classifier-free guidance value for VoxCPM.",
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="Number of inference timesteps.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=2,
        help="Minimum generated token length.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum generated token length.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_audio",
        help="Directory for output WAV files.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Stage initialization timeout in seconds.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Enable vLLM Omni stats logging.",
    )
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required unless $VOXCPM_MODEL is set")
    if (args.ref_audio is None) != (args.ref_text is None):
        parser.error("--ref-audio and --ref-text must be provided together")

    return args


def main(args) -> None:
    prompt = _build_prompt(args)
    output_dir = Path(args.output_dir)

    print(f"Model: {args.model}")
    print(f"Stage config: {args.stage_configs_path}")
    print(f"Voice cloning: {'enabled' if args.ref_audio else 'disabled'}")

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    t_start = time.perf_counter()
    saved_paths: list[Path] = []
    for stage_outputs in omni.generate([prompt]):
        for output in stage_outputs.request_output:
            mm = output.outputs[0].multimodal_output
            saved_paths.append(_save_wav(mm, output_dir, output.request_id))
    elapsed = time.perf_counter() - t_start

    for path in saved_paths:
        print(f"Saved audio to: {path}")
    print(f"Generation finished in {elapsed:.2f}s")


if __name__ == "__main__":
    main(parse_args())
