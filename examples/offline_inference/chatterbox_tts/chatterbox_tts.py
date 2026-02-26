"""Offline inference demo for Chatterbox TTS (Turbo) via vLLM Omni.

Generates speech from text using Chatterbox Turbo with an optional reference
audio for zero-shot voice cloning.  Outputs 24 kHz WAV files.

Usage:
    python chatterbox_tts.py --text "Hello world" --ref-audio ref.wav
    python chatterbox_tts.py --text "Hello world"  # uses default ref audio
"""

import logging
import os

import soundfile as sf
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import Omni

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ResembleAI/chatterbox-turbo"
DEFAULT_STAGE_CONFIGS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "vllm_omni",
    "model_executor",
    "stage_configs",
    "chatterbox_turbo.yaml",
)


def estimate_prompt_len(text: str, speech_cond_prompt_len: int = 375) -> int:
    """Rough estimate of prompt token count for placeholder allocation."""
    text_len = max(1, len(text) // 4 + 10)
    return 1 + speech_cond_prompt_len + text_len + 1


def main(args):
    """Run offline Chatterbox TTS inference."""
    texts = [args.text]
    if args.txt_prompts:
        with open(args.txt_prompts) as f:
            texts = [line.strip() for line in f if line.strip()]
        if not texts:
            raise ValueError(f"No valid prompts found in {args.txt_prompts}")

    ref_audio = args.ref_audio

    inputs = []
    for text in texts:
        additional_information = {
            "text": [text],
        }
        if ref_audio:
            additional_information["ref_audio"] = [ref_audio]

        inputs.append(
            {
                "prompt_token_ids": [0] * estimate_prompt_len(text),
                "additional_information": additional_information,
            }
        )

    stage_configs_path = args.stage_configs_path or DEFAULT_STAGE_CONFIGS
    omni = Omni(
        model=args.model,
        stage_configs_path=stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    omni_generator = omni.generate(inputs, sampling_params_list=None)
    for stage_outputs in omni_generator:
        for output in stage_outputs.request_output:
            request_id = output.request_id
            audio_data = output.outputs[0].multimodal_output["audio"]
            if isinstance(audio_data, list):
                audio_tensor = torch.cat(audio_data, dim=-1)
            else:
                audio_tensor = audio_data

            sr_val = output.outputs[0].multimodal_output["sr"]
            sample_rate = sr_val.item() if hasattr(sr_val, "item") else int(sr_val[-1])

            audio_numpy = audio_tensor.float().detach().cpu().numpy()
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.flatten()

            output_wav = os.path.join(output_dir, f"chatterbox_{request_id}.wav")
            sf.write(output_wav, audio_numpy, samplerate=sample_rate, format="WAV")
            print(f"Request ID: {request_id}, Saved audio to {output_wav}")


def parse_args():
    parser = FlexibleArgumentParser(description="Chatterbox TTS offline inference via vLLM Omni")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! This is a test of the Chatterbox text to speech system.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning (optional).",
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage configs YAML (default: chatterbox_turbo.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="output_audio",
        help="Output directory for generated wav files.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable writing detailed statistics.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing a single stage in seconds.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
