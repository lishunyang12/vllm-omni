# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen3-Omni async end-to-end offline inference.

Supports:
- V0 engine: vllm_omni.entrypoints.async_omni.AsyncOmni
- V1 engine: vllm_omni.entrypoints.async_omni_v1.AsyncOmniV1
- Optional compare mode to run V0 and V1 sequentially with identical inputs
- Optional async-chunk mode
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, NamedTuple

import librosa
import numpy as np
import soundfile as sf
from PIL import Image
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset, video_to_ndarrays
from vllm.multimodal.image import convert_image_mode
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

SEED = 42


class QueryResult(NamedTuple):
    inputs: dict[str, Any]
    limit_mm_per_prompt: dict[str, int]


default_system = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def get_text_query(question: str | None = None) -> QueryResult:
    if question is None:
        question = "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return QueryResult(
        inputs={
            "prompt": prompt,
        },
        limit_mm_per_prompt={},
    )


def get_video_query(question: str | None = None, video_path: str | None = None, num_frames: int = 16) -> QueryResult:
    if question is None:
        question = "Why is this video funny?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_frames = video_to_ndarrays(video_path, num_frames=num_frames)
    else:
        video_frames = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "video": video_frames,
            },
        },
        limit_mm_per_prompt={"video": 1},
    )


def get_image_query(question: str | None = None, image_path: str | None = None) -> QueryResult:
    if question is None:
        question = "What is the content of this image?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        pil_image = Image.open(image_path)
        image_data = convert_image_mode(pil_image, "RGB")
    else:
        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data,
            },
        },
        limit_mm_per_prompt={"image": 1},
    )


def get_audio_query(question: str | None = None, audio_path: str | None = None, sampling_rate: int = 16000) -> QueryResult:
    if question is None:
        question = "What is the content of this audio?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
            },
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_mixed_modalities_query(
    video_path: str | None = None,
    image_path: str | None = None,
    audio_path: str | None = None,
    num_frames: int = 16,
    sampling_rate: int = 16000,
) -> QueryResult:
    question = "What is recited in the audio? What is the content of this image? Why is this video funny?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|video_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_frames = video_to_ndarrays(video_path, num_frames=num_frames)
    else:
        video_frames = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        pil_image = Image.open(image_path)
        image_data = convert_image_mode(pil_image, "RGB")
    else:
        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
                "image": image_data,
                "video": video_frames,
            },
        },
        limit_mm_per_prompt={"audio": 1, "image": 1, "video": 1},
    )


def get_multi_audios_query() -> QueryResult:
    question = "Are these two audio clips the same?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        "<|audio_start|><|audio_pad|><|audio_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": [
                    AudioAsset("winning_call").audio_and_sample_rate,
                    AudioAsset("mary_had_lamb").audio_and_sample_rate,
                ],
            },
        },
        limit_mm_per_prompt={
            "audio": 2,
        },
    )


def get_use_audio_in_video_query() -> QueryResult:
    question = "Describe the content of the video in details, then convert what the baby say into text."
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    asset = VideoAsset(name="baby_reading", num_frames=16)
    audio = asset.get_audio(sampling_rate=16000)
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "video": asset.np_ndarrays,
                "audio": audio,
            },
            "mm_processor_kwargs": {
                "use_audio_in_video": True,
            },
        },
        limit_mm_per_prompt={"audio": 1, "video": 1},
    )


query_map = {
    "text": get_text_query,
    "use_audio": get_audio_query,
    "use_image": get_image_query,
    "use_video": get_video_query,
    "use_multi_audios": get_multi_audios_query,
    "use_mixed_modalities": get_mixed_modalities_query,
    "use_audio_in_video": get_use_audio_in_video_query,
}


def _resolve_engine_versions(args) -> list[str]:
    if args.compare_v0_v1:
        return ["v0", "v1"]
    if args.engine_version in ("v0", "v1"):
        return [args.engine_version]
    return ["v1" if os.getenv("VLLM_OMNI_USE_V1") == "1" else "v0"]


def _resolve_stage_configs_path(args) -> str | None:
    if args.stage_configs_path:
        return args.stage_configs_path
    if not args.async_chunk:
        return None
    repo_root = Path(__file__).resolve().parents[3]
    async_cfg = repo_root / "vllm_omni" / "model_executor" / "stage_configs" / "qwen3_omni_moe_async_chunk.yaml"
    return str(async_cfg)


def _build_query(args) -> QueryResult:
    query_func = query_map[args.query_type]
    if args.query_type == "use_video":
        return query_func(video_path=args.video_path, num_frames=args.num_frames)
    if args.query_type == "use_image":
        return query_func(image_path=args.image_path)
    if args.query_type == "use_audio":
        return query_func(audio_path=args.audio_path, sampling_rate=args.sampling_rate)
    if args.query_type == "use_mixed_modalities":
        return query_func(
            video_path=args.video_path,
            image_path=args.image_path,
            audio_path=args.audio_path,
            num_frames=args.num_frames,
            sampling_rate=args.sampling_rate,
        )
    if args.query_type == "use_multi_audios":
        return query_func()
    if args.query_type == "use_audio_in_video":
        return query_func()
    return query_func()


def _build_prompts(args, query_result: QueryResult) -> list[dict[str, Any]]:
    if args.txt_prompts is None:
        prompts = [query_result.inputs for _ in range(args.num_prompts)]
    else:
        assert args.query_type == "text", "txt-prompts is only supported for text query type"
        with open(args.txt_prompts, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
            prompts = [get_text_query(ln).inputs for ln in lines if ln != ""]
            print(f"[Info] Loaded {len(prompts)} prompts from {args.txt_prompts}")

    if args.modalities is not None:
        output_modalities = args.modalities.split(",")
        for prompt in prompts:
            prompt["modalities"] = output_modalities
    return prompts


def _iter_request_outputs(stage_output: Any) -> list[Any]:
    outputs = stage_output.request_output
    if isinstance(outputs, list):
        return outputs
    return [outputs]


def _extract_audio_numpy(output: Any) -> np.ndarray:
    mm = output.outputs[0].multimodal_output
    audio_val = mm["audio"] if "audio" in mm else mm["model_outputs"]
    if isinstance(audio_val, list):
        if not audio_val:
            return np.array([], dtype=np.float32)
        if hasattr(audio_val[0], "detach"):
            import torch

            audio_val = torch.cat(audio_val, dim=-1)
        else:
            audio_val = np.concatenate(audio_val, axis=-1)
    if hasattr(audio_val, "float"):
        audio_val = audio_val.float().detach().cpu().numpy()
    audio_np = np.asarray(audio_val, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    return audio_np


async def _run_single_engine(
    engine_version: str,
    args,
    prompts: list[dict[str, Any]],
    sampling_params_list: list[SamplingParams],
    output_modalities: list[str] | None,
    stage_configs_path: str | None,
) -> None:
    if engine_version == "v1":
        from vllm_omni.entrypoints.async_omni_v1 import AsyncOmniV1 as AsyncOmniCls

        omni = AsyncOmniCls(
            model=args.model,
            stage_configs_path=stage_configs_path,
            stage_init_timeout=args.stage_init_timeout,
            log_stats=args.log_stats,
            async_chunk=args.async_chunk,
        )
    else:
        from vllm_omni.entrypoints.async_omni import AsyncOmni as AsyncOmniCls

        omni = AsyncOmniCls(
            model=args.model,
            stage_configs_path=stage_configs_path,
            stage_init_timeout=args.stage_init_timeout,
            log_stats=args.log_stats,
            batch_timeout=args.batch_timeout,
            init_timeout=args.init_timeout,
            shm_threshold_bytes=args.shm_threshold_bytes,
            async_chunk=args.async_chunk,
        )

    base_output_dir = Path(args.output_dir if getattr(args, "output_dir", None) else args.output_wav)
    output_dir = base_output_dir / engine_version if args.compare_v0_v1 else base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    async def process_single_prompt(prompt: dict[str, Any], request_id: str):
        async for stage_outputs in omni.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
            output_modalities=output_modalities,
        ):
            if stage_outputs.final_output_type == "text":
                for output in _iter_request_outputs(stage_outputs):
                    out_txt = output_dir / f"{output.request_id}.txt"
                    text_output = output.outputs[0].text
                    prompt_text = output.prompt
                    lines = [
                        "Prompt:\n",
                        str(prompt_text) + "\n",
                        "vllm_text_output:\n",
                        str(text_output).strip() + "\n",
                    ]
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                    print(f"[{engine_version}] Request ID: {output.request_id}, Text saved to {out_txt}")
            elif stage_outputs.final_output_type == "audio":
                for output in _iter_request_outputs(stage_outputs):
                    audio_np = _extract_audio_numpy(output)
                    output_wav = output_dir / f"output_{output.request_id}.wav"
                    sf.write(str(output_wav), audio_np, samplerate=24000, format="WAV")
                    print(f"[{engine_version}] Request ID: {output.request_id}, Saved audio to {output_wav}")

    try:
        tasks = []
        for i, prompt in enumerate(prompts):
            request_id = f"{engine_version}_{i}_{uuid.uuid4()}"
            tasks.append(process_single_prompt(prompt, request_id))
        await asyncio.gather(*tasks)
    finally:
        omni.shutdown()


async def main(args):
    query_result = _build_query(args)
    prompts = _build_prompts(args, query_result)

    thinker_sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.9,
        top_k=-1,
        max_tokens=1200,
        repetition_penalty=1.05,
        logit_bias={},
        seed=SEED,
    )
    talker_sampling_params = SamplingParams(
        temperature=0.9,
        top_k=50,
        max_tokens=4096,
        seed=SEED,
        detokenize=False,
        repetition_penalty=1.05,
        stop_token_ids=[2150],  # TALKER_CODEC_EOS_TOKEN_ID
    )
    code2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=4096 * 16,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.1,
    )
    sampling_params_list = [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]

    output_modalities = args.modalities.split(",") if args.modalities else None
    stage_configs_path = _resolve_stage_configs_path(args)
    versions = _resolve_engine_versions(args)

    print(f"[Info] model={args.model}")
    print(f"[Info] versions={versions}, async_chunk={args.async_chunk}")
    print(f"[Info] stage_configs_path={stage_configs_path}")
    print(f"[Info] query_type={args.query_type}, prompts={len(prompts)}")

    for version in versions:
        await _run_single_engine(
            engine_version=version,
            args=args,
            prompts=prompts,
            sampling_params_list=sampling_params_list,
            output_modalities=output_modalities,
            stage_configs_path=stage_configs_path,
        )


def parse_args():
    parser = FlexibleArgumentParser(description="Qwen3-Omni async end-to-end inference (V0/V1 + optional async chunk)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model name or path.")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="use_mixed_modalities",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--engine-version",
        type=str,
        default="auto",
        choices=["auto", "v0", "v1"],
        help="Engine version. auto uses VLLM_OMNI_USE_V1 env var.",
    )
    parser.add_argument(
        "--compare-v0-v1",
        action="store_true",
        default=False,
        help="Run both engines sequentially and save outputs under output_dir/v0 and output_dir/v1.",
    )
    parser.add_argument(
        "--async-chunk",
        action="store_true",
        default=False,
        help="Enable async chunk mode. If --stage-configs-path is omitted, uses qwen3_omni_moe_async_chunk.yaml.",
    )
    parser.add_argument("--log-stats", action="store_true", default=False, help="Enable statistics logging.")
    parser.add_argument("--stage-init-timeout", type=int, default=300, help="Stage initialization timeout in seconds.")
    parser.add_argument("--batch-timeout", type=int, default=5, help="Batch timeout for V0 engine.")
    parser.add_argument("--init-timeout", type=int, default=300, help="Initialization timeout for V0 engine.")
    parser.add_argument(
        "--shm-threshold-bytes",
        type=int,
        default=65536,
        help="Shared-memory threshold for V0 engine.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Optional stage config YAML. Overrides async-chunk auto config path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_audio",
        help="Output directory for text/audio artifacts.",
    )
    parser.add_argument(
        "--output-wav",
        default="output_audio",
        help="[Deprecated] Output directory alias for backward compatibility.",
    )
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of prompts to generate.")
    parser.add_argument("--txt-prompts", type=str, default=None, help="Path to a .txt file with one prompt per line.")
    parser.add_argument("--video-path", "-v", type=str, default=None, help="Local video path.")
    parser.add_argument("--image-path", "-i", type=str, default=None, help="Local image path.")
    parser.add_argument("--audio-path", "-a", type=str, default=None, help="Local audio path.")
    parser.add_argument("--num-frames", type=int, default=16, help="Video frames to sample.")
    parser.add_argument("--sampling-rate", type=int, default=16000, help="Audio sampling rate.")
    parser.add_argument("--modalities", type=str, default=None, help="Comma-separated output modalities.")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
