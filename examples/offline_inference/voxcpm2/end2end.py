"""Offline VoxCPM2 inference example for vLLM Omni.

Supports:
- Sync one-shot and streaming modes
- Zero-shot synthesis (text only)
- Voice design (text + control instruction)
- Voice cloning (text + reference audio)
- Batch inputs from txt or jsonl

VoxCPM2 outputs 48kHz audio (vs v1's 24kHz).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE_ASYNC = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm2.yaml"
DEFAULT_STAGE_SYNC = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm2_single_stage.yaml"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PromptSpec:
    text: str
    label: str
    reference_audio: str | None = None
    control_instruction: str | None = None
    prompt_audio: str | None = None
    prompt_text: str | None = None


def _build_prompt(args, *, spec: PromptSpec, global_request_id: str | None = None) -> dict[str, Any]:
    additional_information: dict[str, list[Any]] = {
        "text": [spec.text],
        "cfg_value": [args.cfg_value],
        "inference_timesteps": [args.inference_timesteps],
        "min_len": [args.min_len],
        "max_new_tokens": [args.max_new_tokens],
        "temperature": [args.temperature],
        "top_p": [args.top_p],
    }
    if args.streaming_prefix_len is not None:
        additional_information["streaming_prefix_len"] = [args.streaming_prefix_len]
    if spec.reference_audio:
        additional_information["reference_audio"] = [spec.reference_audio]
    if spec.control_instruction:
        additional_information["control_instruction"] = [spec.control_instruction]
    if spec.prompt_audio:
        additional_information["prompt_audio"] = [spec.prompt_audio]
    if spec.prompt_text:
        additional_information["prompt_text"] = [spec.prompt_text]
    if global_request_id is not None:
        additional_information["global_request_id"] = [global_request_id]

    return {
        "prompt_token_ids": [1],
        "additional_information": additional_information,
    }


def _extract_audio_tensor(mm: dict[str, Any]) -> torch.Tensor:
    audio = mm.get("audio", mm.get("model_outputs"))
    if audio is None:
        raise ValueError("No audio output found in multimodal output.")
    if isinstance(audio, list):
        parts = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio]
        audio = torch.cat(parts, dim=-1) if parts else torch.zeros(0)
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio)
    return audio.float().cpu().reshape(-1)


def _extract_sample_rate(mm: dict[str, Any]) -> int:
    sr_raw = mm.get("sr", 48000)
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


def _iter_request_multimodal_outputs(request_output: Any):
    outputs = getattr(request_output, "outputs", None)
    if outputs:
        for output in outputs:
            mm = getattr(output, "multimodal_output", None)
            if isinstance(mm, dict):
                yield mm
    mm = getattr(request_output, "multimodal_output", None)
    if isinstance(mm, dict):
        yield mm


def _load_prompt_specs(args) -> list[PromptSpec]:
    specs: list[PromptSpec] = []

    if args.jsonl_prompts is not None:
        with open(args.jsonl_prompts, encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = item.get("text", "")
                if not text.strip():
                    raise ValueError(f"{args.jsonl_prompts}:{line_no} requires non-empty 'text'")
                specs.append(PromptSpec(
                    text=text.strip(),
                    label=f"item{len(specs) + 1:03d}",
                    reference_audio=item.get("reference_audio", args.reference_audio),
                    control_instruction=item.get("control_instruction", args.control_instruction),
                    prompt_audio=item.get("prompt_audio"),
                    prompt_text=item.get("prompt_text"),
                ))
        if not specs:
            raise ValueError(f"No prompts found in {args.jsonl_prompts}")
        return specs

    if args.txt_prompts is not None:
        with open(args.txt_prompts, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        if not texts:
            raise ValueError(f"No prompts found in {args.txt_prompts}")
        for idx, text in enumerate(texts, start=1):
            specs.append(PromptSpec(
                text=text,
                label=f"item{idx:03d}",
                reference_audio=args.reference_audio,
                control_instruction=args.control_instruction,
            ))
        return specs

    specs.append(PromptSpec(
        text=args.text,
        label="item001",
        reference_audio=args.reference_audio,
        control_instruction=args.control_instruction,
    ))
    return specs


def _is_streaming_stage_config(path: str) -> bool:
    return "no_async_chunk" not in Path(path).name.lower()


def parse_args():
    parser = FlexibleArgumentParser(description="Offline VoxCPM2 inference with vLLM Omni")
    parser.add_argument("--model", type=str, default=os.environ.get("VOXCPM2_MODEL", "openbmb/VoxCPM2"),
                        help="VoxCPM2 model path or HF repo ID.")
    parser.add_argument("--text", type=str,
                        default="This is a VoxCPM2 synthesis example running on vLLM Omni.",
                        help="Text to synthesize.")
    parser.add_argument("--txt-prompts", type=str, default=None, help="Path to .txt file with one text per line.")
    parser.add_argument("--jsonl-prompts", type=str, default=None, help="Path to .jsonl file.")
    # V2-specific: voice design
    parser.add_argument("--control-instruction", type=str, default=None,
                        help="Natural language voice description for voice design mode. "
                        "E.g. 'A warm female voice with happy tone'.")
    # V2-specific: reference cloning (no transcript needed)
    parser.add_argument("--reference-audio", type=str, default=None,
                        help="Reference audio path for voice cloning (VoxCPM2 does not require transcript).")
    parser.add_argument("--stage-configs-path", type=str, default=str(DEFAULT_STAGE_SYNC),
                        help="Stage config YAML path.")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value.")
    parser.add_argument("--inference-timesteps", type=int, default=10, help="Diffusion timesteps.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling threshold.")
    parser.add_argument("--min-len", type=int, default=2, help="Minimum generated length.")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum generated length.")
    parser.add_argument("--streaming-prefix-len", type=int, default=None, help="Streaming window size.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for output WAV files.")
    parser.add_argument("--stage-init-timeout", type=int, default=600, help="Stage init timeout (seconds).")
    parser.add_argument("--log-stats", action="store_true", help="Enable stats logging.")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of inference runs.")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Warmup runs before measured runs.")

    args = parser.parse_args()
    if args.txt_prompts is not None and args.jsonl_prompts is not None:
        parser.error("--txt-prompts and --jsonl-prompts are mutually exclusive")
    if args.output_dir is None:
        args.output_dir = "output_voxcpm2_streaming" if _is_streaming_stage_config(args.stage_configs_path) else "output_voxcpm2"
    args.prompt_specs = _load_prompt_specs(args)
    return args


async def _collect_streaming_audio(
    omni: AsyncOmni, args, spec: PromptSpec, request_id: str,
) -> tuple[torch.Tensor, int, float, float | None]:
    prompt = _build_prompt(args, spec=spec, global_request_id=request_id)
    delta_chunks: list[torch.Tensor] = []
    sample_rate = 48000
    prev_total_samples = 0
    t_start = time.perf_counter()
    first_audio_elapsed: float | None = None

    async for stage_output in omni.generate(prompt, request_id=request_id):
        ro = stage_output.request_output
        seq = ro.outputs[0] if hasattr(ro, "outputs") and ro.outputs else None
        if seq is None:
            continue
        mm = seq.multimodal_output
        if not isinstance(mm, dict):
            continue
        sample_rate = _extract_sample_rate(mm)
        try:
            w = _extract_audio_tensor(mm)
            n = int(w.numel())
            if n == 0:
                continue
            if n > prev_total_samples:
                delta = w.reshape(-1)[prev_total_samples:]
                prev_total_samples = n
            else:
                delta = w.reshape(-1)
                prev_total_samples += int(delta.numel())
            delta_chunks.append(delta)
            if first_audio_elapsed is None and int(delta.numel()) > 0:
                first_audio_elapsed = time.perf_counter() - t_start
        except ValueError:
            pass

    if not delta_chunks:
        raise RuntimeError("No audio chunks received.")
    return torch.cat([c.reshape(-1) for c in delta_chunks], dim=0), sample_rate, time.perf_counter() - t_start, first_audio_elapsed


async def _run_streaming(args) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    omni = AsyncOmni(
        model=args.model, stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats, stage_init_timeout=args.stage_init_timeout,
    )
    paths: list[Path] = []
    t_total = time.perf_counter()
    for run in range(args.num_runs):
        for pi, spec in enumerate(args.prompt_specs):
            request_id = f"stream_{run + 1}_{spec.label}_{uuid.uuid4().hex[:8]}"
            audio, sr, elapsed, ttfa = await _collect_streaming_audio(omni, args, spec, request_id)
            out_path = output_dir / f"output_run{run + 1}_{spec.label}.wav"
            sf.write(out_path, audio.numpy(), sr, format="WAV")
            ttfa_text = f", ttfa={ttfa:.2f}s" if ttfa is not None else ""
            print(f"Saved (streaming) run {run + 1}, prompt {pi + 1}: {out_path} ({elapsed:.2f}s{ttfa_text})")
            paths.append(out_path)
    print(f"Done: {len(paths)} files in {time.perf_counter() - t_total:.2f}s")
    return paths


def _run_sync(args) -> list[Path]:
    output_dir = Path(args.output_dir)
    omni = Omni(
        model=args.model, stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats, stage_init_timeout=args.stage_init_timeout,
    )
    paths: list[Path] = []
    t_total = time.perf_counter()
    for run in range(args.num_runs):
        for pi, spec in enumerate(args.prompt_specs):
            prompt = _build_prompt(args, spec=spec, global_request_id=f"sync_{run + 1}_{spec.label}")
            t_start = time.perf_counter()
            for stage_outputs in omni.generate(prompt):
                ro = stage_outputs.request_output
                if ro is None:
                    continue
                for mm in _iter_request_multimodal_outputs(ro):
                    out_path = _save_wav(mm, output_dir, f"run{run + 1}_{spec.label}")
                    elapsed = time.perf_counter() - t_start
                    print(f"Saved (sync) run {run + 1}, prompt {pi + 1}: {out_path} ({elapsed:.2f}s)")
                    paths.append(out_path)
    print(f"Done: {len(paths)} files in {time.perf_counter() - t_total:.2f}s")
    return paths


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    is_streaming = _is_streaming_stage_config(args.stage_configs_path)
    print(f"Model: {args.model}")
    print(f"Mode: {'streaming' if is_streaming else 'sync'}")
    print(f"Prompts: {len(args.prompt_specs)}")
    print(f"Output: {args.output_dir}")
    if is_streaming:
        asyncio.run(_run_streaming(args))
    else:
        _run_sync(args)


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
