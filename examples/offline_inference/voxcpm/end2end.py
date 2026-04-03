<<<<<<< HEAD
"""Offline split-stage VoxCPM inference with voice cloning and batch processing for vLLM Omni."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE_CONFIG = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm_async_chunk.yaml"


def _save_wav(output_dir: Path, request_id: str, mm: dict) -> None:
    import soundfile as sf
    import torch

    audio_data = mm.get("audio", mm.get("model_outputs"))
    if audio_data is None:
        raise ValueError("No audio output found in multimodal output.")
    
    sr_raw = mm.get("sr", 24000)
    if isinstance(sr_raw, list) and sr_raw:
        sr_raw = sr_raw[-1]
    sr = int(sr_raw.item()) if hasattr(sr_raw, "item") else int(sr_raw)
    
    if isinstance(audio_data, list):
        audio_tensor = torch.cat(audio_data, dim=-1)
    else:
        audio_tensor = torch.as_tensor(audio_data)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_wav = output_dir / f"output_{request_id}.wav"
    sf.write(out_wav, audio_tensor.float().cpu().numpy().flatten(), samplerate=sr, format="WAV")
    print(f"Saved: {out_wav}")


def _build_synthesize_input(args) -> list[dict]:
    additional_information = {
        "text": [args.text],
        "cfg_value": [args.cfg_value],
        "inference_timesteps": [args.inference_timesteps],
        "min_len": [args.min_len],
        "max_new_tokens": [args.max_new_tokens],
    }
    
    return [{
        "prompt_token_ids": [1],
        "additional_information": additional_information,
    }]


def _build_clone_input(args) -> list[dict]:
    additional_information = {
        "text": [args.text],
        "ref_audio": [args.prompt_audio],
        "ref_text": [args.prompt_text],
        "cfg_value": [args.cfg_value],
        "inference_timesteps": [args.inference_timesteps],
        "min_len": [args.min_len],
        "max_new_tokens": [args.max_new_tokens],
    }
    
    return [{
        "prompt_token_ids": [1],
        "additional_information": additional_information,
    }]


def _build_batch_input(args) -> list[dict]:
    inputs = []
    
    if args.jsonl_file:
        with open(args.jsonl_file, "r", encoding="utf-8") as f:
            items = [json.loads(line.strip()) for line in f if line.strip()]
        
        for item in items:
            additional_information = {
                "text": [item["text"]],
                "cfg_value": [args.cfg_value],
                "inference_timesteps": [args.inference_timesteps],
                "min_len": [args.min_len],
                "max_new_tokens": [args.max_new_tokens],
            }
            
            if "audio" in item:
                additional_information["ref_audio"] = [item["audio"]]
            if "text" in item:
                additional_information["ref_text"] = [item["text"]]
            
            inputs.append({
                "prompt_token_ids": [1],
                "additional_information": additional_information,
            })
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        for text in texts:
            additional_information = {
                "text": [text],
                "cfg_value": [args.cfg_value],
                "inference_timesteps": [args.inference_timesteps],
                "min_len": [args.min_len],
                "max_new_tokens": [args.max_new_tokens],
            }
            
            if args.prompt_audio:
                additional_information["ref_audio"] = [args.prompt_audio]
            if args.prompt_text:
                additional_information["ref_text"] = [args.prompt_text]
            
            inputs.append({
                "prompt_token_ids": [1],
                "additional_information": additional_information,
            })
    
    return inputs


def parse_args():
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        FlexibleArgumentParser = argparse.ArgumentParser

    parser = FlexibleArgumentParser(
        description="Offline split-stage VoxCPM inference with voice cloning and batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single text synthesis
  python end2end.py --model "$VOXCPM_MODEL" --text "Hello world"

  # Voice cloning
  python end2end.py --model "$VOXCPM_MODEL" --text "Hello" --prompt-audio ref.wav --prompt-text "hi"

  # Batch processing from text file
  python end2end.py --model "$VOXCPM_MODEL" --input texts.txt --output-dir ./outs

  # Batch processing from text file with voice cloning
  python end2end.py --model "$VOXCPM_MODEL" --input texts.txt --prompt-audio ref.wav --prompt-text "reference" --output-dir ./outs

  # Batch processing from JSONL file
  python end2end.py --model "$VOXCPM_MODEL" --jsonl-file data.jsonl --output-dir ./outs
        """,
    )

=======
"""Offline VoxCPM inference example for vLLM Omni.

Supports both:
- sync one-shot (Omni.generate)
- streaming (AsyncOmni.generate with async_chunk config)
- text-only synthesis
- voice cloning
- text/clone batch inputs from txt or jsonl
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

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE_ASYNC = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm.yaml"
DEFAULT_STAGE_SYNC = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm_no_async_chunk.yaml"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PromptSpec:
    text: str
    label: str
    ref_audio: str | None = None
    ref_text: str | None = None


def _build_prompt(
    args,
    *,
    text: str,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    global_request_id: str | None = None,
) -> dict[str, Any]:
    additional_information: dict[str, list[Any]] = {
        "text": [text],
        "cfg_value": [args.cfg_value],
        "inference_timesteps": [args.inference_timesteps],
        "min_len": [args.min_len],
        "max_new_tokens": [args.max_new_tokens],
    }
    if args.streaming_prefix_len is not None:
        additional_information["streaming_prefix_len"] = [args.streaming_prefix_len]

    if ref_audio:
        additional_information["ref_audio"] = [ref_audio]
    if ref_text:
        additional_information["ref_text"] = [ref_text]
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


def _read_non_empty_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_prompt_specs(args) -> list[PromptSpec]:
    specs: list[PromptSpec] = []

    if args.txt_prompts is not None:
        texts = _read_non_empty_lines(args.txt_prompts)
        if not texts:
            raise ValueError(f"No prompts found in {args.txt_prompts}")
        for idx, text in enumerate(texts, start=1):
            specs.append(
                PromptSpec(
                    text=text,
                    label=f"item{idx:03d}",
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                )
            )
        return specs

    if args.jsonl_prompts is not None:
        with open(args.jsonl_prompts, encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{args.jsonl_prompts}:{line_no} is not valid JSON: {exc}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"{args.jsonl_prompts}:{line_no} must be a JSON object")

                text = item.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{args.jsonl_prompts}:{line_no} requires non-empty string field 'text'")

                ref_audio = item.get("ref_audio", args.ref_audio)
                ref_text = item.get("ref_text", args.ref_text)
                if (ref_audio is None) != (ref_text is None):
                    raise ValueError(
                        f"{args.jsonl_prompts}:{line_no} must provide both 'ref_audio' and 'ref_text' together"
                    )

                specs.append(
                    PromptSpec(
                        text=text.strip(),
                        label=f"item{len(specs) + 1:03d}",
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                    )
                )

        if not specs:
            raise ValueError(f"No prompts found in {args.jsonl_prompts}")
        return specs

    specs.append(
        PromptSpec(
            text=args.text,
            label="item001",
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
        )
    )
    return specs


def _build_prompt_for_spec(args, spec: PromptSpec, *, global_request_id: str | None = None) -> dict[str, Any]:
    return _build_prompt(
        args,
        text=spec.text,
        ref_audio=spec.ref_audio,
        ref_text=spec.ref_text,
        global_request_id=global_request_id,
    )


def _iter_batches(items: list[PromptSpec], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _resolve_output_label(request_id: Any, batch_specs: list[PromptSpec]) -> str:
    request_id_str = str(request_id)
    if request_id_str.isdigit():
        request_idx = int(request_id_str)
        if 0 <= request_idx < len(batch_specs):
            return batch_specs[request_idx].label
    return f"req_{request_id_str}"


def _count_voice_clone_prompts(prompt_specs: list[PromptSpec]) -> int:
    return sum(1 for spec in prompt_specs if spec.ref_audio is not None)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Offline split-stage VoxCPM inference with vLLM Omni (auto sync/streaming by stage config)"
    )
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VOXCPM_MODEL"),
        help="Local VoxCPM model directory. Defaults to $VOXCPM_MODEL.",
    )
    parser.add_argument(
<<<<<<< HEAD
        "--stage-configs-path",
        type=str,
        default=str(DEFAULT_STAGE_CONFIG),
        help="Stage config YAML path. Defaults to split-stage VoxCPM config.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Text to synthesize (single or clone mode).",
    )
    input_group.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input text file (batch mode only).",
    )
    input_group.add_argument(
        "--jsonl-file",
        type=str,
        default=None,
        help="Path to a JSONL file with audio/text pairs for batch processing.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output audio file path (single or clone mode).",
    )
    parser.add_argument(
        "--output-dir",
        "-od",
        type=str,
        default="output_audio",
        help="Output directory (batch mode only).",
    )

    parser.add_argument(
        "--prompt-audio",
        "-pa",
        type=str,
        default=None,
        help="Reference audio file path (clone mode).",
    )
    parser.add_argument(
        "--prompt-text",
        "-pt",
        type=str,
        default=None,
        help="Reference text corresponding to audio (clone mode).",
    )

=======
        "--text",
        type=str,
        default="This is a split-stage VoxCPM synthesis example running on vLLM Omni.",
        help="Text to synthesize. Ignored when --txt-prompts or --jsonl-prompts is used.",
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one synthesis text per line.",
    )
    parser.add_argument(
        "--jsonl-prompts",
        type=str,
        default=None,
        help="Path to a .jsonl file. Each line must contain at least {'text': ...}; clone rows can also set ref_audio/ref_text.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Optional reference audio path for voice cloning. With --txt-prompts, the same reference is applied to every line.",
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
        default=str(DEFAULT_STAGE_SYNC),
        help="Stage config YAML path. Routing is selected only from this path.",
    )
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d
    parser.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
<<<<<<< HEAD
        help="CFG guidance scale (float, recommended 0.5-5.0, default: 2.0).",
=======
        help="Classifier-free guidance value for VoxCPM.",
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
<<<<<<< HEAD
        help="Inference steps (int, 1-100, default: 10).",
=======
        help="Number of inference timesteps.",
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d
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
<<<<<<< HEAD

=======
    parser.add_argument(
        "--streaming-prefix-len",
        type=int,
        default=None,
        help="VoxCPM streaming window (optional, streaming mode only).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output WAV files.",
    )
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d
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
<<<<<<< HEAD

=======
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of full inference runs (same prompt each time). Default 1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of requests submitted together. In streaming mode this is also the per-wave concurrency.",
    )
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required unless $VOXCPM_MODEL is set")
<<<<<<< HEAD

    if args.jsonl_file:
        if args.prompt_audio or args.prompt_text:
            parser.error("--prompt-audio and --prompt-text are not compatible with --jsonl-file")
    else:
        if (args.prompt_audio is None) != (args.prompt_text is None):
            parser.error("--prompt-audio and --prompt-text must be provided together")

    if args.input:
        if not Path(args.input).exists():
            parser.error(f"Input file not found: {args.input}")

    if args.jsonl_file:
        if not Path(args.jsonl_file).exists():
            parser.error(f"JSONL file not found: {args.jsonl_file}")

    if args.prompt_audio:
        if not Path(args.prompt_audio).exists():
            parser.error(f"Reference audio not found: {args.prompt_audio}")
=======
    if args.txt_prompts is not None and args.jsonl_prompts is not None:
        parser.error("--txt-prompts and --jsonl-prompts are mutually exclusive")
    if (args.ref_audio is None) != (args.ref_text is None):
        parser.error("--ref-audio and --ref-text must be provided together")
    if args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.output_dir is None:
        args.output_dir = "output_audio_streaming" if _is_streaming_stage_config(args.stage_configs_path) else "output_audio"
    try:
        args.prompt_specs = _load_prompt_specs(args)
    except ValueError as exc:
        parser.error(str(exc))
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d

    return args


<<<<<<< HEAD
def main(args) -> None:
    from vllm_omni import Omni

    if args.text:
        output_dir = Path(args.output).parent if args.output else Path("output_audio")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Stage config: {args.stage_configs_path}")
    print(f"Output directory: {output_dir}")

    if args.text:
        if args.prompt_audio:
            print(f"Mode: Voice cloning")
            inputs = _build_clone_input(args)
        else:
            print(f"Mode: Single text synthesis")
            inputs = _build_synthesize_input(args)
    elif args.jsonl_file:
        print(f"Mode: Batch processing from JSONL file")
        inputs = _build_batch_input(args)
    else:
        print(f"Mode: Batch processing from text file")
        if args.prompt_audio:
            print(f"Voice cloning: enabled")
        inputs = _build_batch_input(args)

    print(f"Total requests: {len(inputs)}")
=======
def _is_streaming_stage_config(stage_configs_path: str) -> bool:
    cfg_name = Path(stage_configs_path).name.lower()
    # Keep routing purely config-path based:
    # - voxcpm_no_async_chunk.yaml => sync
    # - others (e.g., voxcpm.yaml with async_chunk) => streaming
    return "no_async_chunk" not in cfg_name


async def _run_streaming_single(
    omni: AsyncOmni,
    args: Any,
    spec: PromptSpec,
    output_dir: Path,
    request_id: str,
    *,
    run_index: int,
    num_runs: int,
    prompt_index: int,
    prompt_count: int,
) -> Path:
    prompt = _build_prompt_for_spec(args, spec, global_request_id=request_id)
    delta_chunks: list[torch.Tensor] = []
    sample_rate = 24000
    chunk_i = 0
    prev_total_samples = 0
    t_start = time.perf_counter()

    if run_index == 0 and prompt_index == 0:
        print(f"---prompt---:{prompt}")

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
            logger.info(
                "run=%d/%d prompt=%d/%d chunk=%d delta_samples=%d buf_len=%d finished=%s",
                run_index + 1,
                num_runs,
                prompt_index + 1,
                prompt_count,
                chunk_i,
                int(delta.numel()),
                n,
                stage_output.finished,
            )
            chunk_i += 1
        except ValueError:
            if not stage_output.finished:
                logger.debug("skip non-audio partial output chunk=%d", chunk_i)

    if not delta_chunks:
        raise RuntimeError("No audio chunks received; check stage config and logs.")

    audio_cat = torch.cat([c.reshape(-1) for c in delta_chunks], dim=0)
    output_path = output_dir / f"output_run{run_index + 1}_{spec.label}.wav"
    sf.write(output_path, audio_cat.numpy(), sample_rate, format="WAV")
    elapsed = time.perf_counter() - t_start
    print(
        f"Saved (streaming) run {run_index + 1}/{num_runs}, "
        f"prompt {prompt_index + 1}/{prompt_count}: {output_path} ({elapsed:.2f}s)"
    )
    return output_path


async def _run_streaming(args) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    omni = AsyncOmni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    t_total = time.perf_counter()
    paths: list[Path] = []
    prompt_specs: list[PromptSpec] = args.prompt_specs
    for run in range(args.num_runs):
        for batch_start, batch_specs in _iter_batches(prompt_specs, args.batch_size):
            tasks = []
            for batch_offset, spec in enumerate(batch_specs):
                prompt_index = batch_start + batch_offset
                request_id = f"stream_{run + 1}_{spec.label}_{uuid.uuid4().hex[:8]}"
                tasks.append(
                    _run_streaming_single(
                        omni,
                        args,
                        spec,
                        output_dir,
                        request_id,
                        run_index=run,
                        num_runs=args.num_runs,
                        prompt_index=prompt_index,
                        prompt_count=len(prompt_specs),
                    )
                )
            paths.extend(await asyncio.gather(*tasks))
    total_elapsed = time.perf_counter() - t_total
    print(
        f"All streaming runs finished: {args.num_runs} run(s), "
        f"{len(prompt_specs)} prompt(s), {len(paths)} file(s) in {total_elapsed:.2f}s total"
    )
    return paths


def _run_sync(args) -> list[Path]:
    output_dir = Path(args.output_dir)
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

<<<<<<< HEAD
    t_start = time.perf_counter()
    saved_count = 0

    for i, prompt in enumerate(inputs, 1):
        try:
            print(f"\nProcessing {i}/{len(inputs)}...")
            for stage_outputs in omni.generate([prompt]):
                output = stage_outputs.request_output
                _save_wav(output_dir, output.request_id, output.outputs[0].multimodal_output)
                saved_count += 1
        except Exception as e:
            print(f"Failed on {i}: {e}")

    elapsed = time.perf_counter() - t_start

    print(f"\n{'='*60}")
    print(f"Generation finished in {elapsed:.2f}s")
    print(f"Success: {saved_count}/{len(inputs)}")
    print(f"Average time per sample: {elapsed/max(len(inputs), 1):.2f}s")
    print(f"{'='*60}")
=======
    t_total = time.perf_counter()
    saved_paths: list[Path] = []
    prompt_specs: list[PromptSpec] = args.prompt_specs
    for run in range(args.num_runs):
        t_run = time.perf_counter()
        run_paths: list[Path] = []
        for batch_index, (batch_start, batch_specs) in enumerate(_iter_batches(prompt_specs, args.batch_size), start=1):
            prompts = []
            for batch_offset, spec in enumerate(batch_specs):
                prompt_index = batch_start + batch_offset
                global_request_id = f"sync_run{run + 1}_{spec.label}_{prompt_index + 1:03d}"
                prompts.append(_build_prompt_for_spec(args, spec, global_request_id=global_request_id))
            if run == 0 and batch_index == 1 and prompts:
                print(f"---prompt---:{prompts[0]}")

            batch_paths: list[Path] = []
            for stage_outputs in omni.generate(prompts):
                request_output = stage_outputs.request_output
                if request_output is None:
                    continue
                item_label = _resolve_output_label(request_output.request_id, batch_specs)
                for j, mm in enumerate(_iter_request_multimodal_outputs(request_output)):
                    save_stem = (
                        f"run{run + 1}_batch{batch_index}_{item_label}"
                        if j == 0
                        else f"run{run + 1}_batch{batch_index}_{item_label}_{j}"
                    )
                    batch_paths.append(_save_wav(mm, output_dir, save_stem))
            if not batch_paths:
                raise RuntimeError("No output from Omni.generate")
            run_paths.extend(batch_paths)
            print(f"Saved (sync) run {run + 1}/{args.num_runs}, batch {batch_index}: {len(batch_paths)} file(s)")

        saved_paths.extend(run_paths)
        print(f"Run {run + 1}/{args.num_runs} finished: {len(run_paths)} file(s) ({time.perf_counter() - t_run:.2f}s)")
        for path in run_paths:
            print(f"  {path}")

    total_elapsed = time.perf_counter() - t_total
    print(
        f"All sync runs finished: {args.num_runs} run(s), "
        f"{len(prompt_specs)} prompt(s), {len(saved_paths)} file(s) in {total_elapsed:.2f}s total"
    )
    return saved_paths


def main(args) -> None:
    logging.basicConfig(level=logging.INFO)
    is_streaming = _is_streaming_stage_config(args.stage_configs_path)
    voice_clone_count = _count_voice_clone_prompts(args.prompt_specs)
    print(f"Model: {args.model}")
    print(f"Stage config: {args.stage_configs_path}")
    print(f"Route: {'streaming' if is_streaming else 'sync'} (from stage-configs-path)")
    print(f"Prompt count: {len(args.prompt_specs)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Voice cloning prompts: {voice_clone_count}/{len(args.prompt_specs)}")
    print(f"Num runs: {args.num_runs}")
    if is_streaming:
        asyncio.run(_run_streaming(args))
    else:
        _run_sync(args)
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d


if __name__ == "__main__":
    main(parse_args())
