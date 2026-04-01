"""Offline VoxCPM inference example for vLLM Omni.

Supports both:
- sync one-shot (Omni.generate)
- streaming (AsyncOmni.generate with async_chunk config)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
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


def _build_prompt(args, *, global_request_id: str | None = None) -> dict[str, Any]:
    additional_information: dict[str, list[Any]] = {
        "text": [args.text],
        "cfg_value": [args.cfg_value],
        "inference_timesteps": [args.inference_timesteps],
        "min_len": [args.min_len],
        "max_new_tokens": [args.max_new_tokens],
    }
    if args.streaming_prefix_len is not None:
        additional_information["streaming_prefix_len"] = [args.streaming_prefix_len]

    if args.ref_audio:
        additional_information["ref_audio"] = [args.ref_audio]
    if args.ref_text:
        additional_information["ref_text"] = [args.ref_text]
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


def parse_args():
    parser = FlexibleArgumentParser(
        description="Offline split-stage VoxCPM inference with vLLM Omni (auto sync/streaming by stage config)"
    )
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
        default=str(DEFAULT_STAGE_SYNC),
        help="Stage config YAML path. Routing is selected only from this path.",
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
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of full inference runs (same prompt each time). Default 1.",
    )
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required unless $VOXCPM_MODEL is set")
    if (args.ref_audio is None) != (args.ref_text is None):
        parser.error("--ref-audio and --ref-text must be provided together")
    if args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    if args.output_dir is None:
        args.output_dir = "output_audio_streaming" if _is_streaming_stage_config(args.stage_configs_path) else "output_audio"

    return args


def _is_streaming_stage_config(stage_configs_path: str) -> bool:
    cfg_name = Path(stage_configs_path).name.lower()
    # Keep routing purely config-path based:
    # - voxcpm_no_async_chunk.yaml => sync
    # - others (e.g., voxcpm.yaml with async_chunk) => streaming
    return "no_async_chunk" not in cfg_name


async def _run_streaming_single(
    omni: AsyncOmni,
    args: Any,
    output_dir: Path,
    request_id: str,
    *,
    run_index: int,
    num_runs: int,
) -> Path:
    prompt = _build_prompt(args, global_request_id=request_id)
    delta_chunks: list[torch.Tensor] = []
    sample_rate = 24000
    chunk_i = 0
    prev_total_samples = 0
    t_start = time.perf_counter()

    if run_index == 0:
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
                "run=%d/%d chunk=%d delta_samples=%d buf_len=%d finished=%s",
                run_index + 1,
                num_runs,
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
    # Include run_index so multi-run output names are always distinct from any engine ids.
    output_path = output_dir / f"output_run{run_index + 1}_{request_id}.wav"
    sf.write(output_path, audio_cat.numpy(), sample_rate, format="WAV")
    elapsed = time.perf_counter() - t_start
    print(f"Saved (streaming) run {run_index + 1}/{num_runs}: {output_path} ({elapsed:.2f}s)")
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
    for run in range(args.num_runs):
        request_id = f"stream_{uuid.uuid4().hex[:8]}"
        paths.append(
            await _run_streaming_single(
                omni,
                args,
                output_dir,
                request_id,
                run_index=run,
                num_runs=args.num_runs,
            )
        )
    total_elapsed = time.perf_counter() - t_total
    print(f"All streaming runs finished: {args.num_runs} file(s) in {total_elapsed:.2f}s total")
    return paths


def _run_sync(args) -> list[Path]:
    output_dir = Path(args.output_dir)

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    t_total = time.perf_counter()
    saved_paths: list[Path] = []
    for run in range(args.num_runs):
        run_tag = f"sync_{uuid.uuid4().hex[:8]}"
        prompt = _build_prompt(args, global_request_id=run_tag)
        t_run = time.perf_counter()
        if run == 0:
            print(f"---prompt---:{prompt}")
        run_paths: list[Path] = []
        for stage_outputs in omni.generate([prompt]):
            # Do not use output.request_id as the only filename stem: vLLM may reuse the same
            # id across sequential generate() calls, which overwrites the WAV.
            for j, output in enumerate(stage_outputs.request_output):
                mm = output.outputs[0].multimodal_output
                save_stem = f"run{run + 1}_{run_tag}" if j == 0 else f"run{run + 1}_{run_tag}_{j}"
                run_paths.append(_save_wav(mm, output_dir, save_stem))
        if not run_paths:
            raise RuntimeError("No output from Omni.generate")
        saved_paths.extend(run_paths)
        print(
            f"Saved (sync) run {run + 1}/{args.num_runs}: "
            f"{len(run_paths)} file(s) ({time.perf_counter() - t_run:.2f}s)"
        )
        for path in run_paths:
            print(f"  {path}")

    total_elapsed = time.perf_counter() - t_total
    print(f"All sync runs finished: {args.num_runs} run(s), {len(saved_paths)} file(s) in {total_elapsed:.2f}s total")
    return saved_paths


def main(args) -> None:
    logging.basicConfig(level=logging.INFO)
    is_streaming = _is_streaming_stage_config(args.stage_configs_path)
    print(f"Model: {args.model}")
    print(f"Stage config: {args.stage_configs_path}")
    print(f"Route: {'streaming' if is_streaming else 'sync'} (from stage-configs-path)")
    print(f"Voice cloning: {'enabled' if args.ref_audio else 'disabled'}")
    print(f"Num runs: {args.num_runs}")
    if is_streaming:
        asyncio.run(_run_streaming(args))
    else:
        _run_sync(args)


if __name__ == "__main__":
    main(parse_args())
