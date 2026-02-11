"""
Stage initialization helpers for vLLM-Omni V1 architecture.

Extracts orchestration-level init logic (config extraction, plugin loading,
multiprocessing setup, device mapping, device locking, engine args building)
out of StageAsyncCoreClient into reusable functions.
"""

from __future__ import annotations

import fcntl
import importlib
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.executor import Executor

from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints.omni_stage import _resolve_worker_cls
from vllm_omni.entrypoints.stage_utils import _to_dict, set_stage_devices
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams

logger = init_logger(__name__)


@dataclass
class StageMetadata:
    """Lightweight stage attributes extracted from stage_config."""

    stage_id: int
    stage_type: Literal["llm", "diffusion"]
    engine_output_type: str | None
    is_comprehension: bool
    requires_multimodal_data: bool
    engine_input_source: list[int]
    final_output: bool
    final_output_type: str | None
    default_sampling_params: OmniSamplingParams
    custom_process_input_func: Callable | None
    model_stage: str | None
    runtime_cfg: Any


def extract_stage_metadata(stage_config: Any) -> StageMetadata:
    """Pure data extraction from a stage_config object."""
    stage_id: int = stage_config.stage_id
    stage_type: Literal["llm", "diffusion"] = getattr(
        stage_config, "stage_type", "llm"
    )
    if stage_type == "diffusion":
        raise NotImplementedError(
            "Diffusion not supported with EngineCore. Use V0 architecture."
        )

    engine_args = stage_config.engine_args
    model_stage = getattr(engine_args, "model_stage", None)
    engine_output_type = getattr(engine_args, "engine_output_type", None)
    is_comprehension = getattr(stage_config, "is_comprehension", False)

    runtime_cfg = getattr(stage_config, "runtime", {})
    requires_multimodal_data = getattr(runtime_cfg, "requires_multimodal_data", False)

    engine_input_source: list[int] = getattr(
        stage_config, "engine_input_source", []
    )
    final_output: bool = getattr(stage_config, "final_output", False)
    final_output_type: str | None = getattr(stage_config, "final_output_type", None)

    default_sp = _to_dict(getattr(stage_config, "default_sampling_params", {}))
    SPClass = (
        SamplingParams if stage_type == "llm" else OmniDiffusionSamplingParams
    )
    default_sampling_params: OmniSamplingParams = SPClass(**default_sp)

    custom_process_input_func: Callable | None = None
    if hasattr(stage_config, "custom_process_input_func"):
        mod_path, fn_name = stage_config.custom_process_input_func.rsplit(".", 1)
        custom_process_input_func = getattr(
            importlib.import_module(mod_path), fn_name
        )

    return StageMetadata(
        stage_id=stage_id,
        stage_type=stage_type,
        engine_output_type=engine_output_type,
        is_comprehension=is_comprehension,
        requires_multimodal_data=requires_multimodal_data,
        engine_input_source=engine_input_source,
        final_output=final_output,
        final_output_type=final_output_type,
        default_sampling_params=default_sampling_params,
        custom_process_input_func=custom_process_input_func,
        model_stage=model_stage,
        runtime_cfg=runtime_cfg,
    )


def prepare_engine_environment() -> None:
    """One-time global setup: load plugins, set multiprocessing spawn method."""
    from vllm_omni.plugins import load_omni_general_plugins

    load_omni_general_plugins()

    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        logger.info("[stage_init] Set VLLM_WORKER_MULTIPROC_METHOD=spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def setup_stage_devices(stage_id: int, runtime_cfg: Any) -> None:
    """Device mapping via set_stage_devices for a single stage."""
    try:
        from vllm_omni.platforms import current_omni_platform

        device_type = current_omni_platform.device_type
        set_stage_devices(
            stage_id,
            runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else None,
            device_type=device_type,
        )
        logger.info(
            "[stage_init] Stage-%s set devices for %s, runtime devices: %s",
            stage_id,
            device_type,
            runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else None,
        )
    except Exception as e:
        logger.warning("Device setup failed for stage %s: %s", stage_id, e)


def build_vllm_config(
    stage_config: Any, model: str
) -> tuple[Any, type]:
    """Build engine_args_dict, resolve worker class, create VllmConfig and executor_class.

    Returns:
        (vllm_config, executor_class)
    """
    engine_args = stage_config.engine_args
    stage_type = getattr(stage_config, "stage_type", "llm")
    stage_id = stage_config.stage_id

    engine_args_dict = _to_dict(engine_args)
    engine_args_dict["model"] = model

    if stage_type != "diffusion":
        _resolve_worker_cls(engine_args_dict)

    logger.info(
        "[stage_init] Stage-%s engine_args_dict: %s", stage_id, engine_args_dict
    )

    omni_engine_args = OmniEngineArgs(**engine_args_dict)
    vllm_config = omni_engine_args.create_engine_config(
        usage_context=UsageContext.LLM_CLASS
    )
    executor_class = Executor.get_class(vllm_config)

    return vllm_config, executor_class


def acquire_device_locks(
    stage_id: int,
    engine_args_dict: dict[str, Any],
    stage_init_timeout: int = 300,
) -> list[int]:
    """Acquire exclusive file locks on devices needed by this stage.

    Returns list of lock file descriptors that must be released after init.
    """
    lock_fds: list[int] = []
    try:
        from vllm_omni.platforms import current_omni_platform

        # Get parallel sizes
        if "parallel_config" in engine_args_dict:
            pc = engine_args_dict["parallel_config"]
            tensor_parallel_size = pc.get("tensor_parallel_size", 1)
            pipeline_parallel_size = pc.get("pipeline_parallel_size", 1)
            data_parallel_size = pc.get("data_parallel_size", 1)
            prefill_context_parallel_size = pc.get(
                "prefill_context_parallel_size", 1
            )
            sequence_parallel_size = pc.get("sequence_parallel_size", 1)
            cfg_parallel_size = pc.get("cfg_parallel_size", 1)
        else:
            tensor_parallel_size = engine_args_dict.get("tensor_parallel_size", 1)
            pipeline_parallel_size = engine_args_dict.get(
                "pipeline_parallel_size", 1
            )
            data_parallel_size = engine_args_dict.get("data_parallel_size", 1)
            prefill_context_parallel_size = engine_args_dict.get(
                "prefill_context_parallel_size", 1
            )
            sequence_parallel_size = 1
            cfg_parallel_size = 1

        num_devices_per_stage = (
            tensor_parallel_size
            * pipeline_parallel_size
            * data_parallel_size
            * prefill_context_parallel_size
            * sequence_parallel_size
            * cfg_parallel_size
        )

        # Get physical device IDs
        device_control_env = current_omni_platform.device_control_env_var
        visible_devices_str = os.environ.get(device_control_env)
        physical_devices: list[int] = []

        if visible_devices_str:
            try:
                physical_devices = [
                    int(x.strip())
                    for x in visible_devices_str.split(",")
                    if x.strip()
                ]
            except (ValueError, IndexError):
                pass

        if not physical_devices:
            num_devices = current_omni_platform.get_device_count()
            physical_devices = list(range(num_devices))

        num_devices_to_lock = min(num_devices_per_stage, len(physical_devices))
        devices_to_lock = sorted(physical_devices[:num_devices_to_lock])

        logger.debug(
            "Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d, SP=%d, CFG=%d; "
            "will lock %d devices: %s",
            tensor_parallel_size,
            pipeline_parallel_size,
            data_parallel_size,
            prefill_context_parallel_size,
            sequence_parallel_size,
            cfg_parallel_size,
            num_devices_to_lock,
            devices_to_lock,
        )

        # Acquire locks
        wait_start = time.time()
        for device_id in devices_to_lock:
            lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
            lock_acquired = False

            while not lock_acquired:
                try:
                    lock_fd = os.open(
                        lock_file, os.O_CREAT | os.O_RDWR, 0o644
                    )
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        os.ftruncate(lock_fd, 0)
                        os.write(lock_fd, f"{os.getpid()}\n".encode())
                        os.fsync(lock_fd)
                        lock_acquired = True
                        lock_fds.append(lock_fd)
                        logger.debug(
                            "Acquired exclusive lock for device %s", device_id
                        )
                    except BlockingIOError:
                        os.close(lock_fd)
                        if time.time() - wait_start > stage_init_timeout:
                            logger.warning(
                                "Timeout waiting for device %s initialization "
                                "lock, proceeding anyway",
                                device_id,
                            )
                            break
                        time.sleep(0.1)
                except OSError as e:
                    logger.debug(
                        "Failed to acquire lock for device %s: %s, "
                        "continuing anyway",
                        device_id,
                        e,
                    )
                    try:
                        os.close(lock_fd)
                    except (OSError, NameError):
                        pass
                    break

    except Exception as e:
        logger.debug(
            "[Stage-%s] Failed to set up sequential initialization lock: %s",
            stage_id,
            e,
        )

    return lock_fds


def release_device_locks(lock_fds: list[int]) -> None:
    """Release file locks acquired by acquire_device_locks."""
    for lock_fd in lock_fds:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            logger.debug("Released initialization lock (fd=%s)", lock_fd)
        except (OSError, ValueError):
            pass
