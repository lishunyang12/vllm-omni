# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Stage Configuration System for vLLM-Omni.

Pipeline structure (stages, types, data-flow) is defined in per-model YAML
files and is set by model developers at integration time.
Runtime parameters (gpu_memory_utilization, tp_size, etc.) come from CLI.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

from vllm_omni.config.yaml_util import create_config, load_yaml_config, to_dict

# Pipeline YAMLs live alongside model code in model_executor/models/<model>/
_MODELS_DIR = Path(__file__).resolve().parent.parent / "model_executor" / "models"


def get_pipeline_path(model_dir: str, filename: str) -> Path:
    """Return the full path to a pipeline YAML file.

    Args:
        model_dir: Model subdirectory name (e.g., "qwen3_omni").
        filename: Name of the YAML file (e.g., "pipeline.yaml").

    Returns:
        Absolute path to the file.
    """
    return _MODELS_DIR / model_dir / filename


logger = init_logger(__name__)


class StageType(str, Enum):
    """Type of processing stage in the Omni pipeline."""

    LLM = "llm"
    DIFFUSION = "diffusion"


class StageExecutionType(str, Enum):
    """Merged StageType + WorkerType (per gcanlin).

    Today there are exactly 3 combinations. Using a single enum
    simplifies scheduler inference and stage-type validation.
    """

    LLM_AR = "llm_ar"  # thinker, talker
    LLM_GENERATION = "llm_generation"  # code2wav
    DIFFUSION = "diffusion"  # dit


# Scheduler inferred from execution type — no config needed.
_EXECUTION_TYPE_TO_SCHEDULER: dict[StageExecutionType, str | None] = {
    StageExecutionType.LLM_AR: (
        "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler"
    ),
    StageExecutionType.LLM_GENERATION: (
        "vllm_omni.core.sched.omni_generation_scheduler"
        ".OmniGenerationScheduler"
    ),
    StageExecutionType.DIFFUSION: None,
}


# ---------------------------------------------------------------------------
# Pipeline Configuration (Layer 1 — immutable, Python-defined)
#
# Immutable topology lives in Python code, not YAML (per alex-jw-brooks,
# wuhang2014). frozen=True — changing topology requires a code change.
# Fields organized into logical groups with comments (per wuhang2014).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StagePipelineConfig:
    """Fixed topology for one stage. Not user-configurable.

    Defined by model developers alongside model code. Changing any field
    requires a code change and review.

    Attributes:
        stage_id: Index of this stage in the pipeline.
        model_stage: Logical name ("thinker", "talker", "code2wav", "dit").
        execution_type: Merged stage + worker type (LLM_AR, LLM_GENERATION,
            DIFFUSION). Determines scheduler via ``_EXECUTION_TYPE_TO_SCHEDULER``.
        input_sources: Upstream stage IDs. Empty tuple = entry point.
        final_output: Whether this stage emits user-visible output.
        final_output_type: Output modality ("text", "audio", "image").
        is_comprehension: Marks the understanding stage (tokenizer owner).
        requires_multimodal_data: Whether this stage needs multimodal input.
        hf_config_name: Sub-config key in HF config (e.g. "thinker_config").
        default_sampling_params: Default sampling parameters for this stage.
            Includes both model-intrinsic constraints (``detokenize``,
            ``stop_token_ids``) and tunable defaults (``temperature``,
            ``top_k``). Deploy/CLI values override the tunable ones.
        output_connectors: Named output connector references
            (e.g. ``{"to_stage_1": "connector_of_shared_memory"}``).
        input_connectors: Named input connector references
            (e.g. ``{"from_stage_0": "connector_of_shared_memory"}``).
        custom_process_input_func: Sync input transform function path.
            Part of fixed pipeline topology (not deployment-dependent).
        prompt_expand_func: CFG prompt expansion (Bagel).
        cfg_kv_collect_func: CFG KV cache collection (Bagel).
        omni_kv_config: KV cache transfer config (Bagel, Hunyuan — fixed
            per model, promoted from extras).
        extras: Escape hatch for model-specific fields not yet promoted
            to explicit attributes.

    Note:
        ``engine_output_type`` and ``custom_process_next_stage_input_func``
        are deployment-dependent and live in ``StageDeployConfig``, not here.
    """

    # --- Identity ---
    stage_id: int
    model_stage: str

    # --- Execution ---
    execution_type: StageExecutionType = StageExecutionType.LLM_AR

    # --- DAG topology ---
    input_sources: tuple[int, ...] = ()

    # --- Output ---
    final_output: bool = False
    final_output_type: str | None = None

    # --- Model-intrinsic properties ---
    is_comprehension: bool = False
    requires_multimodal_data: bool = False
    hf_config_name: str | None = None

    # --- Default sampling params (model defaults, overridable by deploy/CLI) ---
    default_sampling_params: dict[str, Any] = field(default_factory=dict)

    # --- Connectors (per-stage references to named connector definitions) ---
    output_connectors: dict[str, str] | None = None
    input_connectors: dict[str, str] | None = None

    # --- Processing hooks (dotted Python paths, sync/topology-fixed only) ---
    custom_process_input_func: str | None = None
    prompt_expand_func: str | None = None
    cfg_kv_collect_func: str | None = None

    # --- Model-specific ---
    omni_kv_config: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline topology for a model. One per model type.

    Defined by model developers alongside model code.
    ``frozen=True`` — changing topology requires a code change and review.

    Attributes:
        model_type: HF model type identifier (e.g. "qwen3_omni_moe").
        async_chunk: Whether async chunk streaming is enabled between stages.
            When true, the engine uses ``custom_process_next_stage_input_func``
            (from deploy config) and activates connectors.
            When false, uses ``custom_process_input_func`` (sync path).
        stages: Ordered tuple of stage configs defining the DAG.
        connectors: Named connector definitions (backend + config).
            Activated when ``async_chunk`` is enabled.
        edges: Edge topology (from, to, window_size).

    Note:
        ``model_arch`` lives in ``StageDeployConfig.engine_args``, not here,
        because it is an engine-level setting overridable by CLI.
    """

    model_type: str
    async_chunk: bool = False
    stages: tuple[StagePipelineConfig, ...] = ()

    # --- Distributed configuration (activated when async_chunk is enabled) ---
    connectors: dict[str, Any] | None = None
    edges: list[dict[str, Any]] | None = None

    def get_stage(self, stage_id: int) -> StagePipelineConfig | None:
        """Look up a stage by its ID."""
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def get_scheduler_cls(self, stage_id: int) -> str | None:
        """Return the inferred scheduler class path for a stage."""
        stage = self.get_stage(stage_id)
        if stage is None:
            return None
        return _EXECUTION_TYPE_TO_SCHEDULER.get(stage.execution_type)

    def validate(self) -> list[str]:
        """Validate pipeline topology.

        Returns:
            List of error messages. Empty if valid.
        """
        errors: list[str] = []
        if not self.stages:
            errors.append("Pipeline has no stages defined")
            return errors
        stage_ids = [s.stage_id for s in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            errors.append("Duplicate stage IDs found")
        stage_id_set = set(stage_ids)
        for stage in self.stages:
            for src in stage.input_sources:
                if src not in stage_id_set:
                    errors.append(
                        f"Stage {stage.stage_id} references "
                        f"non-existent input source {src}"
                    )
                if src == stage.stage_id:
                    errors.append(f"Stage {stage.stage_id} references itself")
        if not any(not s.input_sources for s in self.stages):
            errors.append("No entry point (stage with empty input_sources)")
        return errors


# Pipeline registry — model_type → PipelineConfig
_PIPELINE_REGISTRY: dict[str, PipelineConfig] = {}


def register_pipeline(pipeline: PipelineConfig) -> None:
    """Register a pipeline config for a model type.

    Called at import time by pipeline.py modules under each model directory.
    """
    errors = pipeline.validate()
    if errors:
        logger.warning("Pipeline %s has issues: %s", pipeline.model_type, errors)
    _PIPELINE_REGISTRY[pipeline.model_type] = pipeline


# ---------------------------------------------------------------------------
# Deploy Configuration (Layer 2 — user-tunable, YAML)
#
# The only config file users edit. Loaded from deploy/<model>.yaml.
# All fields overridable by CLI.
# ---------------------------------------------------------------------------

# Deploy YAMLs live in vllm_omni/deploy/
_DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"


@dataclass
class StageDeployConfig:
    """Per-stage deployment knobs. All overridable by CLI.

    Maps to the ``engine_args:`` + ``runtime:`` sections in deploy YAML.

    Attributes:
        stage_id: Stage index (must match PipelineConfig stage_id).
        model_arch: HF architecture class name.
        engine_output_type: Engine output type ("latent", "text", "audio",
            "image"). Deployment-dependent because the same pipeline stage
            may produce different output types depending on deploy mode.
        custom_process_next_stage_input_func: Async chunk input transform
            function path. Lives in deploy (not pipeline) because it is
            only used when ``async_chunk`` is enabled.
        max_num_seqs: Max concurrent sequences.
        gpu_memory_utilization: GPU memory fraction.
        tensor_parallel_size: TP degree.
        enforce_eager: Disable CUDA graphs.
        trust_remote_code: Allow HF remote code.
        enable_prefix_caching: KV cache reuse.
        enable_chunked_prefill: Chunked prefill.
        max_num_batched_tokens: Token budget per iteration.
        max_model_len: Max context length.
        distributed_executor_backend: "mp" or "ray".
        async_scheduling: Async scheduling.
        quantization: Quantization method.
        dtype: Model precision.
        data_parallel_size: DP degree.
        pipeline_parallel_size: PP degree.
        devices: GPU device assignment.
        sampling_params: User-tunable sampling overrides (merged on top of
            ``StagePipelineConfig.default_sampling_params``).
        engine_extras: Escape hatch for model-specific engine args.
    """

    stage_id: int

    # --- Engine args ---
    model_arch: str | None = None
    engine_output_type: str | None = None
    custom_process_next_stage_input_func: str | None = None
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    trust_remote_code: bool = True
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool | None = None
    max_num_batched_tokens: int = 32768
    max_model_len: int | None = None
    distributed_executor_backend: str = "mp"
    async_scheduling: bool | None = None
    quantization: str | None = None
    dtype: str | None = None
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # --- Runtime ---
    devices: str = "0"

    # --- User-tunable sampling overrides ---
    sampling_params: dict[str, Any] | None = None

    # --- Escape hatch for model-specific engine args ---
    engine_extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeployConfig:
    """Loaded from ``deploy/<model>.yaml``. The only YAML file users edit.

    Contains per-stage engine args and platform overrides. Connector
    definitions and edges live in PipelineConfig (topology, not deployment).
    ``async_chunk`` lives in PipelineConfig (topology, not deployment).

    Attributes:
        stages: Per-stage deployment knobs.
        platforms: Platform delta overrides (npu, rocm, xpu) — NOT full copies.
    """

    stages: list[StageDeployConfig] = field(default_factory=list)
    platforms: dict[str, Any] | None = None


def load_deploy_config(path: str | Path) -> DeployConfig:
    """Load a deploy YAML file and return a DeployConfig dataclass.

    Handles the ``engine_args:`` / ``runtime:`` nesting per stage,
    flattening into ``StageDeployConfig`` fields.

    Args:
        path: Path to the deploy YAML file.

    Returns:
        DeployConfig with parsed stages and platform overrides.
    """
    raw = load_yaml_config(path)
    raw_dict = to_dict(raw)

    stages: list[StageDeployConfig] = []
    for stage_data in raw_dict.get("stages", []):
        stage_id = stage_data["stage_id"]

        # Flatten engine_args + runtime into StageDeployConfig fields
        engine_args = stage_data.get("engine_args", {})
        runtime = stage_data.get("runtime", {})

        # Pop known fields from engine_args
        sdc = StageDeployConfig(
            stage_id=stage_id,
            model_arch=engine_args.pop("model_arch", None),
            engine_output_type=engine_args.pop("engine_output_type", None),
            custom_process_next_stage_input_func=engine_args.pop(
                "custom_process_next_stage_input_func", None
            ),
            max_num_seqs=engine_args.pop("max_num_seqs", 64),
            gpu_memory_utilization=engine_args.pop("gpu_memory_utilization", 0.9),
            tensor_parallel_size=engine_args.pop("tensor_parallel_size", 1),
            enforce_eager=engine_args.pop("enforce_eager", False),
            trust_remote_code=engine_args.pop("trust_remote_code", True),
            enable_prefix_caching=engine_args.pop("enable_prefix_caching", False),
            enable_chunked_prefill=engine_args.pop("enable_chunked_prefill", None),
            max_num_batched_tokens=engine_args.pop("max_num_batched_tokens", 32768),
            max_model_len=engine_args.pop("max_model_len", None),
            distributed_executor_backend=engine_args.pop(
                "distributed_executor_backend", "mp"
            ),
            async_scheduling=engine_args.pop("async_scheduling", None),
            quantization=engine_args.pop("quantization", None),
            dtype=engine_args.pop("dtype", None),
            data_parallel_size=engine_args.pop("data_parallel_size", 1),
            pipeline_parallel_size=engine_args.pop("pipeline_parallel_size", 1),
            devices=runtime.pop("devices", "0"),
            sampling_params=stage_data.get("sampling_params", None),
            # Remaining engine_args go into escape hatch
            engine_extras=engine_args,
        )
        stages.append(sdc)

    return DeployConfig(
        stages=stages,
        platforms=raw_dict.get("platforms", None),
    )


def _detect_platform() -> str | None:
    """Detect the current hardware platform.

    Returns:
        Platform name ("npu", "rocm", "xpu") or None for CUDA default.
    """
    try:
        from vllm.platforms import current_platform

        name = current_platform.device_name.lower()
        if "npu" in name:
            return "npu"
        if "rocm" in name or "amd" in name:
            return "rocm"
        if "xpu" in name:
            return "xpu"
    except Exception:
        pass
    return None


def _apply_platform_overrides(
    deploy: DeployConfig,
    platform: str | None = None,
) -> DeployConfig:
    """Merge platform-specific overrides into the deploy config.

    Only overrides fields that the platform section explicitly sets.

    Args:
        deploy: Base deploy config (CUDA defaults).
        platform: Platform name, or None to auto-detect.

    Returns:
        DeployConfig with platform overrides applied.
    """
    if platform is None:
        platform = _detect_platform()
    if platform is None or deploy.platforms is None:
        return deploy
    platform_section = deploy.platforms.get(platform)
    if platform_section is None:
        return deploy

    platform_stages = platform_section.get("stages", [])
    # Index base stages by stage_id for O(1) lookup
    base_by_id = {s.stage_id: s for s in deploy.stages}

    for ps in platform_stages:
        sid = ps["stage_id"]
        base = base_by_id.get(sid)
        if base is None:
            continue
        ea = ps.get("engine_args", {})
        rt = ps.get("runtime", {})
        # Override matching fields
        for key, val in ea.items():
            if hasattr(base, key):
                object.__setattr__(base, key, val)
            else:
                base.engine_extras[key] = val
        if "devices" in rt:
            object.__setattr__(base, "devices", rt["devices"])

    return deploy


def merge_pipeline_deploy(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    cli_overrides: dict[str, Any] | None = None,
) -> list[StageConfig]:
    """Merge PipelineConfig + DeployConfig + CLI overrides into StageConfigs.

    This is the core wiring function that replaces legacy YAML loading
    for models registered in ``_PIPELINE_REGISTRY``.

    Override precedence:
        pipeline (fixed topology) → deploy (CUDA defaults) →
        platform overrides (auto-detected) → CLI args

    Args:
        pipeline: Frozen pipeline topology from registry.
        deploy: Deploy config loaded from YAML.
        cli_overrides: CLI arguments from VllmConfig.

    Returns:
        List of StageConfig objects ready for the engine.
    """
    if cli_overrides is None:
        cli_overrides = {}

    # Apply platform overrides
    deploy = _apply_platform_overrides(deploy)

    # Index deploy stages by stage_id
    deploy_by_id = {s.stage_id: s for s in deploy.stages}

    result: list[StageConfig] = []
    for ps in pipeline.stages:
        ds = deploy_by_id.get(ps.stage_id)

        # --- Derive legacy fields from execution_type ---
        if ps.execution_type == StageExecutionType.LLM_AR:
            stage_type = StageType.LLM
            worker_type = "ar"
        elif ps.execution_type == StageExecutionType.LLM_GENERATION:
            stage_type = StageType.LLM
            worker_type = "generation"
        elif ps.execution_type == StageExecutionType.DIFFUSION:
            stage_type = StageType.DIFFUSION
            worker_type = None
        else:
            stage_type = StageType.LLM
            worker_type = None

        scheduler_cls = _EXECUTION_TYPE_TO_SCHEDULER.get(ps.execution_type)

        # --- Build engine_args from deploy config ---
        yaml_engine_args: dict[str, Any] = {}
        if ds is not None:
            if ds.model_arch:
                yaml_engine_args["model_arch"] = ds.model_arch
            if ds.engine_output_type:
                yaml_engine_args["engine_output_type"] = ds.engine_output_type
            yaml_engine_args["max_num_seqs"] = ds.max_num_seqs
            yaml_engine_args["gpu_memory_utilization"] = ds.gpu_memory_utilization
            yaml_engine_args["tensor_parallel_size"] = ds.tensor_parallel_size
            yaml_engine_args["enforce_eager"] = ds.enforce_eager
            yaml_engine_args["trust_remote_code"] = ds.trust_remote_code
            yaml_engine_args["enable_prefix_caching"] = ds.enable_prefix_caching
            yaml_engine_args["max_num_batched_tokens"] = ds.max_num_batched_tokens
            yaml_engine_args["distributed_executor_backend"] = (
                ds.distributed_executor_backend
            )
            if ds.enable_chunked_prefill is not None:
                yaml_engine_args["enable_chunked_prefill"] = ds.enable_chunked_prefill
            if ds.max_model_len is not None:
                yaml_engine_args["max_model_len"] = ds.max_model_len
            if ds.async_scheduling is not None:
                yaml_engine_args["async_scheduling"] = ds.async_scheduling
            if ds.quantization is not None:
                yaml_engine_args["quantization"] = ds.quantization
            if ds.dtype is not None:
                yaml_engine_args["dtype"] = ds.dtype
            if ds.data_parallel_size != 1:
                yaml_engine_args["data_parallel_size"] = ds.data_parallel_size
            if ds.pipeline_parallel_size != 1:
                yaml_engine_args["pipeline_parallel_size"] = ds.pipeline_parallel_size
            # Async chunk hook from deploy
            if ds.custom_process_next_stage_input_func:
                yaml_engine_args["custom_process_next_stage_input_func"] = (
                    ds.custom_process_next_stage_input_func
                )
            # Engine extras (escape hatch)
            yaml_engine_args.update(ds.engine_extras)

        # Inject async_chunk into engine_args (legacy compat)
        if pipeline.async_chunk:
            yaml_engine_args["async_chunk"] = True

        # --- Build runtime from deploy ---
        yaml_runtime: dict[str, Any] = {"process": True}
        if ds is not None:
            yaml_runtime["devices"] = ds.devices

        # --- Determine custom_process_input_func ---
        # Async mode: use custom_process_next_stage_input_func from deploy
        # Sync mode: use custom_process_input_func from pipeline
        custom_process_input_func = ps.custom_process_input_func

        # --- Build yaml_extras (default_sampling_params, connectors, etc.) ---
        yaml_extras: dict[str, Any] = {}

        # Merge sampling params: pipeline defaults ← deploy overrides
        sampling = dict(ps.default_sampling_params)
        if ds is not None and ds.sampling_params:
            sampling.update(ds.sampling_params)
        if sampling:
            yaml_extras["default_sampling_params"] = sampling

        # Connectors
        if ps.output_connectors:
            yaml_extras["output_connectors"] = dict(ps.output_connectors)
        if ps.input_connectors:
            yaml_extras["input_connectors"] = dict(ps.input_connectors)

        # Model-specific extras from pipeline
        if ps.extras:
            yaml_extras.update(ps.extras)

        stage = StageConfig(
            stage_id=ps.stage_id,
            model_stage=ps.model_stage,
            stage_type=stage_type,
            input_sources=list(ps.input_sources),
            custom_process_input_func=custom_process_input_func,
            final_output=ps.final_output,
            final_output_type=ps.final_output_type,
            worker_type=worker_type,
            scheduler_cls=scheduler_cls,
            hf_config_name=ps.hf_config_name,
            is_comprehension=ps.is_comprehension,
            yaml_engine_args=yaml_engine_args,
            yaml_runtime=yaml_runtime,
            yaml_extras=yaml_extras,
        )
        result.append(stage)

    return result


# ---------------------------------------------------------------------------
# Stage Config (mutable, parsed from pipeline YAML — legacy path)
# ---------------------------------------------------------------------------


@dataclass
class StageConfig:
    """Per-stage configuration from pipeline YAML.

    Topology fields (stage_id, input_sources, etc.) define the DAG.
    Engine and runtime defaults come from the YAML; CLI overrides take
    precedence via ``runtime_overrides``.
    """

    # Identity
    stage_id: int
    model_stage: str

    # Stage type
    stage_type: StageType = StageType.LLM

    input_sources: list[int] = field(default_factory=list)
    custom_process_input_func: str | None = None
    final_output: bool = False
    final_output_type: str | None = None  # "text", "audio", "image"
    worker_type: str | None = None  # "ar" or "generation"
    scheduler_cls: str | None = None
    hf_config_name: str | None = None
    is_comprehension: bool = False

    # Per-stage engine args from pipeline YAML (defaults)
    yaml_engine_args: dict[str, Any] = field(default_factory=dict)
    # Per-stage runtime config from pipeline YAML (devices, etc.)
    yaml_runtime: dict[str, Any] = field(default_factory=dict)
    # Pass-through fields from pipeline YAML (default_sampling_params,
    # output_connectors, input_connectors, tts_args, etc.)
    yaml_extras: dict[str, Any] = field(default_factory=dict)

    # Runtime overrides (populated from CLI, not from pipeline YAML)
    runtime_overrides: dict[str, Any] = field(default_factory=dict)

    def to_omegaconf(self) -> Any:
        """Convert to OmegaConf for backward compatibility with OmniStage.

        Returns:
            OmegaConf DictConfig with stage configuration in legacy format.
        """
        # Start with YAML engine_args defaults
        engine_args: dict[str, Any] = dict(self.yaml_engine_args)

        # Overlay topology-level fields
        engine_args["model_stage"] = self.model_stage
        if self.worker_type:
            engine_args["worker_type"] = self.worker_type
        if self.scheduler_cls:
            engine_args["scheduler_cls"] = self.scheduler_cls
        if self.hf_config_name:
            engine_args["hf_config_name"] = self.hf_config_name

        # CLI overrides take precedence over YAML defaults
        for key, value in self.runtime_overrides.items():
            if key not in ("devices", "max_batch_size"):
                engine_args[key] = value

        # Build runtime config from YAML defaults + CLI overrides
        runtime: dict[str, Any] = dict(self.yaml_runtime)
        runtime.setdefault("process", True)
        if "devices" in self.runtime_overrides:
            runtime["devices"] = self.runtime_overrides["devices"]

        # Legacy compat: migrate runtime.max_batch_size → engine_args.max_num_seqs
        legacy_mbs = runtime.pop("max_batch_size", None)
        cli_mbs = self.runtime_overrides.get("max_batch_size")
        if legacy_mbs is not None or cli_mbs is not None:
            warnings.warn(
                "runtime.max_batch_size is deprecated and will be removed in a "
                "future release. Use engine_args.max_num_seqs instead.",
                FutureWarning,
                stacklevel=2,
            )
            effective_mbs = int(cli_mbs or legacy_mbs or 1)
            engine_args.setdefault("max_num_seqs", effective_mbs)

        engine_args.setdefault("max_num_seqs", 1)

        # Build full config dict
        config_dict: dict[str, Any] = {
            "stage_id": self.stage_id,
            "stage_type": StageType(self.stage_type).value,
            "engine_args": create_config(engine_args),
            "runtime": create_config(runtime),
            "engine_input_source": self.input_sources,  # Legacy field name
            "final_output": self.final_output,
            "final_output_type": self.final_output_type,
            "is_comprehension": self.is_comprehension,
        }

        if self.custom_process_input_func:
            config_dict["custom_process_input_func"] = self.custom_process_input_func

        # Pass through extra YAML fields (default_sampling_params,
        # output_connectors, input_connectors, tts_args, etc.)
        config_dict.update(self.yaml_extras)

        return create_config(config_dict)


@dataclass
class ModelPipeline:
    """Complete pipeline definition for a multi-stage model.

    Defined by model developers, bundled with the model, not user-editable.
    """

    model_type: str
    stages: list[StageConfig]

    # Pipeline-wide behavior flags
    async_chunk: bool = False

    # Optional distributed configuration
    connectors: dict[str, Any] | None = None
    edges: list[dict[str, Any]] | None = None

    def get_stage(self, stage_id: int) -> StageConfig | None:
        """Look up a stage by its ID.

        Args:
            stage_id: The stage ID to search for.

        Returns:
            The matching StageConfig, or None if not found.
        """
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def validate_pipeline(self) -> list[str]:
        """Validate pipeline topology at model integration time (not runtime).

        Checks:
        - All stage IDs are unique
        - All input_sources reference valid stage IDs
        - At least one entry point (stage with empty input_sources)

        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors: list[str] = []

        if not self.stages:
            errors.append("Topology has no stages defined")
            return errors

        # Check for unique stage IDs
        stage_ids = [s.stage_id for s in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            errors.append("Duplicate stage IDs found")

        stage_id_set = set(stage_ids)

        # Check input_sources reference valid stages
        for stage in self.stages:
            for source_id in stage.input_sources:
                if source_id not in stage_id_set:
                    errors.append(f"Stage {stage.stage_id} references non-existent input source {source_id}")
                if source_id == stage.stage_id:
                    errors.append(f"Stage {stage.stage_id} references itself as input source")

        # Check for at least one entry point
        entry_points = [s for s in self.stages if not s.input_sources]
        if not entry_points:
            errors.append("No entry point found (stage with empty input_sources)")

        return errors


class StageConfigFactory:
    """Factory that loads pipeline YAML and merges CLI overrides.

    Handles both single-stage and multi-stage models.
    """

    # Mapping of model types to directories under model_executor/models/.
    PIPELINE_MODELS: dict[str, str] = {
        "qwen3_omni_moe": "qwen3_omni",
        "qwen2_5_omni": "qwen2_5_omni",
        "bagel": "bagel",
        "qwen3_tts": "qwen3_tts",
        "voxtral_tts": "voxtral_tts",
        "mimo_audio": "mimo_audio",
        "glm-image": "glm_image",
        "cosyvoice3": "cosyvoice3",
        "mammothmoda2": "mammoth_moda2",
    }

    # Fallback: map HF architecture class names to pipeline dirs.
    # Used when model_type collides with another model (e.g. MiMo Audio
    # reports model_type="qwen2" which matches plain Qwen2, not our pipeline).
    _ARCHITECTURE_MODELS: dict[str, str] = {
        "MiMoAudioForConditionalGeneration": "mimo_audio",
        "HunyuanImage3ForCausalMM": "hunyuan_image3",
    }

    @classmethod
    def create_from_model(
        cls,
        model: str,
        cli_overrides: dict[str, Any] | None = None,
        deploy_config_path: str | None = None,
    ) -> list[StageConfig] | None:
        """Load pipeline config, merge with deploy + CLI overrides.

        For models registered in ``_PIPELINE_REGISTRY`` (new path):
        reads PipelineConfig from registry + DeployConfig from YAML,
        merges them via ``merge_pipeline_deploy()``.

        For other models (legacy path): loads pipeline YAML from
        ``model_executor/models/<model>/pipeline.yaml``.

        Args:
            model: Model name or path.
            cli_overrides: CLI overrides from VllmConfig/OmniDiffusionConfig.
            deploy_config_path: Optional path to a deploy YAML file.
                If None, auto-resolved from ``deploy/<model_type>.yaml``.

        Returns:
            List of StageConfig objects with CLI overrides applied,
            or None if no pipeline definition was found for this model.
        """
        if cli_overrides is None:
            cli_overrides = {}

        trust_remote_code = cli_overrides.get("trust_remote_code", True)

        # --- New path: check pipeline registry first ---
        model_type, _ = cls._auto_detect_model_type(
            model, trust_remote_code=trust_remote_code
        )
        if model_type and model_type in _PIPELINE_REGISTRY:
            return cls._create_from_registry(
                model_type, cli_overrides, deploy_config_path
            )

        # --- Legacy path: load from pipeline YAML ---
        pipeline = cls._load_pipeline(model, trust_remote_code=trust_remote_code)

        if pipeline is None:
            return None

        errors = pipeline.validate_pipeline()
        if errors:
            logger.warning(f"Pipeline validation warnings for {model}: {errors}")

        # Inject pipeline-wide async_chunk into ALL stages' engine_args.
        # The legacy loader (load_stage_configs_from_yaml) sets async_chunk
        # on every stage so that build_engine_args_dict() can inject the
        # stage_connector_spec.  AsyncOmniEngine.__init__ also reads it
        # from stage_configs[0].engine_args.async_chunk.
        if pipeline.async_chunk:
            for stage in pipeline.stages:
                stage.yaml_engine_args.setdefault("async_chunk", True)

        # Apply CLI overrides
        result: list[StageConfig] = []
        for stage in pipeline.stages:
            # Merge global CLI overrides
            stage.runtime_overrides = cls._merge_cli_overrides(stage, cli_overrides)
            result.append(stage)

        return result

    @classmethod
    def _create_from_registry(
        cls,
        model_type: str,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None = None,
    ) -> list[StageConfig]:
        """Create StageConfigs from pipeline registry + deploy YAML.

        Args:
            model_type: Registered model type (e.g. "qwen3_omni_moe").
            cli_overrides: CLI arguments from VllmConfig.
            deploy_config_path: Optional explicit deploy config path.

        Returns:
            List of StageConfig objects.
        """
        # Import pipeline module to ensure registration
        pipeline_dir = cls.PIPELINE_MODELS.get(model_type, model_type)
        try:
            __import__(
                f"vllm_omni.model_executor.models.{pipeline_dir}.pipeline"
            )
        except ImportError:
            pass

        pipeline_cfg = _PIPELINE_REGISTRY[model_type]

        # Resolve deploy config path
        if deploy_config_path is None:
            deploy_path = _DEPLOY_DIR / f"{model_type}.yaml"
        else:
            deploy_path = Path(deploy_config_path)

        if not deploy_path.exists():
            logger.warning(
                "Deploy config not found: %s — using pipeline defaults only",
                deploy_path,
            )
            deploy_cfg = DeployConfig()
        else:
            deploy_cfg = load_deploy_config(deploy_path)

        # Merge pipeline + deploy + CLI
        stages = merge_pipeline_deploy(pipeline_cfg, deploy_cfg, cli_overrides)

        # Apply CLI overrides (same logic as legacy path)
        for stage in stages:
            stage.runtime_overrides = cls._merge_cli_overrides(
                stage, cli_overrides
            )

        return stages

    @classmethod
    def create_default_diffusion(cls, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Single-stage diffusion - no YAML needed.

        Creates a default diffusion stage configuration for single-stage
        diffusion models. Returns a legacy OmegaConf-compatible dict for
        backward compatibility with OmniStage.

        Args:
            kwargs: Engine arguments from CLI/API.

        Returns:
            List containing a single config dict for the diffusion stage.
        """
        # Calculate devices based on parallel config
        devices = "0"
        if "parallel_config" in kwargs:
            num_devices = kwargs["parallel_config"].world_size
            for i in range(1, num_devices):
                devices += f",{i}"

        engine_args: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in ("parallel_config",):
                continue
            engine_args[key] = value

        # Serialize parallel_config as dict for OmegaConf compatibility
        if "parallel_config" in kwargs:
            engine_args["parallel_config"] = asdict(kwargs["parallel_config"])

        engine_args.setdefault("cache_backend", "none")
        engine_args["model_stage"] = "diffusion"

        # Convert dtype to string for OmegaConf
        if "dtype" in engine_args:
            engine_args["dtype"] = str(engine_args["dtype"])

        engine_args.setdefault("max_num_seqs", 1)

        config_dict: dict[str, Any] = {
            "stage_id": 0,
            "stage_type": StageType.DIFFUSION.value,
            "runtime": {
                "process": True,
                "devices": devices,
            },
            "engine_args": create_config(engine_args),
            "final_output": True,
            "final_output_type": "image",
        }

        return [config_dict]

    @classmethod
    def _load_pipeline(cls, model: str, trust_remote_code: bool = True) -> ModelPipeline | None:
        """Load pipeline YAML for the model.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.

        Returns:
            ModelPipeline if found, None otherwise.
        """
        model_type, hf_config = cls._auto_detect_model_type(model, trust_remote_code=trust_remote_code)
        if model_type is None:
            return None

        pipeline_dir = cls.PIPELINE_MODELS.get(model_type)

        # Fallback: check HF architectures when model_type doesn't match
        if pipeline_dir is None and hf_config is not None:
            for arch in getattr(hf_config, "architectures", []) or []:
                pipeline_dir = cls._ARCHITECTURE_MODELS.get(arch)
                if pipeline_dir is not None:
                    model_type = pipeline_dir
                    break

        if pipeline_dir is None:
            logger.debug(f"No pipeline mapping for model_type: {model_type}")
            return None

        pipeline_path = get_pipeline_path(pipeline_dir, "pipeline.yaml")

        if not pipeline_path.exists():
            logger.debug(f"Pipeline file not found: {pipeline_path}")
            return None

        return cls._parse_pipeline_yaml(pipeline_path, model_type)

    # Keys consumed as explicit StageConfig fields — everything else is
    # passed through via yaml_extras.
    _KNOWN_STAGE_KEYS: set[str] = {
        "stage_id",
        "model_stage",
        "stage_type",
        "input_sources",
        "engine_input_source",
        "custom_process_input_func",
        "final_output",
        "final_output_type",
        "worker_type",
        "scheduler_cls",
        "hf_config_name",
        "is_comprehension",
        "engine_args",
        "runtime",
    }

    @classmethod
    def _parse_pipeline_yaml(cls, path: Path, model_type: str) -> ModelPipeline:
        """Parse a pipeline YAML file.

        Args:
            path: Path to the YAML file.
            model_type: Model type identifier.

        Returns:
            ModelPipeline object.
        """
        config_data = load_yaml_config(path)

        stages: list[StageConfig] = []
        for stage_data in config_data.stages:
            # Use .get() for optional fields — idiomatic for OmegaConf DictConfig
            stage_type_str = stage_data.get("stage_type", "llm")
            stage_type = StageType(stage_type_str) if stage_type_str else StageType.LLM

            # Handle both 'input_sources' (new) and 'engine_input_source' (legacy)
            input_sources = stage_data.get("input_sources", None)
            if input_sources is None:
                input_sources = stage_data.get("engine_input_source", [])
            if input_sources is None:
                input_sources = []
            input_sources = list(input_sources)

            # Extract per-stage engine_args and runtime dicts
            raw_ea = stage_data.get("engine_args", None)
            yaml_engine_args = to_dict(raw_ea) if raw_ea is not None else {}
            raw_rt = stage_data.get("runtime", None)
            yaml_runtime = to_dict(raw_rt) if raw_rt is not None else {}

            # Migrate legacy runtime.max_batch_size → engine_args.max_num_seqs
            if "max_batch_size" in yaml_runtime:
                mbs = yaml_runtime.pop("max_batch_size")
                yaml_engine_args.setdefault("max_num_seqs", int(mbs))
                logger.debug(
                    "Stage %s: migrated runtime.max_batch_size=%s to engine_args.max_num_seqs",
                    stage_data.get("stage_id", "?"),
                    mbs,
                )

            # Topology-level fields that also live inside engine_args in legacy
            # YAMLs (worker_type, scheduler_cls, etc.) — read from both places.
            worker_type = stage_data.get("worker_type", None) or yaml_engine_args.pop("worker_type", None)
            scheduler_cls = stage_data.get("scheduler_cls", None) or yaml_engine_args.pop("scheduler_cls", None)
            hf_config_name = stage_data.get("hf_config_name", None) or yaml_engine_args.pop("hf_config_name", None)
            model_stage = getattr(stage_data, "model_stage", None) or yaml_engine_args.pop("model_stage", None)

            # Collect pass-through fields (default_sampling_params,
            # output_connectors, input_connectors, tts_args, etc.)
            yaml_extras: dict[str, Any] = {}
            for key in stage_data:
                if key not in cls._KNOWN_STAGE_KEYS:
                    val = stage_data[key]
                    try:
                        yaml_extras[key] = to_dict(val)
                    except ValueError:
                        yaml_extras[key] = val

            stage = StageConfig(
                stage_id=stage_data.stage_id,
                model_stage=model_stage or "",
                stage_type=stage_type,
                input_sources=input_sources,
                custom_process_input_func=stage_data.get("custom_process_input_func", None),
                final_output=stage_data.get("final_output", False),
                final_output_type=stage_data.get("final_output_type", None),
                worker_type=worker_type,
                scheduler_cls=scheduler_cls,
                hf_config_name=hf_config_name,
                is_comprehension=stage_data.get("is_comprehension", False),
                yaml_engine_args=yaml_engine_args,
                yaml_runtime=yaml_runtime,
                yaml_extras=yaml_extras,
            )
            stages.append(stage)

        # Get pipeline-wide flags
        async_chunk = config_data.get("async_chunk", False)

        # Get optional connector config — check both top-level and nested
        # under ``runtime`` (legacy stage_configs format).
        connectors = None
        edges = None
        if hasattr(config_data, "connectors"):
            connectors = to_dict(config_data.connectors)
        if hasattr(config_data, "edges"):
            edges = to_dict(config_data.edges)
        if hasattr(config_data, "runtime") and config_data.runtime is not None:
            top_runtime = config_data.runtime
            if connectors is None and hasattr(top_runtime, "connectors"):
                connectors = to_dict(top_runtime.connectors)
            if edges is None and hasattr(top_runtime, "edges"):
                edges = to_dict(top_runtime.edges)

        return ModelPipeline(
            model_type=getattr(config_data, "model_type", model_type),
            stages=stages,
            async_chunk=async_chunk,
            connectors=connectors,
            edges=edges,
        )

    @classmethod
    def _auto_detect_model_type(cls, model: str, trust_remote_code: bool = True) -> tuple[str | None, Any]:
        """Auto-detect model_type from model directory.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.

        Returns:
            Tuple of (model_type, hf_config). Both may be None on failure.
        """
        try:
            from vllm.transformers_utils.config import get_config

            hf_config = get_config(model, trust_remote_code=trust_remote_code)
            return hf_config.model_type, hf_config
        except Exception:
            pass

        # Fallback: read config.json directly for custom model types that
        # are not registered with transformers (e.g. qwen3_tts).
        try:
            from vllm.transformers_utils.config import get_hf_file_to_dict

            config_dict = get_hf_file_to_dict("config.json", model, revision=None)
            if config_dict and "model_type" in config_dict:
                return config_dict["model_type"], None
        except Exception as e:
            logger.debug(f"Failed to auto-detect model type for {model}: {e}")

        return None, None

    # Keys that should never be forwarded as engine overrides (internal /
    # orchestrator-only knobs, complex objects, etc.).
    _INTERNAL_KEYS: set[str] = {
        "model",
        "stage_configs_path",
        "stage_id",
        "stage_init_timeout",
        "init_timeout",
        "shm_threshold_bytes",
        "worker_backend",
        "ray_address",
        "batch_timeout",
        "log_stats",
        "tokenizer",
        "parallel_config",
    }

    @classmethod
    def _merge_cli_overrides(
        cls,
        stage: StageConfig,
        cli_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge CLI overrides into stage runtime config.

        All CLI arguments registered by engine config classes (e.g.
        EngineArgs / OmniDiffusionConfig) are accepted as overrides
        unless they appear in ``_INTERNAL_KEYS``.

        Handles:
        - Global overrides (apply to all stages)
        - Per-stage overrides (--stage-N-* format, take precedence)

        Args:
            stage: The stage to merge overrides into.
            cli_overrides: CLI arguments from VllmConfig/OmniDiffusionConfig.

        Returns:
            Dict of runtime overrides for this stage.
        """
        result: dict[str, Any] = {}

        # Apply global overrides – any key not in the internal blocklist
        # is forwarded so that engine-registered params work out of the box.
        for key, value in cli_overrides.items():
            if key in cls._INTERNAL_KEYS:
                continue
            if re.match(r"stage_\d+_", key):
                # Per-stage keys handled below
                continue
            if value is not None:
                result[key] = value

        # Apply per-stage overrides (--stage-N-* format, take precedence)
        stage_prefix = f"stage_{stage.stage_id}_"
        for key, value in cli_overrides.items():
            if key.startswith(stage_prefix) and value is not None:
                param_name = key[len(stage_prefix) :]
                if param_name in cls._INTERNAL_KEYS:
                    continue
                result[param_name] = value

        return result
