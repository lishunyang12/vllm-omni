# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage configuration system for vLLM-Omni."""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

from vllm_omni.config.yaml_util import create_config, load_yaml_config, to_dict

_MODELS_DIR = Path(__file__).resolve().parent.parent / "model_executor" / "models"


def get_pipeline_path(model_dir: str, filename: str) -> Path:
    return _MODELS_DIR / model_dir / filename


logger = init_logger(__name__)


class StageType(str, Enum):
    """Type of processing stage in the Omni pipeline."""

    # TODO(@lishunyang12): remove once all models migrate to StageExecutionType
    LLM = "llm"
    DIFFUSION = "diffusion"


class StageExecutionType(str, Enum):
    """Merged StageType + WorkerType — 3 combinations today."""

    LLM_AR = "llm_ar"
    LLM_GENERATION = "llm_generation"
    DIFFUSION = "diffusion"


_EXECUTION_TYPE_TO_SCHEDULER: dict[StageExecutionType, str | None] = {
    StageExecutionType.LLM_AR: ("vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler"),
    StageExecutionType.LLM_GENERATION: ("vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler"),
    StageExecutionType.DIFFUSION: None,
}


@dataclass(frozen=True)
class StagePipelineConfig:
    """Fixed topology for one stage (frozen, not user-configurable)."""

    stage_id: int
    model_stage: str
    execution_type: StageExecutionType = StageExecutionType.LLM_AR
    input_sources: tuple[int, ...] = ()
    final_output: bool = False
    final_output_type: str | None = None
    is_comprehension: bool = False
    requires_multimodal_data: bool = False
    hf_config_name: str | None = None
    engine_output_type: str | None = None
    # Optional per-stage architecture override. When ``None`` (the common
    # case), the stage uses the pipeline-level ``model_arch``. Used by
    # multi-arch pipelines like Qwen3-TTS where the talker and code2wav
    # stages have different model classes.
    model_arch: str | None = None
    sampling_constraints: dict[str, Any] = field(default_factory=dict)
    custom_process_input_func: str | None = None
    custom_process_next_stage_input_func: str | None = None
    prompt_expand_func: str | None = None
    cfg_kv_collect_func: str | None = None
    omni_kv_config: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline topology for a model (frozen)."""

    model_type: str
    model_arch: str = ""
    stages: tuple[StagePipelineConfig, ...] = ()

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
        """Return list of topology errors (empty if valid)."""
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
                    errors.append(f"Stage {stage.stage_id} references non-existent input source {src}")
                if src == stage.stage_id:
                    errors.append(f"Stage {stage.stage_id} references itself")
        if not any(not s.input_sources for s in self.stages):
            errors.append("No entry point (stage with empty input_sources)")
        return errors


_PIPELINE_REGISTRY: dict[str, PipelineConfig] = {}


def register_pipeline(pipeline: PipelineConfig) -> None:
    """Register a pipeline config (called at import time by pipeline.py modules)."""
    errors = pipeline.validate()
    if errors:
        logger.warning("Pipeline %s has issues: %s", pipeline.model_type, errors)
    _PIPELINE_REGISTRY[pipeline.model_type] = pipeline


_DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"


@dataclass
class StageDeployConfig:
    """Per-stage deployment knobs (all CLI-overridable)."""

    stage_id: int
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
    devices: str = "0"
    output_connectors: dict[str, str] | None = None
    input_connectors: dict[str, str] | None = None
    default_sampling_params: dict[str, Any] | None = None
    engine_extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeployConfig:
    """Loaded from deploy/<model>.yaml — the only config file users edit."""

    async_chunk: bool = True
    connectors: dict[str, Any] | None = None
    edges: list[dict[str, Any]] | None = None
    stages: list[StageDeployConfig] = field(default_factory=list)
    platforms: dict[str, Any] | None = None
    # Optional explicit pipeline registration key. When set, overrides the
    # auto-detected ``model_type`` lookup in the pipeline registry. Used by
    # variant deploys whose topology differs from the default for the same
    # HuggingFace model_type — e.g. ``qwen3_tts_no_async_chunk`` reuses the
    # qwen3_tts model classes but registers a different pipeline (different
    # processor functions, no SharedMemoryConnector).
    pipeline: str | None = None


_STAGE_NON_ENGINE_KEYS = frozenset(
    {
        "stage_id",
        "devices",
        "output_connectors",
        "input_connectors",
        "default_sampling_params",
        "engine_extras",
    }
)

# Fields on StageDeployConfig that are populated from engine_args dict
_STAGE_DEPLOY_FIELDS = {
    f.name: f for f in __import__("dataclasses").fields(StageDeployConfig) if f.name not in _STAGE_NON_ENGINE_KEYS
}


def _parse_stage_deploy(stage_data: dict[str, Any]) -> StageDeployConfig:
    """Parse a single stage entry from deploy YAML into StageDeployConfig."""
    if "engine_args" in stage_data:
        engine_args = dict(stage_data["engine_args"])
        devices = stage_data.get("runtime", {}).get("devices", stage_data.get("devices", "0"))
    else:
        engine_args = {k: v for k, v in stage_data.items() if k not in _STAGE_NON_ENGINE_KEYS and k != "stage_id"}
        devices = stage_data.get("devices", "0")

    kwargs: dict[str, Any] = {"stage_id": stage_data["stage_id"], "devices": devices}
    for name, f in _STAGE_DEPLOY_FIELDS.items():
        if name in engine_args:
            kwargs[name] = engine_args.pop(name)

    kwargs["output_connectors"] = stage_data.get("output_connectors")
    kwargs["input_connectors"] = stage_data.get("input_connectors")
    kwargs["default_sampling_params"] = stage_data.get("default_sampling_params")
    kwargs["engine_extras"] = engine_args
    return StageDeployConfig(**kwargs)


def _deep_merge_stage(base: dict, overlay: dict) -> dict:
    """Deep-merge overlay stage dict onto base, matching by stage_id."""
    merged = dict(base)
    for k, v in overlay.items():
        if k == "default_sampling_params" and k in merged and isinstance(v, dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


def _merge_stage_lists(
    base_stages: list[dict[str, Any]] | None,
    overlay_stages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge two ``stages:`` lists by ``stage_id`` (overlay wins per field)."""
    by_id: dict[int, dict[str, Any]] = {s["stage_id"]: s for s in (base_stages or [])}
    for overlay_stage in overlay_stages or []:
        sid = overlay_stage["stage_id"]
        if sid in by_id:
            by_id[sid] = _deep_merge_stage(by_id[sid], overlay_stage)
        else:
            by_id[sid] = overlay_stage
    return list(by_id.values())


def _merge_platforms(
    base: dict[str, Any] | None,
    overlay: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Deep-merge two ``platforms:`` blocks per-platform, per-stage_id."""
    if not base and not overlay:
        return None
    base = base or {}
    overlay = overlay or {}
    merged: dict[str, Any] = {}
    for plat in set(base) | set(overlay):
        bp = base.get(plat) or {}
        op = overlay.get(plat) or {}
        merged_plat = {**bp, **{k: v for k, v in op.items() if k != "stages"}}
        merged_plat["stages"] = _merge_stage_lists(bp.get("stages"), op.get("stages"))
        merged[plat] = merged_plat
    return merged


def _resolve_deploy_yaml(path: str | Path) -> dict[str, Any]:
    """Load a deploy YAML with optional ``base_config`` inheritance."""
    raw_dict = to_dict(load_yaml_config(path))

    base_path = raw_dict.pop("base_config", None)
    if base_path is None:
        return raw_dict

    # Resolve relative to the overlay file's directory
    base_path = Path(path).parent / base_path
    base_dict = _resolve_deploy_yaml(base_path)

    # Merge top-level scalars: overlay wins. ``stages:`` and ``platforms:``
    # are deep-merged below so an overlay can layer on top of the base.
    merged = {
        **base_dict,
        **{k: v for k, v in raw_dict.items() if k not in ("stages", "platforms")},
    }
    merged["stages"] = _merge_stage_lists(base_dict.get("stages"), raw_dict.get("stages"))
    merged_platforms = _merge_platforms(base_dict.get("platforms"), raw_dict.get("platforms"))
    if merged_platforms is not None:
        merged["platforms"] = merged_platforms

    return merged


def load_deploy_config(path: str | Path) -> DeployConfig:
    """Load a deploy YAML (with optional base_config inheritance)."""
    raw_dict = _resolve_deploy_yaml(path)

    stages = [_parse_stage_deploy(s) for s in raw_dict.get("stages", [])]

    return DeployConfig(
        async_chunk=raw_dict.get("async_chunk", True),
        connectors=raw_dict.get("connectors", None),
        edges=raw_dict.get("edges", None),
        stages=stages,
        platforms=raw_dict.get("platforms", None),
        pipeline=raw_dict.get("pipeline", None),
    )


def _detect_platform() -> str | None:
    """Return "npu", "rocm", "xpu", or None (CUDA default)."""
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
    """Merge platform-specific stage overrides into deploy config."""
    if platform is None:
        platform = _detect_platform()
    if platform is None or deploy.platforms is None:
        return deploy
    platform_section = deploy.platforms.get(platform)
    if platform_section is None:
        return deploy

    platform_stages = platform_section.get("stages", [])
    base_by_id = {s.stage_id: s for s in deploy.stages}

    for ps in platform_stages:
        base = base_by_id.get(ps["stage_id"])
        if base is None:
            continue
        if "engine_args" in ps:
            overrides = dict(ps["engine_args"])
            if "devices" in ps.get("runtime", {}):
                object.__setattr__(base, "devices", ps["runtime"]["devices"])
        else:
            overrides = {k: v for k, v in ps.items() if k not in ("stage_id", "devices")}
            if "devices" in ps:
                object.__setattr__(base, "devices", ps["devices"])
        for key, val in overrides.items():
            if hasattr(base, key):
                object.__setattr__(base, key, val)
            else:
                base.engine_extras[key] = val

    return deploy


def merge_pipeline_deploy(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    cli_overrides: dict[str, Any] | None = None,
) -> list[StageConfig]:
    """Merge pipeline + deploy + platform overrides → list[StageConfig]."""
    if cli_overrides is None:
        cli_overrides = {}

    deploy = _apply_platform_overrides(deploy)
    deploy_by_id = {s.stage_id: s for s in deploy.stages}

    result: list[StageConfig] = []
    for ps in pipeline.stages:
        ds = deploy_by_id.get(ps.stage_id)

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

        yaml_engine_args: dict[str, Any] = {}
        # Per-stage model_arch override (e.g. Qwen3-TTS code2wav stage)
        # falls back to the pipeline-level model_arch when not set.
        yaml_engine_args["model_arch"] = ps.model_arch or pipeline.model_arch
        if ps.engine_output_type:
            yaml_engine_args["engine_output_type"] = ps.engine_output_type
        if ps.custom_process_next_stage_input_func:
            yaml_engine_args["custom_process_next_stage_input_func"] = ps.custom_process_next_stage_input_func

        if ds is not None:
            for k, v in asdict(ds).items():
                if k in _STAGE_NON_ENGINE_KEYS or v is None:
                    continue
                yaml_engine_args[k] = v
            yaml_engine_args.update(ds.engine_extras)

        if deploy.async_chunk:
            yaml_engine_args["async_chunk"] = True

        yaml_runtime: dict[str, Any] = {"process": True}
        if ds is not None:
            yaml_runtime["devices"] = ds.devices

        custom_process_input_func = ps.custom_process_input_func

        yaml_extras: dict[str, Any] = {}

        sampling: dict[str, Any] = {}
        if ds is not None and ds.default_sampling_params:
            sampling.update(ds.default_sampling_params)
        sampling.update(ps.sampling_constraints)
        if sampling:
            yaml_extras["default_sampling_params"] = sampling

        if ds is not None and ds.output_connectors:
            yaml_extras["output_connectors"] = dict(ds.output_connectors)
        if ds is not None and ds.input_connectors:
            yaml_extras["input_connectors"] = dict(ds.input_connectors)
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


@dataclass
class StageConfig:
    """Per-stage config (legacy path). Used by both new and legacy loaders.

    TODO(@lishunyang12): replace with ResolvedStageConfig once all models are migrated.
    """

    stage_id: int
    model_stage: str
    stage_type: StageType = StageType.LLM
    input_sources: list[int] = field(default_factory=list)
    custom_process_input_func: str | None = None
    final_output: bool = False
    final_output_type: str | None = None
    worker_type: str | None = None
    scheduler_cls: str | None = None
    hf_config_name: str | None = None
    is_comprehension: bool = False
    yaml_engine_args: dict[str, Any] = field(default_factory=dict)
    yaml_runtime: dict[str, Any] = field(default_factory=dict)
    yaml_extras: dict[str, Any] = field(default_factory=dict)
    runtime_overrides: dict[str, Any] = field(default_factory=dict)

    def to_omegaconf(self) -> Any:
        """TODO(@lishunyang12): remove once engine consumes ResolvedStageConfig directly."""
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
    """Complete pipeline definition for a multi-stage model (legacy).

    TODO(@lishunyang12): remove once all models migrate to PipelineConfig.
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
        cli_explicit_keys: set[str] | None = None,
    ) -> list[StageConfig] | None:
        """Load pipeline + deploy config, merge with CLI overrides.

        Checks _PIPELINE_REGISTRY first (new path), falls back to legacy YAML.

        ``cli_explicit_keys`` is the set of CLI keys the user actually typed
        (captured at the parser layer in ``vllm serve``). When ``None`` —
        which is the case for programmatic ``Omni()`` callers — every kwarg
        in ``cli_overrides`` is treated as explicit.
        """
        if cli_overrides is None:
            cli_overrides = {}

        trust_remote_code = cli_overrides.get("trust_remote_code", True)

        # --- New path: check pipeline registry first ---
        model_type, _ = cls._auto_detect_model_type(model, trust_remote_code=trust_remote_code)
        if model_type and model_type in cls.PIPELINE_MODELS:
            pipeline_dir = cls.PIPELINE_MODELS[model_type]
            try:
                __import__(f"vllm_omni.model_executor.models.{pipeline_dir}.pipeline")
            except ImportError:
                pass
        if model_type and model_type in _PIPELINE_REGISTRY:
            return cls._create_from_registry(model_type, cli_overrides, deploy_config_path, cli_explicit_keys)

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
        cli_explicit_keys: set[str] | None = None,
    ) -> list[StageConfig]:
        """Create StageConfigs from pipeline registry + deploy YAML.

        Precedence (high → low):
            explicit CLI args  >  deploy YAML  >  parser default CLI values

        ``cli_explicit_keys`` carries the set of long-option attribute names
        the user actually typed (captured in ``OmniServeCommand.cmd``). Any
        kwarg whose key is not in that set is treated as a parser default
        and is only used to fill fields YAML doesn't already cover. When the
        set is ``None`` (programmatic ``Omni()`` callers, which have no
        argparse layer), every kwarg is treated as explicit.

        If the loaded deploy YAML has a ``pipeline:`` field set, it
        overrides the auto-detected ``model_type`` for the pipeline
        registry lookup. This lets variant deploys (e.g.
        ``qwen3_tts_no_async_chunk``) reuse the model classes of a
        different HuggingFace ``model_type`` while running a different
        topology (different processor functions, different connectors).
        """
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

        # Resolve which pipeline registration to use. The deploy YAML's
        # explicit ``pipeline:`` field (if set) wins over the auto-detected
        # model_type so variant topologies can be selected without renaming
        # the model.
        pipeline_key = deploy_cfg.pipeline or model_type
        if pipeline_key not in _PIPELINE_REGISTRY:
            raise KeyError(
                f"Pipeline {pipeline_key!r} not in registry "
                f"(deploy {deploy_path.name!r} requested it via 'pipeline:' field "
                f"or via auto-detected model_type). Available: "
                f"{sorted(_PIPELINE_REGISTRY.keys())}"
            )
        pipeline_cfg = _PIPELINE_REGISTRY[pipeline_key]

        stages = merge_pipeline_deploy(pipeline_cfg, deploy_cfg, cli_overrides)

        # Split CLI overrides by explicitness. Per-stage `stage_<id>_*` keys
        # are always treated as explicit (the user typed them, either as a
        # flag or via --stage-overrides JSON).
        explicit_overrides: dict[str, Any] = {}
        default_overrides: dict[str, Any] = {}
        for key, value in cli_overrides.items():
            if value is None:
                continue
            is_per_stage = bool(re.match(r"stage_\d+_", key))
            is_explicit = cli_explicit_keys is None or key in cli_explicit_keys or is_per_stage
            if is_explicit:
                explicit_overrides[key] = value
            else:
                default_overrides[key] = value

        for stage in stages:
            # Default CLI values fill only fields YAML doesn't already set.
            # Explicit CLI values override YAML on top of that.
            yaml_keys = set(stage.yaml_engine_args)
            fallback = {k: v for k, v in default_overrides.items() if k not in yaml_keys}
            merged = {**fallback, **explicit_overrides}
            stage.runtime_overrides = cls._merge_cli_overrides(stage, merged)

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
