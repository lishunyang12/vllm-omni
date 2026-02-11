# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Two-Tier Stage Configuration System for vLLM-Omni.

Design Principles:
- Tier-1 (Pipeline Topology): INTERNAL ONLY - set by model developers at integration time
- Tier-2 (Runtime Config): User-configurable via CLI args (VllmConfig/OmniDiffusionConfig params)

Users interact only with Tier-2 (CLI). Tier-1 topology is bundled with models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.config.yaml_util import create_config, load_yaml_raw, to_dict
from vllm_omni.model_executor.stage_topologies import get_topology_path

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class StageType(str, Enum):
    """Type of processing stage in the Omni pipeline."""

    LLM = "llm"
    DIFFUSION = "diffusion"


@dataclass
class StageConfig:
    """Cleaned-up stage config - only multi-stage relevant fields.

    Note: Engine params (gpu_memory_utilization, tp_size, etc.) come from
    VllmConfig or OmniDiffusionConfig via CLI, NOT from this class.

    This class represents Tier-1 (Internal) configuration that is:
    - Set by model developers at integration time
    - NOT user-editable
    - Defines pipeline topology, worker types, and processing hooks

    Attributes:
        stage_id: Unique identifier for this stage in the pipeline.
        model_stage: Stage name (e.g., "thinker", "talker", "code2wav", "dit").
        stage_type: Type of stage - LLM or DIFFUSION.
        input_sources: List of upstream stage IDs this stage receives input from.
        custom_process_input_func: Full module path to custom input processing function.
        final_output: Whether this stage produces final output.
        final_output_type: Type of final output ("text", "audio", "image").
        worker_type: Worker type ("ar" or "generation").
        scheduler_cls: Full module path to a custom scheduler class.
            When None (the default), the engine uses its built-in scheduler.
            Model developers may specify a custom class (e.g.,
            "vllm_omni.core.scheduler.OmniScheduler") to override scheduling
            behaviour for a particular stage.
        hf_config_name: Name of the HuggingFace config to use.
        is_comprehension: Whether this stage handles comprehension (input understanding).
    """

    # Identity
    stage_id: int
    model_stage: str

    # Stage type
    stage_type: StageType = StageType.LLM

    # Pipeline topology (Tier-1 - Internal, set by developer).
    # Lists upstream stage IDs this stage receives data from.
    # Future: may be derived from StageTopology.edges for richer
    # edge metadata (e.g., data format, buffering policy).
    input_sources: list[int] = field(default_factory=list)

    # Processing hooks (Tier-1 - Internal)
    custom_process_input_func: str | None = None

    # Output configuration (Tier-1 - Internal)
    final_output: bool = False
    final_output_type: str | None = None  # "text", "audio", "image"

    # Worker configuration (Tier-1 - Internal)
    worker_type: str | None = None  # "ar" or "generation"
    scheduler_cls: str | None = None
    hf_config_name: str | None = None

    # Comprehension flag
    is_comprehension: bool = False

    # Runtime overrides (Tier-2 - populated from CLI, not from topology file)
    runtime_overrides: dict[str, Any] = field(default_factory=dict)

    def to_omegaconf(self) -> Any:
        """Convert to OmegaConf for backward compatibility with OmniStage.

        Returns:
            OmegaConf DictConfig with stage configuration in legacy format.
        """
        # Build engine_args dict with required fields
        engine_args: dict[str, Any] = {
            "model_stage": self.model_stage,
        }

        if self.worker_type:
            engine_args["worker_type"] = self.worker_type
        if self.scheduler_cls:
            engine_args["scheduler_cls"] = self.scheduler_cls
        if self.hf_config_name:
            engine_args["hf_config_name"] = self.hf_config_name

        # Apply runtime overrides from Tier-2 (CLI args)
        for key, value in self.runtime_overrides.items():
            if key not in ("devices", "max_batch_size"):
                engine_args[key] = value

        # Build runtime config
        runtime: dict[str, Any] = {
            "process": True,
            "max_batch_size": self.runtime_overrides.get("max_batch_size", 1),
        }
        if "devices" in self.runtime_overrides:
            runtime["devices"] = self.runtime_overrides["devices"]

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

        return create_config(config_dict)


@dataclass
class StageTopology:
    """Internal Tier-1 topology - bundled with model, not user-editable.

    This class represents the complete pipeline topology for a multi-stage model.
    It is defined by model developers and validated at integration time (not runtime).

    Attributes:
        model_type: Model type identifier (e.g., "qwen3_omni_moe").
        stages: List of StageConfig objects defining the pipeline stages.
        connectors: Optional connector configuration for distributed deployment.
        edges: Optional explicit edge definitions for the pipeline topology.
    """

    model_type: str
    stages: list[StageConfig]

    # Optional distributed configuration
    connectors: dict[str, Any] | None = None
    edges: list[dict[str, Any]] | None = None

    def validate_topology(self) -> list[str]:
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

    def get_stage(self, stage_id: int) -> StageConfig | None:
        """Get stage by ID.

        Args:
            stage_id: The stage ID to look up.

        Returns:
            StageConfig if found, None otherwise.
        """
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None


class StageConfigFactory:
    """Factory merges Tier-1 pipeline topology with Tier-2 CLI overrides.

    This factory is the main entry point for creating stage configurations.
    It handles:
    - Loading internal Tier-1 pipeline topology files
    - Auto-detecting model architecture
    - Merging CLI overrides (Tier-2) into stage configs
    - Supporting both single-stage and multi-stage models
    """

    # Mapping of model types to topology file names
    TOPOLOGY_FILES: dict[str, str] = {
        "qwen3_omni_moe": "qwen3_omni_moe.yaml",
        "qwen2_5_omni": "qwen2_5_omni.yaml",
        "bagel": "bagel.yaml",
    }

    # Mapping of model types to architecture classes
    ARCH_MAPPING: dict[str, str] = {
        "qwen3_omni_moe": "Qwen3OmniMoeForConditionalGeneration",
        "qwen2_5_omni": "Qwen2_5OmniForConditionalGeneration",
        "bagel": "BagelForConditionalGeneration",
    }

    @classmethod
    def create_from_model(
        cls,
        model: str,
        cli_overrides: dict[str, Any] | None = None,
        stage_id_filter: int | None = None,
    ) -> list[StageConfig]:
        """Load internal topology, merge with CLI overrides.

        Args:
            model: Model name or path.
            cli_overrides: Tier-2 CLI overrides from VllmConfig/OmniDiffusionConfig.
            stage_id_filter: If specified, only return the stage with this ID
                           (for independent stage launch).

        Returns:
            List of StageConfig objects with CLI overrides applied.
        """
        if cli_overrides is None:
            cli_overrides = {}

        # Try to auto-detect model type and load topology
        topology = cls._load_topology(model)

        if topology is None:
            # No topology found - return empty list (caller should use default diffusion)
            return []

        # Validate pipeline topology
        errors = topology.validate_topology()
        if errors:
            logger.warning(f"Topology validation warnings for {model}: {errors}")

        # Apply CLI overrides and filter stages
        result: list[StageConfig] = []
        for stage in topology.stages:
            if stage_id_filter is not None and stage.stage_id != stage_id_filter:
                continue

            # Merge global CLI overrides
            stage.runtime_overrides = cls._merge_cli_overrides(stage, cli_overrides)
            result.append(stage)

        return result

    @classmethod
    def create_default_diffusion(cls, kwargs: dict[str, Any]) -> list[Any]:
        """Single-stage diffusion - no YAML needed.

        Creates a default diffusion stage configuration for single-stage
        diffusion models using a typed ``StageConfig`` dataclass, then
        converts to OmegaConf for backward compatibility with OmniStage.

        Args:
            kwargs: Engine arguments from CLI/API.

        Returns:
            List containing a single OmegaConf config for the diffusion stage.
        """
        # Calculate devices based on parallel config
        devices = "0"
        if "parallel_config" in kwargs:
            num_devices = kwargs["parallel_config"].world_size
            for i in range(1, num_devices):
                devices += f",{i}"

        # Collect engine args – skip non-serializable objects
        engine_args: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in ("parallel_config",):
                continue
            engine_args[key] = value

        engine_args.setdefault("cache_backend", "none")
        engine_args["model_stage"] = "diffusion"

        # Convert dtype to string for OmegaConf
        if "dtype" in engine_args:
            engine_args["dtype"] = str(engine_args["dtype"])

        # Build typed StageConfig
        stage = StageConfig(
            stage_id=0,
            model_stage="diffusion",
            stage_type=StageType.DIFFUSION,
            final_output=True,
            final_output_type="image",
            runtime_overrides={
                "devices": devices,
                "max_batch_size": 1,
                **engine_args,
            },
        )

        # Convert to OmegaConf dict for backward compat with OmniStage.
        # Diffusion stages carry all engine args inside runtime_overrides,
        # so we build the legacy dict directly to preserve that shape.
        config_dict = {
            "stage_id": stage.stage_id,
            "stage_type": stage.stage_type.value,
            "runtime": {
                "process": True,
                "devices": devices,
                "max_batch_size": 1,
            },
            "engine_args": create_config(engine_args),
            "final_output": stage.final_output,
            "final_output_type": stage.final_output_type,
        }

        return [config_dict]

    @classmethod
    def _load_topology(cls, model: str) -> StageTopology | None:
        """Load internal Tier-1 pipeline topology YAML for the model.

        Args:
            model: Model name or path.

        Returns:
            StageTopology if found, None otherwise.
        """
        model_type = cls._auto_detect_model_type(model)
        if model_type is None:
            return None

        topology_file = cls.TOPOLOGY_FILES.get(model_type)
        if topology_file is None:
            logger.debug(f"No topology file mapping for model_type: {model_type}")
            return None

        topology_path = get_topology_path(topology_file)

        if not topology_path.exists():
            logger.debug(f"Topology file not found: {topology_path}")
            return None

        return cls._parse_topology_yaml(topology_path, model_type)

    @classmethod
    def _parse_topology_yaml(cls, path: Path, model_type: str) -> StageTopology:
        """Parse a Tier-1 pipeline topology YAML file.

        Args:
            path: Path to the YAML file.
            model_type: Model type identifier.

        Returns:
            StageTopology object.
        """
        config_data = load_yaml_raw(path)

        stages: list[StageConfig] = []
        for stage_data in config_data.stages:
            stage_type_str = getattr(stage_data, "stage_type", "llm")
            stage_type = StageType(stage_type_str) if stage_type_str else StageType.LLM

            # Handle both 'input_sources' (new) and 'engine_input_source' (legacy)
            input_sources = getattr(stage_data, "input_sources", None)
            if input_sources is None:
                input_sources = getattr(stage_data, "engine_input_source", [])
            if input_sources is None:
                input_sources = []
            input_sources = list(input_sources)

            stage = StageConfig(
                stage_id=stage_data.stage_id,
                model_stage=stage_data.model_stage,
                stage_type=stage_type,
                input_sources=input_sources,
                custom_process_input_func=getattr(stage_data, "custom_process_input_func", None),
                final_output=getattr(stage_data, "final_output", False),
                final_output_type=getattr(stage_data, "final_output_type", None),
                worker_type=getattr(stage_data, "worker_type", None),
                scheduler_cls=getattr(stage_data, "scheduler_cls", None),
                hf_config_name=getattr(stage_data, "hf_config_name", None),
                is_comprehension=getattr(stage_data, "is_comprehension", False),
            )
            stages.append(stage)

        # Get optional connector config
        connectors = to_dict(config_data.connectors) if hasattr(config_data, "connectors") else None
        edges = to_dict(config_data.edges) if hasattr(config_data, "edges") else None

        return StageTopology(
            model_type=getattr(config_data, "model_type", model_type),
            stages=stages,
            connectors=connectors,
            edges=edges,
        )

    @classmethod
    def _auto_detect_model_type(cls, model: str) -> str | None:
        """Auto-detect model_type from model directory.

        Args:
            model: Model name or path.

        Returns:
            Model type string if detected, None otherwise.
        """
        try:
            from vllm.transformers_utils.config import get_config

            hf_config = get_config(model, trust_remote_code=True)
            return hf_config.model_type
        except Exception as e:
            logger.debug(f"Failed to auto-detect model type for {model}: {e}")
            return None

    @classmethod
    def _auto_detect_model_arch(cls, model: str) -> str | None:
        """Auto-detect model_arch from model directory.

        Args:
            model: Model name or path.

        Returns:
            Model architecture class name if detected, None otherwise.
        """
        model_type = cls._auto_detect_model_type(model)
        if model_type is None:
            return None

        # Check mapping first
        if model_type in cls.ARCH_MAPPING:
            return cls.ARCH_MAPPING[model_type]

        # Fallback: generate from model_type
        # Convert snake_case to PascalCase and add suffix
        parts = model_type.split("_")
        pascal_case = "".join(part.capitalize() for part in parts)
        return f"{pascal_case}ForConditionalGeneration"

    # Well-known Tier-2 runtime parameters.  Any CLI arg whose name
    # matches one of these keys is forwarded to every stage by default.
    # Additional engine-registered args are also accepted (see
    # _merge_cli_overrides), so this set does NOT need to be exhaustive.
    RUNTIME_PARAMS: set[str] = {
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "devices",
        "enforce_eager",
        "max_num_batched_tokens",
        "trust_remote_code",
        "max_batch_size",
        "distributed_executor_backend",
        "enable_prefix_caching",
    }

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
        EngineArgs / OmniDiffusionConfig) are accepted as overrides,
        not just the well-known ``RUNTIME_PARAMS`` set.

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
            if key.startswith("stage_"):
                # Per-stage keys handled below
                continue
            if value is not None:
                result[key] = value

        # Apply per-stage overrides (--stage-N-* format, take precedence)
        stage_prefix = f"stage_{stage.stage_id}_"
        for key, value in cli_overrides.items():
            if key.startswith(stage_prefix) and value is not None:
                param_name = key[len(stage_prefix):]
                result[param_name] = value

        return result

    @classmethod
    def auto_allocate_memory(
        cls,
        stages: list[StageConfig],
        total_memory: float = 0.9,
    ) -> dict[int, float]:
        """Calculate safe memory allocation when stages share devices.

        Args:
            stages: List of stages to allocate memory for.
            total_memory: Total memory utilization target (default 0.9).

        Returns:
            Dict mapping stage_id to gpu_memory_utilization.
        """
        if not stages:
            return {}

        # Group stages by device
        device_stages: dict[str, list[int]] = {}
        for stage in stages:
            devices = stage.runtime_overrides.get("devices", "0")
            if devices not in device_stages:
                device_stages[devices] = []
            device_stages[devices].append(stage.stage_id)

        # Allocate memory proportionally
        result: dict[int, float] = {}
        for device, stage_ids in device_stages.items():
            share = total_memory / len(stage_ids)
            for stage_id in stage_ids:
                result[stage_id] = share

        return result
