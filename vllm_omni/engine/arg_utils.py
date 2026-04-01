import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any

from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger

from vllm_omni.config import OmniModelConfig
from vllm_omni.engine.output_modality import OutputModality
from vllm_omni.model_executor.models.voxcpm.configuration_voxcpm import VoxCPMConfig
from vllm_omni.model_executor.models.voxcpm.native_config import (
    detect_native_voxcpm_model_type,
    ensure_hf_compatible_voxcpm_config,
)
from vllm_omni.plugins import load_omni_general_plugins

logger = init_logger(__name__)


def _register_omni_hf_configs() -> None:
    try:
        from transformers import AutoConfig

        from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
        from vllm_omni.model_executor.models.voxtral_tts.configuration_voxtral_tts import (
            VoxtralTTSConfig,
        )
    except Exception as exc:  # pragma: no cover - best-effort optional registration
        logger.warning("Skipping omni HF config registration due to import error: %s", exc)
        return

    for model_type, config_cls in [
        ("qwen3_tts", Qwen3TTSConfig),
        ("cosyvoice3", CosyVoice3Config),
        ("voxtral_tts", VoxtralTTSConfig),
        ("voxcpm", VoxCPMConfig),
    ]:
        try:
            AutoConfig.register(model_type, config_cls)
        except ValueError:
            pass


def _maybe_prepare_model_hf_config_path(model: str, hf_config_path: str | None) -> str | None:
    if hf_config_path:
        return hf_config_path

    if detect_native_voxcpm_model_type(model) == "voxcpm":
        return ensure_hf_compatible_voxcpm_config(model)

    return hf_config_path


def register_omni_models_to_vllm():
    from vllm.model_executor.models import ModelRegistry

    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    _register_omni_hf_configs()

    supported_archs = ModelRegistry.get_supported_archs()
    for arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items():
        if arch not in supported_archs:
            ModelRegistry.register_model(arch, f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}:{cls_name}")


@dataclass
class OmniEngineArgs(EngineArgs):
    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str | None = None
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    stage_connector_spec: dict[str, Any] = field(default_factory=dict)
    async_chunk: bool = False
    omni_kv_config: dict | None = None
    quantization_config: Any | None = None
    worker_type: str | None = None
    task_type: str | None = None

    def __post_init__(self) -> None:
        load_omni_general_plugins()
        super().__post_init__()

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "OmniEngineArgs":
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)})
        return engine_args

    def _ensure_omni_models_registered(self):
        if hasattr(self, "_omni_models_registered"):
            return True
        register_omni_models_to_vllm()
        self._omni_models_registered = True
        return True

    def create_model_config(self) -> OmniModelConfig:
        self._ensure_omni_models_registered()
        self.hf_config_path = _maybe_prepare_model_hf_config_path(self.model, self.hf_config_path)

        stage_connector_config = {
            "name": self.stage_connector_spec.get("name", "SharedMemoryConnector"),
            "extra": self.stage_connector_spec.get("extra", {}).copy(),
        }
        stage_connector_config["extra"]["stage_id"] = self.stage_id

        if self.model_arch:
            if self.hf_overrides is None:
                self.hf_overrides = {}
            if isinstance(self.hf_overrides, dict):
                self.hf_overrides.setdefault("architectures", [self.model_arch])

        model_config = super().create_model_config()

        return OmniModelConfig.from_vllm_model_config(
            model_config=model_config,
            stage_id=self.stage_id,
            async_chunk=self.async_chunk,
            model_stage=self.model_stage,
            model_arch=self.model_arch,
            worker_type=self.worker_type,
            engine_output_type=self.engine_output_type,
            hf_config_name=self.hf_config_name,
            custom_process_next_stage_input_func=self.custom_process_next_stage_input_func,
            stage_connector_config=stage_connector_config,
            omni_kv_config=self.omni_kv_config,
            task_type=self.task_type,
        )

    @property
    def output_modality(self) -> OutputModality:
        return OutputModality.from_string(self.engine_output_type)
