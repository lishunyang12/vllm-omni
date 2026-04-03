from transformers import AutoConfig

from vllm_omni.model_executor.models.voxcpm.configuration_voxcpm import VoxCPMConfig

AutoConfig.register("voxcpm", VoxCPMConfig)

__all__ = ["VoxCPMConfig"]
