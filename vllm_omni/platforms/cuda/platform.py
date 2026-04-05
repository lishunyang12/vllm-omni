# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger
from vllm.platforms.cuda import CudaPlatformBase
from vllm.platforms.interface import DeviceCapability

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


class CudaOmniPlatform(OmniPlatform, CudaPlatformBase):
    """CUDA/GPU implementation of OmniPlatform (default).

    Inherits all CUDA-specific implementations from vLLM's CudaPlatform,
    and adds Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.CUDA

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_ar_worker.GPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/model_executor/stage_configs"

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        from vllm_omni.diffusion.envs import PACKAGES_CHECKER

        # has_flash_attn from envs.py is already hardware-aware:
        #   - sm80/86/89/90: True iff fa3_fwd_interface / flash_attn_interface / FA2 importable
        #   - sm100+ (Blackwell): True iff flash-attn>=4.0 (FA4) importable
        #   - otherwise: False
        packages_info = PACKAGES_CHECKER.get_packages_info()
        flash_attn_supported = packages_info.get("has_flash_attn", False)

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            if backend_upper == "FLASH_ATTN" and not flash_attn_supported:
                logger.warning(
                    "Flash Attention is not available on this GPU. On Blackwell+ "
                    "(sm100+), install flash-attn>=4.0 (FA4). On sm80-90, install "
                    "fa3-fwd / flash-attention / flash-attn>=2.6. Falling back to "
                    "TORCH_SDPA backend."
                )
                logger.info("Defaulting to diffusion attention backend SDPA")
                return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.info("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        if flash_attn_supported:
            logger.info("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.info("Defaulting to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return True

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("cuda")
        return torch.device("cuda", local_rank)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.cuda.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        return torch.version.cuda

    @classmethod
    def synchronize(cls) -> None:
        torch.cuda.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.cuda.mem_get_info(device)
        return free

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)
