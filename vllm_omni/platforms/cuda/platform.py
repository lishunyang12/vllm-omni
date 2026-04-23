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

        # Check compute capability for Flash Attention support.
        # FA requires sm_80+. Blackwell (sm_10x/sm_12x) only works with FA builds
        # that include the Blackwell CUTE kernel — plain FA2 will crash there.
        #
        # Known Blackwell SKUs:
        #   sm_100 = B200 / GB200 (datacenter)
        #   sm_103 = B300 / GB300 (Blackwell Ultra)
        #   sm_120 = RTX Pro 6000, RTX 50-series (consumer)
        #   sm_121 = consumer Blackwell refresh
        _known_blackwell_sms = {(10, 0), (10, 3), (12, 0), (12, 1)}
        compute_capability = cls.get_device_capability()
        compute_supported = False
        is_blackwell = False
        sm_str = ""
        if compute_capability is not None:
            major, minor = compute_capability
            capability = major * 10 + minor
            compute_supported = capability >= 80
            sm_str = f"sm_{major}{minor}"
            # Accept major in {10, 11, 12} to cover future Blackwell refreshes.
            is_blackwell = major in (10, 11, 12)
            if is_blackwell and (major, minor) not in _known_blackwell_sms:
                logger.info(
                    "Detected Blackwell-class GPU %s (untested variant); routing to CUDNN_ATTN with SDPA fallback.",
                    sm_str,
                )

        # Check if FA packages are available
        packages_info = PACKAGES_CHECKER.get_packages_info()
        packages_available = packages_info.get("has_flash_attn", False)

        # Both compute capability and packages must be available for FA
        flash_attn_supported = compute_supported and packages_available

        # cuDNN 9.5+ ships Blackwell FMHA kernels. If the runtime is older,
        # the CUDNN_ATTN default would still work via internal fallback but
        # without the tuned Blackwell path, so we skip routing there.
        cudnn_version = torch.backends.cudnn.version() or 0
        cudnn_blackwell_ready = cudnn_version >= 90500

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            if backend_upper == "FLASH_ATTN" and not flash_attn_supported:
                if not compute_supported:
                    logger.warning(
                        "Flash Attention requires GPU with compute capability >= 8.0. "
                        "Falling back to TORCH_SDPA backend."
                    )
                elif not packages_available:
                    logger.warning("Flash Attention packages not available. Falling back to TORCH_SDPA backend.")
                logger.info("Defaulting to diffusion attention backend SDPA")
                return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.info("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        if is_blackwell and cudnn_blackwell_ready:
            logger.info(
                "Defaulting to diffusion attention backend CUDNN_ATTN (Blackwell %s, cuDNN %d)",
                sm_str,
                cudnn_version,
            )
            return DiffusionAttentionBackendEnum.CUDNN_ATTN.get_path()

        if is_blackwell and not cudnn_blackwell_ready:
            logger.warning(
                "Detected Blackwell %s but cuDNN %d < 9.5 — no tuned Blackwell FMHA. "
                "Falling through to FLASH_ATTN / SDPA.",
                sm_str,
                cudnn_version,
            )

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
