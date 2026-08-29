# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig

from .config import (
    DEFAULT_OFFLOAD_COMPONENTS,
    DIT_COMPONENT,
    OFFLOAD_COMPONENTS,
    TEXT_ENCODER_COMPONENT,
    DLOTransfer,
    OffloadStrategy,
    get_diffusion_offload_config,
    parse_dlo_transfer,
    parse_offload_components,
    resolve_offload_strategy,
    validate_offload_host_registration,
)
from .offload_plan import OffloadPlan

logger = init_logger(__name__)


@runtime_checkable
class SupportsModelCpuOffload(Protocol):
    """Pipeline-owned lifecycle for model-level CPU offload.

    Pipelines with non-forward component entry points (for example VAE
    ``decode_latent`` methods) need to activate those stages explicitly, so
    generic forward-hook discovery cannot manage their full lifecycle.
    """

    def enable_omni_model_cpu_offload(
        self,
        *,
        device: torch.device,
        pin_memory: bool,
        use_hsdp: bool,
        offload_components: frozenset[str] | None = None,
    ) -> None: ...

    def disable_omni_model_cpu_offload(self) -> None: ...


@dataclass
class OffloadConfig:
    strategy: OffloadStrategy
    pin_cpu_memory: bool = True
    use_hsdp: bool = False
    dp_size: int = 1  # derived from parallel_config, not user-configurable
    # True: add DP sharding + AllGather. False: stream complete rank-local
    # blocks from the loader-selected host backing with H2D only.
    dlo_use_allgather: bool = True
    dlo_resident_layers: int = 0  # leading DiT layers kept on device
    # Optional per-worker ceiling for registering an HWR mmap. Zero means no
    # additional ceiling; pin_cpu_memory controls whether registration is tried.
    dlo_host_registration_limit_gib: float = 0.0
    model_path: str | None = None  # checkpoint path for mmap weight loading
    components: frozenset[str] = DEFAULT_OFFLOAD_COMPONENTS
    components_explicit: bool = False
    dlo_transfers: dict[str, DLOTransfer] | None = None

    def __post_init__(self) -> None:
        self.components = parse_offload_components(self.components)
        self.dlo_transfers = parse_dlo_transfer(
            self.dlo_transfers,
            legacy_use_allgather=self.dlo_use_allgather,
        )
        # Preserve the old field as the DiT transfer compatibility view.
        self.dlo_use_allgather = self.uses_allgather(DIT_COMPONENT)

    def offloads(self, component: str) -> bool:
        return component in self.components

    def transfer_for(self, component: str) -> DLOTransfer:
        if self.dlo_transfers is None:
            raise RuntimeError("DLO transfers were not initialized")
        try:
            return self.dlo_transfers[component]
        except KeyError as exc:
            raise ValueError(f"Unknown DLO component {component!r}") from exc

    def uses_allgather(self, component: str) -> bool:
        return self.transfer_for(component) is DLOTransfer.ALLGATHER

    def offloads_encoder(self, name: str, plan: OffloadPlan | None = None) -> bool:
        """Return whether the selector covers a discovered encoder path.

        Plans declare non-standard encoder names explicitly. The name-based
        fallback preserves compatibility with pipelines that predate OffloadPlan.
        """
        declared_component = None if plan is None else plan.encoder_component_types.get(name)
        if declared_component is not None:
            if declared_component != TEXT_ENCODER_COMPONENT:
                raise ValueError(f"OffloadPlan maps encoder {name!r} to unknown component {declared_component!r}")
            return self.offloads(declared_component)

        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name.startswith(TEXT_ENCODER_COMPONENT) or leaf_name.endswith(TEXT_ENCODER_COMPONENT):
            return self.offloads(TEXT_ENCODER_COMPONENT)
        return False

    def should_offload_encoder(self, name: str, plan: OffloadPlan | None = None) -> bool:
        """Apply explicit selection while preserving the legacy encoder topology."""
        return not self.components_explicit or self.offloads_encoder(name, plan)

    @classmethod
    def from_od_config(cls, od_config: OmniDiffusionConfig) -> "OffloadConfig":
        """Extract and validate offload settings from OmniDiffusionConfig.

        ``diffusion_offload_config`` is the canonical public selector. The
        historical ``enable_*_offload`` booleans remain compatibility aliases;
        ambiguous combinations fail instead of using silent precedence.

        The ``dp_size`` is automatically derived from ``parallel_config`` —
        it is NOT a user-configurable parameter. The distributed layerwise
        offload works with whatever DP/SP parallelism is already set up.

        Args:
            od_config: OmniDiffusionConfig with offload settings

        Returns:
            OffloadConfig with validated settings
        """
        strategy = resolve_offload_strategy(od_config)
        public_config = get_diffusion_offload_config(od_config)
        enable_distributed_layerwise_offload = strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
        pin_cpu_memory = (
            public_config.pin_memory
            if public_config is not None and public_config.pin_memory is not None
            else getattr(od_config, "pin_cpu_memory", True)
        )

        parallel_config = getattr(od_config, "parallel_config", None)
        use_hsdp = getattr(parallel_config, "use_hsdp", False) if parallel_config else False
        # Derive dp_size from parallel_config — not user-configurable.
        # The offload adapts to whatever DP/SP is already configured.
        dp_size = 1
        if parallel_config is not None:
            dp_size = getattr(parallel_config, "data_parallel_size", 1)
            # HSDP shard and replica sizes determine the effective group size.
            hsdp_shard_size = getattr(parallel_config, "hsdp_shard_size", -1) if use_hsdp else -1
            hsdp_replicate_size = getattr(parallel_config, "hsdp_replicate_size", 1) if use_hsdp else 1
            if use_hsdp and hsdp_shard_size > 0:
                dp_size = hsdp_shard_size * hsdp_replicate_size

            # When there is no DP but SP > 1, shard weights across SP ranks.
            # AllGather reconstructs full weights per layer; each rank then
            # computes on its SP portion of the sequence.  This gives N×
            # compute parallelism with 1/N H2D transfer, reusing the exact
            # same AllGather code path — only the process group changes.
            if dp_size <= 1:
                sp_size = getattr(parallel_config, "sequence_parallel_size", 1)
                if sp_size and sp_size > 1:
                    dp_size = sp_size

        if public_config is not None:
            components = frozenset(public_config.components)
            dlo_transfers = {component: DLOTransfer.RANK_LOCAL for component in OFFLOAD_COMPONENTS}
            for component, layer_options in public_config.layer_options.items():
                dlo_transfers[component] = layer_options.weight_transfer or DLOTransfer.RANK_LOCAL
            dit_config = public_config.layer_options.get(DIT_COMPONENT)
            dlo_resident_layers = 0 if dit_config is None else dit_config.resident_layers
            components_explicit = True
        else:
            components = DEFAULT_OFFLOAD_COMPONENTS
            # The compatibility scalar controlled DiT sharding only. Auxiliary
            # components in the legacy topology were always streamed from
            # each rank's loader-produced weights, so preserve that behavior
            # instead of applying the scalar to the text encoder as well.
            dlo_transfers = {
                DIT_COMPONENT: (
                    DLOTransfer.ALLGATHER if getattr(od_config, "dlo_use_allgather", True) else DLOTransfer.RANK_LOCAL
                ),
                TEXT_ENCODER_COMPONENT: DLOTransfer.RANK_LOCAL,
            }
            dlo_resident_layers = int(getattr(od_config, "dlo_resident_layers", 0))
            components_explicit = False

        dit_uses_allgather = dlo_transfers[DIT_COMPONENT] is DLOTransfer.ALLGATHER
        dlo_host_registration_limit_gib = validate_offload_host_registration(od_config)

        if enable_distributed_layerwise_offload and all(
            transfer is DLOTransfer.RANK_LOCAL
            for component, transfer in dlo_transfers.items()
            if component in components
        ):
            logger.info(
                "Distributed layerwise offload: all selected components use "
                "rank-local transfer (no DLO shard or AllGather)"
            )

        # HSDP already shards parameters into DTensors.  Running distributed
        # layerwise offload on top would shard each to_local() again, producing
        # incorrect reconstruction after AllGather.  Reject this combination.
        if (
            enable_distributed_layerwise_offload
            and use_hsdp
            and any(dlo_transfers[component] is DLOTransfer.ALLGATHER for component in components)
        ):
            raise ValueError(
                "Distributed layerwise offload with AllGather is incompatible with "
                "HSDP: HSDP parameters are already sharded DTensors, and the offloader "
                "would double-shard them. Set weight_transfer='rank-local' for the affected "
                "component in diffusion_offload_config, or disable HSDP."
            )

        return cls(
            strategy=strategy,
            pin_cpu_memory=pin_cpu_memory,
            use_hsdp=use_hsdp,
            dp_size=dp_size,
            dlo_use_allgather=dit_uses_allgather,
            dlo_resident_layers=dlo_resident_layers,
            dlo_host_registration_limit_gib=dlo_host_registration_limit_gib,
            model_path=getattr(od_config, "model", None),
            components=components,
            components_explicit=components_explicit,
            dlo_transfers=dlo_transfers,
        )


class OffloadBackend(ABC):
    """Base class for CPU offload backends"""

    def __init__(self, config: OffloadConfig, device: torch.device):
        self.config = config
        self.device = device
        self.enabled = False

    @abstractmethod
    def enable(self, pipeline: nn.Module) -> None:
        """Enable offloading on the pipeline.

        Discovers modules, moves them to appropriate devices, and
        registers forward hooks for swapping/prefetching.

        Args:
            pipeline: Diffusion pipeline model (e.g., Wan22Pipeline)
        """
        raise NotImplementedError

    @abstractmethod
    def disable(self) -> None:
        """Disable offloading and cleanup resources.

        Removes all registered hooks. Does NOT move modules back to
        original devices (caller responsible for that).
        """
        raise NotImplementedError

    def shutdown(self) -> None:
        """Release backend resources during final process teardown.

        Backends that support a later re-enable may override this to avoid
        rebuilding ordinary model weights that the exiting process will never
        use. The default preserves the regular disable behavior.
        """
        self.disable()

    def is_enabled(self) -> bool:
        return self.enabled
