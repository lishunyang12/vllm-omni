# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public diffusion CPU-offload configuration helpers."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

DIT_COMPONENT = "dit"
TEXT_ENCODER_COMPONENT = "text_encoder"
OFFLOAD_COMPONENTS = frozenset({DIT_COMPONENT, TEXT_ENCODER_COMPONENT})
DEFAULT_OFFLOAD_COMPONENTS = frozenset({DIT_COMPONENT})


class OffloadMode(str, Enum):
    """User-facing offload granularity."""

    MODULE = "module"
    LAYER = "layer"


class OffloadStrategy(str, Enum):
    """Resolved internal backend strategy."""

    NONE = "none"
    MODEL_LEVEL = "model_level"
    LAYER_WISE = "layer_wise"
    DISTRIBUTED_LAYER_WISE = "distributed_layer_wise"


class DLOTransfer(str, Enum):
    """How one component's next block reaches the device."""

    ALLGATHER = "allgather"
    RANK_LOCAL = "rank-local"


@dataclass(frozen=True)
class LayerOffloadOptions:
    """Layer-mode settings for one selected diffusion component."""

    weight_transfer: DLOTransfer | None = None
    resident_layers: int = 0


@dataclass(frozen=True)
class DiffusionOffloadConfig:
    """Validated user-facing diffusion offload configuration."""

    mode: OffloadMode
    components: frozenset[str]
    layer_options: dict[str, LayerOffloadOptions]
    pin_memory: bool | None = None


_LEGACY_STRATEGY_FIELDS = {
    "enable_cpu_offload": OffloadStrategy.MODEL_LEVEL,
    "enable_layerwise_offload": OffloadStrategy.LAYER_WISE,
    "enable_distributed_layerwise_offload": OffloadStrategy.DISTRIBUTED_LAYER_WISE,
}
_LEGACY_STRATEGY_PRIORITY = (
    OffloadStrategy.DISTRIBUTED_LAYER_WISE,
    OffloadStrategy.LAYER_WISE,
    OffloadStrategy.MODEL_LEVEL,
)


def _parse_mode(value: Any) -> OffloadMode:
    if isinstance(value, OffloadMode):
        return value
    if not isinstance(value, str):
        raise TypeError(f"diffusion_offload_config.mode must be a string, got {type(value).__name__}")
    try:
        return OffloadMode(value)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in OffloadMode)
        raise ValueError(f"Unknown diffusion offload mode {value!r}; choose from: {choices}") from exc


def _parse_transfer(value: Any) -> DLOTransfer:
    if isinstance(value, DLOTransfer):
        return value
    if not isinstance(value, str):
        raise TypeError(f"offload transfer must be a string, got {type(value).__name__}")
    try:
        return DLOTransfer(value)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in DLOTransfer)
        raise ValueError(f"Unknown offload transfer {value!r}; choose from: {choices}") from exc


def _parse_layer_options(component: str, value: Any) -> LayerOffloadOptions:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"diffusion_offload_config.layer_options[{component!r}] must be a mapping, got {type(value).__name__}"
        )
    unknown = sorted(set(value) - {"weight_transfer", "resident_layers"})
    if unknown:
        raise ValueError(
            f"Unknown diffusion offload setting(s) for {component}: {', '.join(unknown)}; "
            "choose from: weight_transfer, resident_layers"
        )
    weight_transfer = _parse_transfer(value["weight_transfer"]) if "weight_transfer" in value else None
    resident_layers = value.get("resident_layers", 0)

    if type(resident_layers) is not int or resident_layers < 0:
        raise ValueError(f"resident_layers for {component} must be a non-negative integer")
    parsed = LayerOffloadOptions(
        weight_transfer=weight_transfer,
        resident_layers=resident_layers,
    )
    if component != DIT_COMPONENT and parsed.resident_layers:
        raise ValueError("resident_layers currently supports only the 'dit' component")
    return parsed


def parse_diffusion_offload_config(value: Any) -> DiffusionOffloadConfig | None:
    """Validate the compact ``diffusion_offload_config`` public API."""
    if value is None:
        return None
    if isinstance(value, DiffusionOffloadConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"diffusion_offload_config must be a mapping, got {type(value).__name__}")

    unknown = sorted(set(value) - {"mode", "components", "layer_options", "pin_memory"})
    if unknown:
        raise ValueError(
            f"Unknown diffusion_offload_config field(s): {', '.join(unknown)}; "
            "choose from: mode, components, layer_options, pin_memory"
        )
    if "mode" not in value:
        raise ValueError("diffusion_offload_config requires 'mode'")
    if "components" not in value:
        raise ValueError("diffusion_offload_config requires 'components'")

    mode = _parse_mode(value["mode"])
    raw_components = value["components"]
    if isinstance(raw_components, (str, Mapping)) or not isinstance(raw_components, Collection):
        raise TypeError("diffusion_offload_config.components must be a non-empty list of component names")
    components = parse_offload_components(raw_components)

    raw_layer_options = value.get("layer_options", {})
    if not isinstance(raw_layer_options, Mapping):
        raise TypeError("diffusion_offload_config.layer_options must be a mapping")
    if any(not isinstance(component, str) for component in raw_layer_options):
        raise TypeError("diffusion_offload_config.layer_options keys must be component names")
    unselected_options = sorted(set(raw_layer_options) - components)
    if unselected_options:
        raise ValueError(
            "diffusion_offload_config.layer_options requires selecting the same component(s): "
            + ", ".join(unselected_options)
        )
    layer_options = {
        component: _parse_layer_options(component, raw_layer_options.get(component, {})) for component in components
    }

    pin_memory = value.get("pin_memory")
    if pin_memory is not None and type(pin_memory) is not bool:
        raise TypeError("diffusion_offload_config.pin_memory must be a bool")

    if mode is OffloadMode.MODULE and raw_layer_options:
        raise ValueError(
            "diffusion_offload_config.layer_options requires mode='layer'; "
            f"configured for: {', '.join(sorted(raw_layer_options))}"
        )
    dit = layer_options.get(DIT_COMPONENT)
    if dit is not None and dit.resident_layers and dit.weight_transfer is DLOTransfer.ALLGATHER:
        raise ValueError("resident_layers requires dit.weight_transfer='rank-local'")

    return DiffusionOffloadConfig(
        mode=mode,
        components=components,
        layer_options=layer_options,
        pin_memory=pin_memory,
    )


def get_diffusion_offload_config(config: Any) -> DiffusionOffloadConfig | None:
    """Read and validate the canonical compact config."""
    return parse_diffusion_offload_config(getattr(config, "diffusion_offload_config", None))


def _legacy_strategy(config: Any) -> OffloadStrategy:
    enabled_aliases = [
        (field, strategy) for field, strategy in _LEGACY_STRATEGY_FIELDS.items() if bool(getattr(config, field, False))
    ]
    if not enabled_aliases:
        return OffloadStrategy.NONE

    strategy = next(
        candidate
        for candidate in _LEGACY_STRATEGY_PRIORITY
        if any(enabled_strategy is candidate for _, enabled_strategy in enabled_aliases)
    )
    return strategy


def _public_strategy(config: Any, public: DiffusionOffloadConfig) -> OffloadStrategy:
    hwr_enabled = getattr(config, "host_weight_runtime_mode", "disabled") != "disabled"
    if hwr_enabled:
        raise ValueError("diffusion_offload_config cannot be combined with Host Weight Runtime")
    if public.mode is OffloadMode.MODULE:
        return OffloadStrategy.MODEL_LEVEL

    needs_distributed_backend = any(
        settings.weight_transfer is DLOTransfer.ALLGATHER or settings.resident_layers
        for settings in public.layer_options.values()
    )
    return OffloadStrategy.DISTRIBUTED_LAYER_WISE if needs_distributed_backend else OffloadStrategy.LAYER_WISE


def _validate_legacy_layer_options(config: Any, public: DiffusionOffloadConfig | None) -> None:
    """Reject ambiguous or invalid compatibility DLO tuning before loading."""
    resident_layers = getattr(config, "dlo_resident_layers", 0)
    if type(resident_layers) is not int or resident_layers < 0:
        raise ValueError(f"dlo_resident_layers must be a non-negative integer, got {resident_layers!r}")

    if public is not None:
        conflicting_fields = []
        if getattr(config, "dlo_use_allgather", True) is not True:
            conflicting_fields.append("dlo_use_allgather")
        if resident_layers:
            conflicting_fields.append("dlo_resident_layers")
        if float(getattr(config, "dlo_host_registration_limit_gib", 0.0)) > 0:
            conflicting_fields.append("dlo_host_registration_limit_gib")
        if conflicting_fields:
            raise ValueError(
                "diffusion_offload_config cannot be combined with legacy DLO option(s): "
                + ", ".join(conflicting_fields)
            )
        return

    if resident_layers and bool(getattr(config, "dlo_use_allgather", True)):
        raise ValueError(
            "dlo_resident_layers requires the DiT DLO transfer to be rank-local; set dlo_use_allgather=False"
        )


def resolve_offload_strategy(config: Any) -> OffloadStrategy:
    """Resolve the compact config or compatibility boolean entry points."""
    public = get_diffusion_offload_config(config)
    _validate_legacy_layer_options(config, public)
    legacy = _legacy_strategy(config)
    if public is None:
        return legacy

    strategy = _public_strategy(config, public)
    if legacy is not OffloadStrategy.NONE and legacy is not strategy:
        raise ValueError("diffusion_offload_config cannot be combined with legacy enable_*_offload flags")
    return strategy


def materialize_legacy_offload_flags(config: Any) -> OffloadStrategy:
    """Keep existing strategy readers working after resolving the compact API."""
    strategy = resolve_offload_strategy(config)
    public = get_diffusion_offload_config(config)
    for field, field_strategy in _LEGACY_STRATEGY_FIELDS.items():
        setattr(config, field, strategy is field_strategy)
    if public is not None and public.pin_memory is not None:
        setattr(config, "pin_cpu_memory", public.pin_memory)
    return strategy


def parse_offload_components(value: Collection[str]) -> frozenset[str]:
    """Validate an internal component collection."""
    if isinstance(value, (str, Mapping)) or not isinstance(value, Collection):
        raise TypeError(f"offload components must be a collection of names, got {type(value).__name__}")
    if any(not isinstance(item, str) for item in value):
        raise TypeError("offload component entries must be strings")
    components = [item.strip() for item in value]
    if not components or any(not item for item in components):
        raise ValueError("offload components must not be empty")
    unknown = sorted(set(components) - OFFLOAD_COMPONENTS)
    if unknown:
        choices = ", ".join(sorted(OFFLOAD_COMPONENTS))
        raise ValueError(f"Unknown diffusion offload component(s): {', '.join(unknown)}; choose from: {choices}")
    return frozenset(components)


def parse_dlo_transfer(
    value: Mapping[str, str | DLOTransfer] | None,
    *,
    legacy_use_allgather: bool = True,
) -> dict[str, DLOTransfer]:
    """Resolve an internal per-component transfer mapping."""
    fallback = DLOTransfer.ALLGATHER if legacy_use_allgather else DLOTransfer.RANK_LOCAL
    resolved = {component: fallback for component in OFFLOAD_COMPONENTS}
    if value is None:
        return resolved
    if not isinstance(value, Mapping):
        raise TypeError(f"offload transfers must be a mapping, got {type(value).__name__}")
    for component, raw_transfer in value.items():
        if component not in OFFLOAD_COMPONENTS:
            choices = ", ".join(sorted(OFFLOAD_COMPONENTS))
            raise ValueError(f"Unknown offload transfer component {component!r}; choose from: {choices}")
        resolved[component] = _parse_transfer(raw_transfer)
    return resolved


def component_uses_allgather(config: Any, component: str = DIT_COMPONENT) -> bool:
    """Return whether one selected component uses AllGather transport."""
    public = get_diffusion_offload_config(config)
    if public is not None:
        try:
            settings = public.layer_options[component]
        except KeyError as exc:
            raise ValueError(f"Offload component {component!r} is not selected") from exc
        return settings.weight_transfer is DLOTransfer.ALLGATHER
    if component not in OFFLOAD_COMPONENTS:
        raise ValueError(f"Unknown offload component {component!r}")
    # The legacy scalar is a DiT-only compatibility view; legacy
    # auxiliary component hooks always use rank-local transfer.
    return component == DIT_COMPONENT and bool(getattr(config, "dlo_use_allgather", True))


def selected_offload_components(config: Any) -> frozenset[str]:
    """Resolve selected components while preserving legacy topology defaults."""
    public = get_diffusion_offload_config(config)
    if public is not None:
        return frozenset(public.components)
    return DEFAULT_OFFLOAD_COMPONENTS


def should_offload_component(config: Any, component: str) -> bool:
    """Return whether an active layer policy selects ``component``."""
    if component not in OFFLOAD_COMPONENTS:
        raise ValueError(f"Unknown offload component: {component}")
    if resolve_offload_strategy(config) not in {
        OffloadStrategy.LAYER_WISE,
        OffloadStrategy.DISTRIBUTED_LAYER_WISE,
    }:
        return False
    return component in selected_offload_components(config)


def any_selected_component_uses_allgather(config: Any) -> bool:
    """Return whether an enabled layer backend requires weight collectives."""
    if resolve_offload_strategy(config) is not OffloadStrategy.DISTRIBUTED_LAYER_WISE:
        return False
    return any(component_uses_allgather(config, component) for component in selected_offload_components(config))
