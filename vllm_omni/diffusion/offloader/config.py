# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public diffusion CPU-offload configuration helpers."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

DIT_COMPONENT = "dit"
TEXT_ENCODER_COMPONENT = "text_encoder"
OFFLOAD_COMPONENTS = frozenset({DIT_COMPONENT, TEXT_ENCODER_COMPONENT})
DEFAULT_OFFLOAD_COMPONENTS = frozenset({DIT_COMPONENT})
DLO_COMPONENTS = OFFLOAD_COMPONENTS

LEGACY_OFFLOAD_REMOVAL_VERSION = "0.30"
# Reserved transport marker stored in the existing ``extras`` mapping so a
# structured config round trip can distinguish derived aliases from user input.
_MATERIALIZED_EXTRAS_KEY = "_diffusion_offload_flags_materialized"


class OffloadMode(str, Enum):
    """User-facing offload granularity."""

    MODULE = "module"
    LAYER = "layer"


class OffloadStrategy(str, Enum):
    """Resolved internal backend strategy."""

    NONE = "none"
    MODEL_LEVEL = "model"
    LAYER_WISE = "layerwise"
    DISTRIBUTED_LAYER_WISE = "distributed-layerwise"


class DLOTransfer(str, Enum):
    """How one component's next block reaches the device."""

    ALLGATHER = "allgather"
    RANK_LOCAL = "rank-local"


@dataclass(frozen=True)
class LayerOffloadOptions:
    """Layer-mode settings for one selected diffusion component."""

    transfer: DLOTransfer | None = None
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


def _legacy_flags_materialized(config: Any) -> bool:
    """Read materialization provenance across structured config transport."""
    if bool(getattr(config, _MATERIALIZED_EXTRAS_KEY, False)):
        return True
    extras = getattr(config, "extras", None)
    return isinstance(extras, Mapping) and bool(extras.get(_MATERIALIZED_EXTRAS_KEY, False))


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
    if isinstance(value, LayerOffloadOptions):
        parsed = value
    else:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"diffusion_offload_config.layer_options[{component!r}] must be a mapping, got {type(value).__name__}"
            )
        unknown = sorted(set(value) - {"transfer", "resident_layers"})
        if unknown:
            raise ValueError(
                f"Unknown diffusion offload setting(s) for {component}: {', '.join(unknown)}; "
                "choose from: transfer, resident_layers"
            )
        transfer = _parse_transfer(value["transfer"]) if "transfer" in value else None
        resident_layers = value.get("resident_layers", 0)
        if type(resident_layers) is not int or resident_layers < 0:
            raise ValueError(f"resident_layers for {component} must be a non-negative integer")
        parsed = LayerOffloadOptions(
            transfer=transfer,
            resident_layers=resident_layers,
        )

    transfer = _parse_transfer(parsed.transfer) if parsed.transfer is not None else None
    if type(parsed.resident_layers) is not int or parsed.resident_layers < 0:
        raise ValueError(f"resident_layers for {component} must be a non-negative integer")
    parsed = LayerOffloadOptions(
        transfer=transfer,
        resident_layers=parsed.resident_layers,
    )
    if component != DIT_COMPONENT and parsed.resident_layers:
        raise ValueError("resident_layers currently supports only the 'dit' component")
    return parsed


def parse_diffusion_offload_config(value: Any) -> DiffusionOffloadConfig | None:
    """Validate the compact ``diffusion_offload_config`` public API."""
    if value is None:
        return None
    if isinstance(value, DiffusionOffloadConfig):
        value = {
            "mode": value.mode,
            "components": value.components,
            "layer_options": value.layer_options,
            "pin_memory": value.pin_memory,
        }
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

    if mode is OffloadMode.MODULE:
        if raw_layer_options:
            raise ValueError(
                "diffusion_offload_config.layer_options requires mode='layer'; "
                f"configured for: {', '.join(sorted(raw_layer_options))}"
            )
    dit = layer_options.get(DIT_COMPONENT)
    if dit is not None and dit.resident_layers and dit.transfer is DLOTransfer.ALLGATHER:
        raise ValueError("resident_layers requires dit.transfer='rank-local'")

    return DiffusionOffloadConfig(
        mode=mode,
        components=components,
        layer_options=layer_options,
        pin_memory=pin_memory,
    )


def get_diffusion_offload_config(config: Any) -> DiffusionOffloadConfig | None:
    """Read and validate the compact public config from a diffusion-like object."""
    return parse_diffusion_offload_config(getattr(config, "diffusion_offload_config", None))


def _legacy_strategy(config: Any, *, warn: bool) -> OffloadStrategy:
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
    if warn and not _legacy_flags_materialized(config):
        fields = ", ".join(field for field, _ in enabled_aliases)
        warnings.warn(
            f"{fields} {'are' if len(enabled_aliases) > 1 else 'is'} deprecated and will be removed in "
            f"v{LEGACY_OFFLOAD_REMOVAL_VERSION}; "
            "use diffusion_offload_config instead",
            FutureWarning,
            stacklevel=3,
        )
    return strategy


def _public_strategy(config: Any, public: DiffusionOffloadConfig) -> OffloadStrategy:
    if public.mode is OffloadMode.MODULE:
        return OffloadStrategy.MODEL_LEVEL

    needs_distributed_backend = any(
        settings.transfer is DLOTransfer.ALLGATHER or settings.resident_layers
        for settings in public.layer_options.values()
    )
    needs_distributed_backend = needs_distributed_backend or (
        getattr(config, "host_weight_runtime_mode", "disabled") != "disabled"
        or float(getattr(config, "dlo_host_registration_limit_gib", 0.0)) > 0
    )
    return OffloadStrategy.DISTRIBUTED_LAYER_WISE if needs_distributed_backend else OffloadStrategy.LAYER_WISE


def _validate_legacy_layer_options(config: Any, public: DiffusionOffloadConfig | None) -> None:
    """Reject ambiguous or invalid deprecated DLO tuning before loading."""
    resident_layers = getattr(config, "dlo_resident_layers", 0)
    if type(resident_layers) is not int or resident_layers < 0:
        raise ValueError(f"dlo_resident_layers must be a non-negative integer, got {resident_layers!r}")

    materialized = _legacy_flags_materialized(config)
    if public is not None:
        if materialized:
            return
        conflicting_fields = []
        if getattr(config, "dlo_use_allgather", True) is not True:
            conflicting_fields.append("dlo_use_allgather")
        if resident_layers:
            conflicting_fields.append("dlo_resident_layers")
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
    """Resolve the compact config or the deprecated boolean entry points."""
    public = get_diffusion_offload_config(config)
    _validate_legacy_layer_options(config, public)
    legacy = _legacy_strategy(config, warn=public is None)
    if public is None:
        return legacy

    strategy = _public_strategy(config, public)
    if legacy is not OffloadStrategy.NONE and (not _legacy_flags_materialized(config) or legacy is not strategy):
        raise ValueError("diffusion_offload_config cannot be combined with legacy enable_*_offload flags")
    return strategy


def uses_offload_strategy(config: Any, *strategies: OffloadStrategy) -> bool:
    """Return whether a diffusion-like config resolves to one of ``strategies``."""
    return resolve_offload_strategy(config) in strategies


def materialize_legacy_offload_flags(config: Any) -> OffloadStrategy:
    """Keep existing runtime readers working after resolving the compact API."""
    strategy = resolve_offload_strategy(config)
    public = get_diffusion_offload_config(config)
    for field, field_strategy in _LEGACY_STRATEGY_FIELDS.items():
        setattr(config, field, strategy is field_strategy)
    if public is not None:
        dit = public.layer_options.get(DIT_COMPONENT)
        setattr(
            config,
            "dlo_use_allgather",
            bool(dit is not None and dit.transfer is DLOTransfer.ALLGATHER),
        )
        setattr(config, "dlo_resident_layers", 0 if dit is None else dit.resident_layers)
        if public.pin_memory is not None:
            setattr(config, "pin_cpu_memory", public.pin_memory)
    setattr(config, _MATERIALIZED_EXTRAS_KEY, True)
    extras = getattr(config, "extras", None)
    if isinstance(extras, dict):
        extras[_MATERIALIZED_EXTRAS_KEY] = True
    return strategy


def parse_offload_components(value: str | Collection[str] | None) -> frozenset[str]:
    """Validate an internal component collection."""
    if value is None:
        return DEFAULT_OFFLOAD_COMPONENTS
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Collection):
        values = list(value)
    else:
        raise TypeError(f"offload components must be a string or collection, got {type(value).__name__}")
    if any(not isinstance(item, str) for item in values):
        raise TypeError("offload component entries must be strings")
    components = [item.strip() for item in values]
    if not components or any(not item for item in components):
        raise ValueError("offload components must not be empty")
    unknown = sorted(set(components) - OFFLOAD_COMPONENTS)
    if unknown:
        choices = ", ".join(sorted(OFFLOAD_COMPONENTS))
        raise ValueError(f"Unknown diffusion offload component(s): {', '.join(unknown)}; choose from: {choices}")
    return frozenset(components)


def parse_dlo_transfer(
    value: str | Mapping[str, str | DLOTransfer] | None,
    *,
    legacy_use_allgather: bool = True,
) -> dict[str, DLOTransfer]:
    """Resolve an internal scalar or per-component transfer mapping."""
    fallback = DLOTransfer.ALLGATHER if legacy_use_allgather else DLOTransfer.RANK_LOCAL
    resolved = {component: fallback for component in DLO_COMPONENTS}
    if value is None:
        return resolved
    if isinstance(value, str):
        transfer = _parse_transfer(value)
        return {component: transfer for component in DLO_COMPONENTS}
    if not isinstance(value, Mapping):
        raise TypeError(f"offload transfers must be a string or mapping, got {type(value).__name__}")
    for component, raw_transfer in value.items():
        if component not in DLO_COMPONENTS:
            choices = ", ".join(sorted(DLO_COMPONENTS))
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
        return settings.transfer is DLOTransfer.ALLGATHER
    if component not in DLO_COMPONENTS:
        raise ValueError(f"Unknown offload component {component!r}")
    return bool(getattr(config, "dlo_use_allgather", True))


def selected_offload_components(config: Any) -> frozenset[str]:
    """Resolve selected components while preserving legacy topology defaults."""
    public = get_diffusion_offload_config(config)
    if public is not None:
        return frozenset(public.components)
    return DEFAULT_OFFLOAD_COMPONENTS


def any_selected_component_uses_allgather(config: Any) -> bool:
    """Return whether an enabled layer backend requires weight collectives."""
    if not uses_offload_strategy(config, OffloadStrategy.DISTRIBUTED_LAYER_WISE):
        return False
    return any(component_uses_allgather(config, component) for component in selected_offload_components(config))
