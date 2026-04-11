# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for building quantization configs.

build_quant_config() delegates to vLLM's quantization registry.
The only extension point is _OVERRIDES for methods that need a
different QuantizationConfig subclass in the OMNI context (e.g. GGUF).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    get_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4LinearMethod,
)

from .component_config import ComponentQuantizationConfig

logger = init_logger(__name__)


class FluxNvFp4LinearMethod(ModelOptNvFp4LinearMethod):
    """ModelOpt NVFP4 LinearMethod variant for NVIDIA/BFL FLUX.2 checkpoints.

    NVIDIA's BFL-released FLUX.2 NVFP4 safetensors pack FP4 nibbles in the
    opposite byte order to what vLLM upstream's FlashInfer CUTLASS path
    expects for LLM ModelOpt checkpoints. Loading such a checkpoint through
    the stock path produces numerically-stable but semantically garbage
    output (pure noise images). This subclass adds a single per-byte nibble
    swap before delegating to the upstream
    ``process_weights_after_loading``. Everything else (weight shape,
    weight_scale padding, kernel dispatch, forward path) is inherited.

    Confirmed by sgl-project/sglang#20137, which added the same one-line
    workaround for the same checkpoint family.

    Defined at module level so ``multiprocessing.spawn`` can pickle it by
    qualified name when the ``OmniDiffusionConfig`` is sent to a worker
    subprocess.
    """

    # Log only the first few layers to avoid spamming for ~144 linears.
    _debug_log_budget: int = 4

    def process_weights_after_loading(self, layer):  # type: ignore[override]
        import torch

        w_before = layer.weight.data
        before_shape = tuple(w_before.shape)
        before_dtype = w_before.dtype
        # Sample a few bytes so we can see whether upstream's kernel re-layout
        # leaves our swap in place or rewrites the weight tensor.
        before_sample = w_before.flatten()[:8].tolist() if w_before.numel() >= 8 else w_before.flatten().tolist()

        # Swap high/low nibbles.
        swapped = ((w_before >> 4) | ((w_before & 0x0F) << 4)).to(torch.uint8).contiguous()
        layer.weight.data = swapped
        after_swap_sample = swapped.flatten()[:8].tolist() if swapped.numel() >= 8 else swapped.flatten().tolist()

        super().process_weights_after_loading(layer)

        w_after = layer.weight.data
        after_shape = tuple(w_after.shape)
        after_dtype = w_after.dtype
        after_sample = w_after.flatten()[:8].tolist() if w_after.numel() >= 8 else w_after.flatten().tolist()

        if FluxNvFp4LinearMethod._debug_log_budget > 0:
            FluxNvFp4LinearMethod._debug_log_budget -= 1
            logger.info(
                "[FluxNvFp4] process_weights_after_loading: prefix=%s\n"
                "  before swap:  shape=%s dtype=%s sample=%s\n"
                "  after swap:   sample=%s\n"
                "  after super:  shape=%s dtype=%s sample=%s",
                getattr(layer, "prefix", "<no-prefix>"),
                before_shape,
                before_dtype,
                before_sample,
                after_swap_sample,
                after_shape,
                after_dtype,
                after_sample,
            )


def _build_gguf(**kw: Any) -> QuantizationConfig:
    """Lazy import to avoid pulling in CUDA/pynvml at module load time."""
    from .gguf_config import DiffusionGGUFConfig

    return DiffusionGGUFConfig(**kw)


def _build_int8(**kw: Any) -> QuantizationConfig:
    """Lazy import for Int8 diffusion config (supports CUDA + NPU)."""
    from .int8_config import DiffusionInt8Config

    return DiffusionInt8Config(**kw)


def _build_inc(**kw: Any) -> QuantizationConfig:
    """Lazy import for INC/AutoRound config with checkpoint kwarg normalization."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    # Map checkpoint key 'bits' to INCConfig's 'weight_bits'
    if "bits" in kw and "weight_bits" not in kw:
        kw["weight_bits"] = kw.pop("bits")

    # Filter to only valid INCConfig params
    valid = set(inspect.signature(INCConfig.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in kw.items() if k in valid}
    return INCConfig(**filtered)


def _build_modelopt_nvfp4(**kw: Any) -> QuantizationConfig:
    """Lazy import for ModelOpt NVFP4 config.

    ``ModelOptNvFp4Config.__init__`` takes internal fields
    (``is_checkpoint_nvfp4_serialized``, ...) that are inconvenient to specify
    from a checkpoint's ``quantization_config`` block. The classmethod
    ``from_config`` accepts the friendlier on-disk schema
    (``{"quant_algo", "kv_cache_quant_algo", "exclude_modules", "group_size"}``)
    and normalizes it, so we route through that. Matches the pattern
    upstream vLLM uses when loading ModelOpt-quantized LLM checkpoints.
    """
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

    return ModelOptNvFp4Config.from_config({"quantization": kw})


def _build_modelopt_nvfp4_flux(**kw: Any) -> QuantizationConfig:
    """ModelOpt NVFP4 variant for NVIDIA/BFL FLUX.2 checkpoints.

    Takes the stock ``ModelOptNvFp4Config`` built via ``from_config``, then
    installs ``FluxNvFp4LinearMethod`` as the per-instance
    ``LinearMethodCls`` so our nibble-swap override runs on each quantized
    layer at load time. See ``FluxNvFp4LinearMethod`` for why the swap is
    needed.
    """
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

    config = ModelOptNvFp4Config.from_config({"quantization": kw})
    # Instance-level override — ModelOptQuantConfigBase.get_quant_method
    # looks up ``self.LinearMethodCls``, which resolves the instance
    # attribute before the class attribute.
    config.LinearMethodCls = FluxNvFp4LinearMethod  # type: ignore[assignment]
    return config


_OVERRIDES: dict[str, Callable[..., QuantizationConfig]] = {
    "gguf": _build_gguf,
    "int8": _build_int8,
    "inc": _build_inc,
    "auto-round": _build_inc,
    "modelopt_fp4": _build_modelopt_nvfp4,
    "modelopt_fp4_flux": _build_modelopt_nvfp4_flux,
}

SUPPORTED_QUANTIZATION_METHODS: list[str] = list(dict.fromkeys(QUANTIZATION_METHODS + list(_OVERRIDES.keys())))


def _build_single(method: str, **kwargs: Any) -> QuantizationConfig:
    """Build a single QuantizationConfig by method name.

    Resolution: _OVERRIDES first, then vLLM registry via from_config().
    """
    method = method.lower()

    if method in _OVERRIDES:
        return _OVERRIDES[method](**kwargs)

    if method not in QUANTIZATION_METHODS:
        raise ValueError(f"Unknown quantization method: {method!r}. Supported: {SUPPORTED_QUANTIZATION_METHODS}")

    config_cls = get_quantization_config(method)

    try:
        return config_cls(**kwargs)
    except TypeError:
        sig = inspect.signature(config_cls.__init__)
        raise TypeError(
            f"Cannot instantiate {config_cls.__name__} with kwargs {kwargs}. Expected signature: {sig}"
        ) from None


def _is_per_component_dict(spec: dict[str, Any]) -> bool:
    """Check if a dict describes per-component quantization.

    A per-component dict has no "method" key and all values are
    str, dict, or None. To avoid misdetecting a flat config with
    all-string values (e.g. {"activation_scheme": "static"}), we
    require at least one value to be None or a dict with "method".
    """
    if "method" in spec:
        return False
    if not all(isinstance(v, (dict, str, type(None))) for v in spec.values()):
        return False
    return any(v is None or (isinstance(v, dict) and "method" in v) for v in spec.values())


def _build_component_config(spec: dict[str, Any]) -> ComponentQuantizationConfig:
    """Build ComponentQuantizationConfig from a per-component dict."""
    component_configs: dict[str, QuantizationConfig | None] = {}
    default_config: QuantizationConfig | None = None

    for prefix, value in spec.items():
        if value is None:
            config = None
        elif isinstance(value, str):
            config = _build_single(value)
        elif isinstance(value, dict):
            value = dict(value)  # avoid mutating caller's dict
            method = value.pop("method", None)
            if method is None:
                raise ValueError(f"Component '{prefix}' config dict must have a 'method' key")
            config = _build_single(method, **value)
        else:
            raise TypeError(f"Component '{prefix}' config must be str, dict, or None, got {type(value).__name__}")

        if prefix == "default":
            default_config = config
        else:
            component_configs[prefix] = config

    logger.info(
        "Per-component quantization: %s",
        {k: (v.get_name() if v else None) for k, v in component_configs.items()},
    )
    return ComponentQuantizationConfig(component_configs, default_config)


def build_quant_config(
    spec: str | dict[str, Any] | QuantizationConfig | None,
    **kwargs: Any,
) -> QuantizationConfig | None:
    """Build a quantization config from a flexible specification.

    Args:
        spec: None/"none", method name str, dict with "method" key,
              per-component dict, or QuantizationConfig passthrough.
        **kwargs: Extra params merged with dict spec.
    """
    if spec is None:
        return None

    if isinstance(spec, QuantizationConfig):
        return spec

    if isinstance(spec, str):
        if spec.lower() == "none":
            return None
        logger.info("Building quantization config: %s", spec)
        return _build_single(spec, **kwargs)

    if isinstance(spec, Mapping):
        spec = dict(spec)

        if _is_per_component_dict(spec):
            return _build_component_config(spec)

        method = spec.pop("method", None)
        if method is None:
            raise ValueError(
                "Dict quantization config must have a 'method' key or "
                "be a per-component config with component prefixes as keys."
            )
        merged = {**spec, **kwargs}
        logger.info("Building quantization config: %s", method)
        return _build_single(method, **merged)

    raise TypeError(f"quantization config must be str, dict, QuantizationConfig, or None, got {type(spec).__name__}")
