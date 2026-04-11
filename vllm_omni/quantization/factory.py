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


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


class FluxNvFp4LinearMethod(ModelOptNvFp4LinearMethod):
    """ModelOpt NVFP4 LinearMethod variant for NVIDIA/BFL FLUX.2 checkpoints.

    NVIDIA's BFL-released FLUX.2 NVFP4 safetensors need two layout fixes on
    top of the upstream LLM ModelOpt path before the CUTLASS TMA kernel
    produces correct results:

    1. **FP4 nibble swap** — BFL packs the high/low nibble in the opposite
       byte order to what upstream expects. Without the swap the kernel
       reads bogus FP4 values (pure noise image).

    2. **weight_scale padding + blockwise swizzle** — the CUTLASS TMA
       kernel wants the per-block scales in a specific interleaved layout:
       reshape ``[M, K] → [M/128, 4, 32, K/4, 4]`` then permute axes
       ``(0, 3, 2, 1, 4)`` (i.e. swap the ``M``-inner and ``K``-major
       groups). Without this, CUTLASS silently produces wrong results and
       accuracy drops ~5% cos-sim vs cuDNN.

    Both are confirmed by SGLang's diffusion-side fixes:
    - sgl-project/sglang#20137 (nibble swap)
    - sgl-project/sglang#22064 (scale swizzle — the 5% cos-sim fix)

    We apply both *before* delegating to upstream's
    ``process_weights_after_loading``. Upstream's kernel.process then
    reformats for its dispatch, expecting the standardized layout that our
    pre-swap produces. Everything downstream (alpha, input_global_scale,
    apply(), kernel dispatch) is inherited.

    Defined at module level so ``multiprocessing.spawn`` can pickle it by
    qualified name when the ``OmniDiffusionConfig`` is sent to a worker
    subprocess.
    """

    def process_weights_after_loading(self, layer):  # type: ignore[override]
        import torch

        # -- (1) FP4 nibble swap on the packed weight bytes ---------------
        w = layer.weight.data
        layer.weight.data = ((w >> 4) | ((w & 0x0F) << 4)).to(torch.uint8).contiguous()

        # -- (2) weight_scale pad + blockwise swizzle for CUTLASS TMA ----
        scales = layer.weight_scale.data
        if scales.ndim != 2:
            raise RuntimeError(f"FluxNvFp4LinearMethod expects weight_scale of ndim 2, got {scales.shape}")
        M, K = scales.shape
        M_padded = _round_up(M, 128)
        K_padded = _round_up(K, 4)
        if M_padded != M or K_padded != K:
            padded = torch.zeros((M_padded, K_padded), dtype=scales.dtype, device=scales.device)
            padded[:M, :K] = scales
        else:
            padded = scales
        # Blockwise interleave. Ported from sgl-project/sglang#22064.
        # Reshape [M, K] → [M/128, 4, 32, K/4, 4], permute to [M/128, K/4, 32, 4, 4],
        # then flatten back to [M, K]. This mirrors the layout the CUTLASS
        # TMA kernel wants the scales in.
        swizzled = padded.reshape(M_padded // 128, 4, 32, K_padded // 4, 4)
        swizzled = swizzled.permute(0, 3, 2, 1, 4).contiguous()
        layer.weight_scale.data = swizzled.reshape(M_padded, K_padded).contiguous()

        # -- (3) Delegate to upstream for global_scale / alpha / kernel ---
        super().process_weights_after_loading(layer)


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
