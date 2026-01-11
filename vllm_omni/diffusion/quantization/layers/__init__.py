from torch import nn

# TODO: Import DiTQuantLinear once dit_linear.py is finished.
# from .dit_linear import DiTQuantLinear
from ..config import DiTQuantizationConfig


def get_linear_layer(
    in_features: int, out_features: int, bias: bool = True, quant_config: DiTQuantizationConfig | None = None
) -> nn.Module:
    """
    Factory function to create a Linear layer for Diffusion Transformers.
    """

    if quant_config is not None:
        # TODO: Add logic here to optionally inspect layer names if we want to support
        # granular skipping inside this factory (though usually handled in config).

        # TODO: Instantiate and return DiTQuantLinear(in_features, out_features, bias, quant_config)
        # return DiTQuantLinear(...)
        pass

    # TODO: Fallback to standard PyTorch Linear if no config is present.
    return nn.Linear(in_features, out_features, bias=bias)


__all__ = [
    # "DiTQuantLinear",
    "get_linear_layer",
]
