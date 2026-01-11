from typing import Any

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class DiTQuantizationConfig(QuantizationConfig):
    """
    Configuration for Diffusion Transformer (DiT) Quantization.
    """

    def __init__(
        self,
        quant_method: str,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: list[str] | None = None,
    ):
        # TODO: Store the initialization parameters.
        # Ensure 'quant_method' is saved so we can route to the right kernel later.
        self.quant_method = quant_method

        # TODO: Define the default list of modules to skip if `modules_to_not_convert` is None.
        # Crucial: Must skip "t_embedder" (time embeddings) and "x_embedder" (patch embeddings)
        # to maintain diffusion stability.
        self.modules_to_not_convert = modules_to_not_convert or []
        pass

    def get_name(self) -> str:
        # TODO: Return the name of the quantization method (e.g., "dit_fp8").
        return self.quant_method

    def get_supported_act_dtypes(self) -> list[str]:
        # TODO: Return supported activation types. Usually ["float16", "bfloat16"].
        return []

    def get_min_capability(self) -> int:
        # TODO: Implement capability check.
        # E.g., if method is "fp8", require capability >= 89 (Ada/Hopper).
        # if method is "awq", might need less.
        return 70

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DiTQuantizationConfig":
        """Factory method to create config from a dictionary (e.g. config.json)."""
        # TODO: Extract 'quantization_config' dict from the input 'config'.
        # TODO: Parse 'bits', 'group_size', 'zero_point' from that dict.
        # TODO: Return an instance of cls().
        raise NotImplementedError("Parsing logic not implemented yet")

    def get_quant_method(self, layer: Any, input_type: str) -> str | None:
        """
        Determines if a specific layer should be quantized.
        """
        # TODO: Logic to check if 'layer.name' matches any string in 'self.modules_to_not_convert'.
        # If it matches, return None (skip quantization).

        # TODO: Otherwise, return self.quant_method string to tell vLLM to quantize it.
        return None
