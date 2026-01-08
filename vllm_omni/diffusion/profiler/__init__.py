# vllm_omni/diffusion/profiler/__init__.py

from .torch_profiler import TorchProfiler

# Default profiler – can be changed later via config
CurrentProfiler = TorchProfiler

__all__ = ["CurrentProfiler", "TorchProfiler"]