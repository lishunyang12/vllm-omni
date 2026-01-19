# Profiling vLLM-Omni

> **Warning:** Profiling incurs significant overhead. Use only for development and debugging, never in production.

vLLM-Omni uses the PyTorch Profiler to analyze performance across both **Multi-Stage LLMs** and **Diffusion Models**. For best results, profile on a single GPU to avoid synchronization noise.

## 1. Quick Start

**Step 1: Set the Output Directory**
Before running any script, set this environment variable. The system detects this and automatically saves traces here.

```bash
export VLLM_TORCH_PROFILER_DIR=./profiles
```

**Step 2: Run Profiling**

**A. For Diffusion Models (e.g., Wan 2.2, Z-Image)**

Diffusion profiling is End-to-End, capturing VAE encoding, denoising loops, and decoding. Most example scripts automatically enable profiling if ```VLLM_TORCH_PROFILER_DIR``` is set.

**CLI Usage:**
```python
# Example: Running Text-to-Video with profiling enabled
export VLLM_TORCH_PROFILER_DIR=./profiles

python examples/offline_inference/text_to_video/text_to_video.py \
    --model "Wan-AI/Wan2.2-I2V-A14B-Diffusers" \
    --num_inference_steps 15
```

**B. For Non-diffusion models (e.g., Qwen-Omni)**
\
It is best to limit profiling to one iteration to keep trace files manageable.

```bash
export VLLM_PROFILER_MAX_ITERS=1
```

**Python Usage**: Wrap your generation logic with start_profile() and stop_profile().

```python
from vllm_omni import OmniLLM

omni_llm = OmniLLM.from_engine_args(engine_args)

# Start profiling (pass stages=[0, 2] to filter specific stages)
omni_llm.start_profile()

outputs = omni_llm.generate(prompts, sampling_params)

# Stop and save traces
omni_llm.stop_profile()
```

**Analyzing Omni Traces**

Output files are saved to your configured ```VLLM_TORCH_PROFILER_DIR```.

**Output**
**Chrome Trace** (```.json.gz```): Visual timeline of kernels and stages. Open in Perfetto UI.

**Viewing Tools:**

- [Perfetto](https://ui.perfetto.dev/)(recommended)
- ```chrome://tracing```(Chrome only)

**Note**: vLLM-Omni reuses the PyTorch Profiler infrastructure from vLLM. See the official vLLM profiler documentation:  [vLLM Profiling Guide](https://docs.vllm.ai/en/latest/dev/profiling.html)
