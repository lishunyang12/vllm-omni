# Profiling vLLM-Omni

> **Warning:** Profiling incurs significant overhead. Use only for development and debugging, never in production.

vLLM-Omni uses the PyTorch Profiler to analyze performance across both **Multi-Stage LLMs** and **Diffusion Models**. 

## 1. Quick Start

**Step 1: Set the Output Directory**
Before running any script, set this environment variable. The system detects this and automatically saves traces here.

```bash
export VLLM_TORCH_PROFILER_DIR=./profiles
```

**Step 2: Run Profiling**

**A. For Diffusion Models (e.g., Wan 2.2, Z-Image)**

Diffusion profiling is End-to-End, capturing encoding, denoising loops, and decoding. Most example scripts automatically enable profiling if ```VLLM_TORCH_PROFILER_DIR``` is set.

**CLI Usage:**
```python
# Example: Running Text-to-Video with profiling enabled
export VLLM_TORCH_PROFILER_DIR=./profiles

python examples/offline_inference/text_to_video/text_to_video.py \
    --model "Wan-AI/Wan2.2-I2V-A14B-Diffusers" \
    # Reduce steps to avoid large trace
    --num_inference_steps 2 
```

**B. For Non-diffusion models (e.g., Qwen-Omni)**

It is best to limit profiling to one iteration to keep trace files manageable.

```bash
export VLLM_PROFILER_MAX_ITERS=1
```

**Python Usage**: Wrap your generation logic with start_profile() and stop_profile().

```python
from vllm_omni import OmniLLM

omni_llm = OmniLLM.from_engine_args(engine_args)

# Start profiling all active stages
outputs = omni_llm.generate(prompts, sampling_params)

# Stop profiling and save traces
omni_llm.stop_profile()
```

**Selective Stage Profiling**
The profiler is default to function across all stages. But It is highly recommended to profile specific stages by passing the stages list, preventing from producing too large trace files:
```python
# Only profile Stage 1
omni_llm.start_profile(stages=[1])
```

```python
# Stage 0 (Thinker) and Stage 2 (Audio Decoder) for qwen omni
omni_llm.start_profile(stages=[0, 2])
```

**Examples**:

1. **Qwen-omni 2.5**:  [https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen2_5_omni/end2end.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen2_5_omni/end2end.py)

2. **Qwen-omni 3.0**:   [https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen3_omni/end2end.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen3_omni/end2end.py)

**3. Online Inference(Async)**

For online serving using AsyncOmni, the methods are asynchronous. This allows you to toggle profiling dynamically without restarting the server.

```python
from vllm_omni import AsyncOmni

# Inside an async function:
async_omni = AsyncOmni.from_engine_args(engine_args)

await async_omni.start_profile()

async for output in async_omni.generate(prompt, sampling_params, request_id):
    # Process outputs...
    pass

await async_omni.stop_profile()
```

**Analyzing Omni Traces**

Output files are saved to your configured ```VLLM_TORCH_PROFILER_DIR```.

**Output**
**Chrome Trace** (```.json.gz```): Visual timeline of kernels and stages. Open in Perfetto UI.

**Viewing Tools:**

- [Perfetto](https://ui.perfetto.dev/)(recommended)
- ```chrome://tracing```(Chrome only)

**Note**: vLLM-Omni reuses the PyTorch Profiler infrastructure from vLLM. See the official vLLM profiler documentation:  [vLLM Profiling Guide](https://docs.vllm.ai/en/latest/dev/profiling.html)
