# Profiling vLLM-Omni

Profiling is only intended for vLLM-Omni developers and maintainers to understand the proportion of time spent in different parts of the codebase. **vLLM-Omni end-users should never turn on profiling** as it will significantly slow down the inference.

vLLM-Omni supports cross-stage profiling using the PyTorch Profiler. Since Omni runs multiple engine instances (stages) in separate processes, you can capture traces for all stages simultaneously or target specific ones.

**1. Enabling the Profiler**

Before running your script, you must set the ```VLLM_TORCH_PROFILER_DIR``` environment
variable.

```Bash
export VLLM_TORCH_PROFILER_DIR=/path/to/save/traces
```

**2. Offline Inference**

For offlinie processing using ```OmniLLM```, you can wrap your ```generate``` calls with ```start_profile``` and ```stop_profile()```.

Basic Usage(All Stages)
```Python
from vllm_omni import OmniLLM

omni_llm = OmniLLM.from_engine_args(engine_args)

# Start profiling all active stages
omni_llm.start_profile()

outputs = omni_llm.generate(prompts, sampling_params)

# Stop profiling and save traces
omni_llm.stop_profile()
```

**Selective Stage Profiling**
If you only want to profile a specific stage (e.g., Stage 1), pass the stages list:
```python
# Only profile Stage 1
omni_llm.start_profile(stages=[1])
```

```python
# Stage 0 (Thinker) and Stage 2 (Audio Decoder) for qwen omni
omni_llm.start_profile(stages=[0, 2])
```

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

**4. Analyzing Omni Traces**

After ``stop_profile()`` completes (and the file write wait time has passed), the directory specified in ```VLLM_TORCH_PROFILER_DIR``` will contain the trace files.

File Structure: Files are saved directly in the specified directory. You will see one file per GPU worker.

Naming pattern: ```<timestamp>-rank-<gpu_rank>.<host_id>.pt.trace.json.gz```
   - Tensor Parallelism (TP): If a stage uses TP > 1, you will get multiple files for that stage
        - Example (Stage 0 with TP=2):
          - ```...rank-0...json.gz``` (GPU 0 trace)
          - ```...rank-1...json.gz``` (GPU 1 trace)
        - You can load both files into Perfetto simultaneously to visualize the synchronization (all-reduce) between GPUs.
   - Stage Identification: You can identify the stage by inspecting the file size or the timestamps.
        - Text/LLM Stages: Typically produce smaller files (e.g., ~50MB).
        - Audio/Talker Stages: Produce massive files (e.g., ~100MB+ compressed) due to the high volume of kernels executed during audio decoding.
   - Summary Logs: You also see profiler_out_<stage_id>.txt containing text-based summary tables (CPU/CUDA time percentages).

Viewing Tools:
1. [Perfetto](https://ui.perfetto.dev/): (Recommended): Best for handling large audio trace files.
2. ```chrome://tracing```: Good for smaller text-only traces.

[!TIP] If the trace file ends in ```gz```, you must unzip it (```gzip -d filename.json.gz```) before loading it into Chrome Tracing. Perfetto often handles GZIP files directly.
