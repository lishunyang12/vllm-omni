# Profiling vLLM-Omni

Profiling is only intended for vLLM-Omni developers and maintainers to understand the proportion of time spent in different parts of the codebase. **vLLM-Omni end-users should never turn on profiling** as it will significantly slow down the inference.

vLLM-Omni supports cross-stage profiling using the PyTorch Profiler. Since Omni runs multiple engine instances (stages) in separate processes, you can capture traces for all stages simultaneously or target specific ones.

**1. Enabling the Profiler**

Before running your script, you must set the ```VLLM_TORCH_PROFILER_DIR``` environment
variable. This is where the ```.json``` trace files will be saved.

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

After calling ``stop_profile()``, the directory specified in ```VLLM_TORCH_PROFILER_DIR``` will contain subdirectories for each stage:

   - ```stage_0_<model_name>/```: Contains traces for the first stage.

   - ```stage_1_<model_name>/```: Contains traces for the second stage.

Each directory contains standard Chrome trace ```.json``` files. You can upload these to [Perfetto](https://ui.perfetto.dev/) or ```chrome://tracing``` to analyze the execution timeline of each stage.

[!TIP] Each stage runs as a separate process. When you call start_profile() on the OmniLLM object, it sends a control signal to all underlying workers to begin recording simultaneously.