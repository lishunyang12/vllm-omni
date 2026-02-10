# Profiling vLLM-Omni

> **Warning:** Profiling incurs significant overhead. Use only for development and debugging, never in production.

vLLM-Omni provides a profiling module (`vllm_omni/profiler/`) aligned with upstream vLLM 0.16.0 semantics. It captures **performance traces** (TensorBoard/Chrome traces) using `tensorboard_trace_handler` and supports delay/max iteration control.

## Quick Start

```python
from vllm_omni import Omni
from vllm_omni.profiler import ProfilerConfig

# Configure profiler at initialization
omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    profiler_config=ProfilerConfig(
        profiler="torch",
        torch_profiler_dir="./profiles",
    )
)

# Profile your workload
omni.start_profile()
outputs = omni.generate({"prompt": "a cat"}, sampling_params)
omni.stop_profile()

# Trace files are written to ./profiles/ by each worker
```

## Command Line Usage

All offline inference examples support profiling via CLI arguments:

```bash
# Enable profiling
python text_to_image.py --model MODEL --profile-dir ./profiles
```

## ProfilerConfig

```python
from vllm_omni.profiler import ProfilerConfig

ProfilerConfig(
    profiler="torch",                          # Required: "torch" or "cuda"
    torch_profiler_dir="./profiles",           # Required when profiler="torch"
    torch_profiler_with_stack=True,            # Enable stack tracing
    torch_profiler_with_flops=False,           # Enable FLOPS counting
    torch_profiler_use_gzip=True,              # Save traces in gzip format
    torch_profiler_dump_cuda_time_total=True,  # Dump CUDA time stats on stop
    torch_profiler_record_shapes=False,        # Record tensor shapes
    torch_profiler_with_memory=False,          # Enable memory profiling
    delay_iterations=0,                        # Skip N iterations before starting
    max_iterations=0,                          # Stop after N iterations (0=unlimited)
)
```

### Serialization

`ProfilerConfig` supports `to_dict()` / `from_dict()` for cross-process RPC serialization.

## Output Files

| File | Format | How to View |
|------|--------|-------------|
| `*.trace.json.gz` | TensorBoard trace | TensorBoard, chrome://tracing, or ui.perfetto.dev |
| `profiler_out_*.txt` | CUDA time stats | Any text editor |

---

## Profiling Omni-Modality Models

### Selective Stage Profiling

Profile specific stages to keep trace files manageable:

```python
# Profile all stages
omni.start_profile()

# Only profile Stage 1
omni.start_profile(stages=[1])

# Stage 0 (Thinker) and Stage 2 (Audio Decoder) for Qwen Omni
omni.start_profile(stages=[0, 2])
```

### Examples

- **Qwen2.5-Omni**: [examples/offline_inference/qwen2_5_omni/end2end.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen2_5_omni/end2end.py)
- **Qwen3-Omni**: [examples/offline_inference/qwen3_omni/end2end.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen3_omni/end2end.py)

---

## Profiling Diffusion Models

Diffusion profiling is end-to-end, capturing encoding, denoising loops, and decoding.

### Minimizing Trace Size

For profiling, minimize dimensions to keep trace files manageable:

```bash
python image_to_video.py \
    --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --image input.png \
    --prompt "A cat playing with yarn" \
    --profile-dir ./profiles \
    \
    # Minimize dimensions for profiling:
    --height 48 \
    --width 64 \
    --num_frames 2 \
    --num_inference_steps 2
```

### Examples

- **Image Edit**: [examples/offline_inference/image_to_image/image_edit.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py)
- **Image to Video**: [examples/offline_inference/image_to_video/](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_video)
- **Text to Image**: [examples/offline_inference/text_to_image/text_to_image.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/text_to_image/text_to_image.py)

---

## Viewing Traces

### Performance Traces (`.trace.json.gz`)

- [TensorBoard](https://www.tensorflow.org/tensorboard) (recommended)
- [Perfetto UI](https://ui.perfetto.dev/)
- `chrome://tracing` (Chrome only)

---

## API Reference

### ProfilerConfig

```python
@dataclass
class ProfilerConfig:
    profiler: Literal["torch", "cuda"] | None = None
    torch_profiler_dir: str = ""
    torch_profiler_with_stack: bool = True
    torch_profiler_with_flops: bool = False
    torch_profiler_use_gzip: bool = True
    torch_profiler_dump_cuda_time_total: bool = True
    torch_profiler_record_shapes: bool = False
    torch_profiler_with_memory: bool = False
    delay_iterations: int = 0
    max_iterations: int = 0
```

### TorchProfiler

```python
class TorchProfiler:
    def __init__(self, config: ProfilerConfig, worker_name: str = "", local_rank: int = 0): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def step(self) -> None: ...
    def shutdown(self) -> None: ...
    @property
    def is_running(self) -> bool: ...
```

### Omni Methods

```python
# Start profiling for specified stages (None = all)
omni.start_profile(stages: list[int] | None = None) -> None

# Stop profiling for specified stages (None = all)
omni.stop_profile(stages: list[int] | None = None) -> None
```

---

## Best Practices

1. **Profile specific stages**: Use `omni.start_profile(stages=[0])` to reduce overhead and file size

2. **Minimize dimensions for diffusion**: Use small height/width/frames/steps when profiling

3. **Compare before/after**: Profile before and after optimizations to measure impact

4. **Use during development only**: Disable profiling in production for performance

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Import error | Missing module | Check `vllm_omni/profiler/__init__.py` |
| OOM during profiling | Profiler overhead | Reduce model dimensions |
| Huge trace files | Too many steps/frames | Reduce `num_inference_steps`, `num_frames` |

---

## Online Serving Profiling

When running the vLLM-Omni API server, profiling can be enabled via CLI
and controlled via HTTP endpoints at runtime.

### Starting the Server with Profiling Enabled

```bash
python -m vllm_omni.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Omni-7B \
    --profiler-config profiler=torch,torch_profiler_dir=./profiles
```

### HTTP Endpoints

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| POST | `/start_profile` | `{"stages": [0, 1, 2]}` (optional) | Start profiling |
| POST | `/stop_profile` | `{"stages": [0, 1, 2]}` (optional) | Stop profiling |

If `stages` is omitted or null, all stages are profiled.

### Stage IDs for Qwen Omni Models

| Stage | Qwen2.5-Omni | Qwen3-Omni |
|-------|-------------|------------|
| 0 | Thinker (understanding) | Thinker (MoE understanding) |
| 1 | Talker (text → RVQ codes) | Talker (code predictor) |
| 2 | Code2Wav (codes → audio) | Code2Wav (codes → audio) |

### Examples

```bash
# Profile all stages (default)
curl -X POST http://localhost:8000/start_profile

# Profile only the Thinker stage
curl -X POST http://localhost:8000/start_profile \
    -H "Content-Type: application/json" \
    -d '{"stages": [0]}'

# Profile Thinker and Talker stages
curl -X POST http://localhost:8000/start_profile \
    -H "Content-Type: application/json" \
    -d '{"stages": [0, 1]}'

# Stop profiling (traces written to torch_profiler_dir)
curl -X POST http://localhost:8000/stop_profile
```

### Tips

1. **Profile one stage at a time** for smaller, more focused traces
2. **Profile the Thinker** (stage 0) to analyze LLM bottlenecks
3. **Profile the Talker** (stage 1) to analyze codec generation
4. **Profile Code2Wav** (stage 2) to analyze audio synthesis
5. Trace files are named per-stage (e.g., `stage-0_*.trace.json.gz`)
