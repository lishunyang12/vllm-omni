# Profiling vLLM-Omni

> **Warning:** Profiling incurs significant overhead. Use only for development and debugging, never in production.

vLLM-Omni provides a unified profiling module (`vllm_omni/profiler/`) that captures both **performance traces** (Chrome traces) and **memory timelines** automatically. This module works across both omni-modality models and diffusion models.

## Quick Start

```python
from vllm_omni import Omni
from vllm_omni.profiler import ProfilerConfig

# Configure profiler at initialization
omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    profiler_config=ProfilerConfig(output_dir="./profiles")
)

# Profile your workload
omni.start_profile()
outputs = omni.generate({"prompt": "a cat"}, sampling_params)
results = omni.stop_profile()

# View results
print(f"Performance trace: {results['traces'][0]}")
print(f"Memory timeline: {results['timelines'][0]}")
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
    output_dir="./profiles",    # Where to save profiling outputs
)
```

When profiling is enabled, both performance trace and memory timeline are captured automatically.

## Output Files

| File | Format | How to View |
|------|--------|-------------|
| `*_rank0.json.gz` | Chrome trace | chrome://tracing or ui.perfetto.dev |
| `*_timeline.html` | Memory timeline | Any browser |

### File Naming Convention

```
{output_dir}/
├── stage_{id}_{timestamp}_rank{rank}.json.gz       # Performance trace
└── stage_{id}_{timestamp}_rank{rank}_timeline.html # Memory timeline
```

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

### Example: Streaming Generation with Profiling

```python
from vllm_omni import Omni
from vllm_omni.profiler import ProfilerConfig

# Configure profiler
profiler_config = ProfilerConfig(output_dir="./profiles")

omni = Omni(model="Qwen/Qwen2.5-Omni-7B", profiler_config=profiler_config)

# Start profiling specific stages
omni.start_profile(stages=[0])

# Initialize generator
omni_generator = omni.generate(prompts, sampling_params_list)

total_requests = len(prompts)
processed_count = 0

# Main Processing Loop
for stage_outputs in omni_generator:
    processed_count += len(stage_outputs.request_output)

    # Stop profiler when all requests are done
    if processed_count >= total_requests:
        print(f"Processed {processed_count}/{total_requests}. Stopping profiler...")
        results = omni.stop_profile()
        print(f"Trace saved to: {results['traces'][0]}")

omni.close()
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

**Why minimize dimensions?**
- **Spatial (height/width)**: Reduces memory usage so profiler doesn't crash
- **Temporal (frames)**: Video models process 3D tensors; fewer frames = smaller traces
- **Steps**: Profiling 2 steps gives same performance data as 50 steps

### Examples

- **Image Edit**: [examples/offline_inference/image_to_image/image_edit.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py)
- **Image to Video**: [examples/offline_inference/image_to_video/](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_video)
- **Text to Image**: [examples/offline_inference/text_to_image/text_to_image.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/text_to_image/text_to_image.py)

---

## Understanding Memory Timeline

The memory timeline (`.html` file) shows GPU memory usage over time, categorized by type.

**Inference Categories:**
- **Model Weights**: Loaded model parameters
- **KV Cache**: Key-Value cache for attention (LLM/transformers)
- **Activations**: Intermediate computation buffers
- **Input/Output**: Input tensors and generated outputs

**Diffusion-Specific Categories:**
- **Latents**: Latent space tensors
- **Noise**: Random noise tensors for denoising
- **VAE Buffers**: Encoder/decoder intermediate states
- **Attention Buffers**: Self/cross-attention computation

### Debugging OOM Errors

```python
# 1. Configure profiler
profiler_config = ProfilerConfig(output_dir="./oom_debug")

# 2. Initialize with config
omni = Omni(model="...", profiler_config=profiler_config)

# 3. Start profiling
omni.start_profile()

# 4. Run the workload that causes OOM
try:
    outputs = omni.generate(...)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("OOM occurred, collecting profiler data...")

# 5. Stop and collect results (even after OOM)
results = omni.stop_profile()
print(f"Timeline: {results['timelines'][0]}")
```

---

## Viewing Traces

### Performance Traces (`.json.gz`)

- [Perfetto UI](https://ui.perfetto.dev/) (recommended)
- `chrome://tracing` (Chrome only)

### Memory Timelines (`.html`)

- Open in any browser

---

## API Reference

### ProfilerConfig

```python
@dataclass
class ProfilerConfig:
    output_dir: str = "./profiles"    # Output directory for profiling files
```

### Omni/OmniLLM Methods

```python
# Start profiling for specified stages (None = all)
omni.start_profile(stages: list[int] | None = None) -> None

# Stop profiling and collect results
omni.stop_profile(stages: list[int] | None = None) -> dict
# Returns:
# {
#     "traces": [...],        # Performance traces (.json.gz)
#     "timelines": [...],     # Memory timelines (.html)
#     "memory_stats": {...},  # Per-stage memory statistics
# }
```

### Memory Stats

The `memory_stats` dictionary contains per-stage statistics:

```python
{
    0: {  # Stage 0
        "peak_allocated_mb": 1234.5,
        "current_allocated_mb": 1000.0,
        "peak_reserved_mb": 2000.0,
        "current_reserved_mb": 1500.0,
    },
    # ... more stages
}
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
| No timeline file | matplotlib not installed | Install matplotlib: `pip install matplotlib` |
| Import error | Missing module | Check `vllm_omni/profiler/__init__.py` |
| OOM during profiling | Profiler overhead | Reduce model dimensions |
| Huge trace files | Too many steps/frames | Reduce `num_inference_steps`, `num_frames` |

> **Note:** Asynchronous (online) profiling is not fully supported. Use `start_profile()` and `stop_profile()` only in offline inference scripts. Do not use in server-mode or streaming scenarios—traces may be incomplete.
