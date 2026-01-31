# Profiling vLLM-Omni

> **Warning:** Profiling incurs significant overhead. Use only for development and debugging, never in production.

vLLM-Omni provides a unified profiling module (`vllm_omni/profiler/`) that supports both **performance profiling** (Chrome traces) and **GPU memory profiling** (snapshots + timelines). This module works across both omni-modality models and diffusion models.

## Quick Start

```python
from vllm_omni import Omni
from vllm_omni.profiler import ProfilerConfig

# Configure profiler at initialization
omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    profiler_config=ProfilerConfig(
        output_dir="./profiles",
        performance=True,  # Chrome trace (default)
        memory=True,       # Memory snapshot + timeline
    )
)

# Profile your workload
omni.start_profile()
outputs = omni.generate({"prompt": "a cat"}, sampling_params)
results = omni.stop_profile()

# View results
print(f"Performance trace: {results['traces'][0]}")
print(f"Memory snapshot: {results['snapshots'][0]}")
print(f"Memory timeline: {results['timelines'][0]}")
```

## Command Line Usage

All offline inference examples support profiling via CLI arguments:

```bash
# Performance profiling only (default)
python text_to_image.py --model MODEL --profile-dir ./profiles

# Memory profiling only
python text_to_image.py --model MODEL --profile-dir ./profiles \
    --no-profile-performance --profile-memory

# Both together (recommended for debugging)
python text_to_image.py --model MODEL --profile-dir ./profiles --profile-memory
```

## ProfilerConfig Options

```python
from vllm_omni.profiler import ProfilerConfig

ProfilerConfig(
    output_dir="./profiles",    # Where to save files
    performance=True,           # Enable Chrome trace (default: True)
    memory=False,               # Enable memory profiling (default: False)
    backend="torch",            # "torch" (current) or "nsight" (future)
    max_entries=100000,         # Max memory records (higher = more overhead)
)
```

## Output Files

| File | Format | How to View |
|------|--------|-------------|
| `*_rank0.json.gz` | Chrome trace | chrome://tracing or ui.perfetto.dev |
| `*_snapshot.pickle` | PyTorch snapshot | https://pytorch.org/memory_viz (drag & drop) |
| `*_timeline.html` | HTML | Any browser |

### File Naming Convention

```
{output_dir}/
├── stage_{id}_{timestamp}_rank{rank}.json.gz           # Performance trace
├── stage_{id}_{timestamp}_rank{rank}_snapshot.pickle   # Memory snapshot
└── stage_{id}_{timestamp}_rank{rank}_timeline.html     # Memory timeline
```

---

## Profiling Omni-Modality Models

### Selective Stage Profiling

It is highly recommended to profile specific stages to keep trace files manageable:

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
profiler_config = ProfilerConfig(
    output_dir="./profiles",
    performance=True,
    memory=False,
)

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
    --profile-memory \
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

## GPU Memory Profiling

Memory profiling helps debug OOM errors, detect memory leaks, and understand allocation patterns.

### Understanding Memory Timeline

The categorized timeline shows memory usage by type:

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
# 1. Configure profiler with memory enabled
profiler_config = ProfilerConfig(
    output_dir="./oom_debug",
    performance=False,  # Focus on memory
    memory=True,
    max_entries=50000,  # Lower to reduce overhead
)

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
print(f"Snapshot: {results['snapshots'][0]}")
```

### Memory Snapshot Visualization

The `.pickle` file can be visualized at https://pytorch.org/memory_viz:

1. Navigate to https://pytorch.org/memory_viz
2. Drag and drop the `.pickle` file
3. Explore:
   - **Timeline**: See allocations over time
   - **Stack traces**: Hover over allocations to see call stacks
   - **Allocation sizes**: Filter by size to find large allocations

---

## Viewing Traces

### Performance Traces (`.json.gz`)

- [Perfetto UI](https://ui.perfetto.dev/) (recommended)
- `chrome://tracing` (Chrome only)

### Memory Snapshots (`.pickle`)

- [PyTorch Memory Viz](https://pytorch.org/memory_viz) - drag and drop

### Memory Timelines (`.html`)

- Open in any browser

---

## API Reference

### ProfilerConfig

```python
@dataclass
class ProfilerConfig:
    output_dir: str = "./profiles"    # Output directory
    performance: bool = True          # Enable performance trace
    memory: bool = False              # Enable memory profiling
    backend: str = "torch"            # Profiler backend
    max_entries: int = 100000         # Max memory records
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
#     "snapshots": [...],     # Memory snapshots (.pickle)
#     "timelines": [...],     # Categorized timelines (.html)
#     "memory_stats": {...},  # Per-stage statistics
# }
```

---

## Best Practices

1. **Set `max_entries` appropriately**: Lower values (e.g., 10000) for short runs reduce memory overhead

2. **Profile specific stages**: Use `omni.start_profile(stages=[0])` to reduce overhead

3. **Use memory profiling during development**: Disable in production for performance

4. **Minimize dimensions for diffusion**: Use small height/width/frames/steps when profiling

5. **Compare before/after**: Profile before and after optimizations to measure impact

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No snapshot file | PyTorch < 2.0 | Upgrade PyTorch |
| Empty timeline | No CUDA allocations | Increase `max_entries` |
| Import error | Missing module | Check `vllm_omni/profiler/__init__.py` |
| OOM during profiling | Profiler overhead | Reduce `max_entries` |
| Huge trace files | Too many steps/frames | Reduce `num_inference_steps`, `num_frames` |
| Timeline missing categories | Custom tensors | May appear as "Other" |

> **Note:** Asynchronous (online) profiling is not fully supported. Use `start_profile()` and `stop_profile()` only in offline inference scripts. Do not use in server-mode or streaming scenarios—traces may be incomplete.
