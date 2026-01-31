# GPU Memory Profiling in vLLM-Omni

This guide explains how to profile GPU memory usage for debugging OOM errors,
detecting memory leaks, and understanding memory allocation patterns.

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

## Understanding Memory Timeline

The categorized timeline shows memory usage by type. Categories depend on the workload:

### Inference Categories
- **Model Weights**: Loaded model parameters
- **KV Cache**: Key-Value cache for attention (LLM/transformers)
- **Activations**: Intermediate computation buffers
- **Input/Output**: Input tensors and generated outputs

### Diffusion-Specific Categories
- **Latents**: Latent space tensors
- **Noise**: Random noise tensors for denoising
- **VAE Buffers**: Encoder/decoder intermediate states
- **Attention Buffers**: Self/cross-attention computation

## Debugging OOM Errors

1. Enable memory profiling before the OOM occurs
2. Look for allocations that grow over time (memory leaks)
3. Check peak memory usage in the timeline
4. Use stack traces in the snapshot to identify which code allocated the most memory

### Example Workflow

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

## Memory Snapshot Visualization

The `.pickle` file can be visualized at https://pytorch.org/memory_viz:

1. Navigate to https://pytorch.org/memory_viz
2. Drag and drop the `.pickle` file
3. Explore:
   - **Timeline**: See allocations over time
   - **Stack traces**: Hover over allocations to see call stacks
   - **Allocation sizes**: Filter by size to find large allocations

## Memory Timeline (HTML)

The `.html` file shows categorized memory usage:

1. Open in any browser
2. View memory breakdown by category
3. Identify which component uses the most memory

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

### Omni Methods

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

## Best Practices

1. **Set `max_entries` appropriately**: Lower values (e.g., 10000) for short runs reduce memory overhead of profiling itself

2. **Profile specific stages**: Use `omni.start_profile(stages=[0])` to reduce overhead

3. **Use memory profiling during development**: Disable in production for performance

4. **Compare before/after**: Profile before and after optimizations to measure impact

5. **Check peak vs current**: Large difference suggests memory fragmentation

## Troubleshooting

### Snapshot file is empty or corrupted
- Ensure `torch.cuda.memory._record_memory_history()` is supported (PyTorch 2.0+)
- Check that profiling was started before the allocations you want to capture

### Timeline doesn't show categories
- Categories require PyTorch's profiler to identify tensor types
- Custom tensors may appear as "Other"

### High overhead from profiling
- Reduce `max_entries` to lower values
- Use `performance=False` if you only need memory data
- Profile only specific stages instead of all
