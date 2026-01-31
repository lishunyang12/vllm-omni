# PR: Add GPU Memory Profiling Support

## Summary

Add GPU memory snapshot and timeline visualization capabilities to vLLM-Omni, enabling debugging of CUDA OOM errors and understanding memory allocation patterns.

### Background

As discussed with @Canling, we should consolidate the profiling usage for diffusion and omni pipelines so that we don't have to stay in sync with the vLLM repo's profiler implementation. This PR introduces a standalone `vllm_omni/profiler/` module that is independent of vLLM's profiler, giving us full control over profiling features and avoiding upstream dependency issues.

## Changes

### New Module: `vllm_omni/profiler/`
- `config.py` - `ProfilerConfig` dataclass with `performance` and `memory` flags
- `base.py` - `ProfilerBase` abstract class
- `torch_profiler.py` - Single `TorchProfiler` class handling both performance traces and memory profiling

### Refactoring
- Consolidated profiler into single `TorchProfiler` (removed separate MemoryProfiler/UnifiedProfiler)
- Removed legacy `VLLM_TORCH_PROFILER_DIR` environment variable support
- Config-based approach: profiling requires explicit `ProfilerConfig`
- Unified profiling interface for both diffusion and omni pipelines

### Updated Entrypoints
- `vllm_omni/entrypoints/omni.py` - Added `profiler_config` parameter, updated `start_profile()`/`stop_profile()` to return snapshots, timelines, memory_stats
- `vllm_omni/entrypoints/omni_llm.py` - Added `profiler_config` parameter, overloaded vLLM's `start_profile()`/`stop_profile()` methods to use our shared profiler
- `vllm_omni/entrypoints/omni_stage.py` - Handle profiler tasks with new config
- `vllm_omni/diffusion/profiler/__init__.py` - Re-export from shared module

### Updated Examples (all offline inference)
- `text_to_image.py`, `image_edit.py`, `text_to_video.py`, `image_to_video.py`
- `bagel/end2end.py`, `lora_inference.py`, `text_to_audio.py`
- `qwen2_5_omni/end2end.py`, `qwen3_omni/end2end.py`, `qwen3_tts/end2end.py`
- Added `--profile-dir`, `--profile-performance`, `--no-profile-performance`, `--profile-memory` CLI args

### New Tests
- `tests/profiler/test_config.py` - Config validation tests
- `tests/profiler/test_torch_profiler.py` - Performance/memory profiling E2E tests

### Documentation
- `docs/contributing/profiling.md` - Unified profiling guide (performance + memory)

## Usage

```bash
# Performance profiling (default)
python text_to_image.py --model MODEL --profile-dir ./profiles

# Memory profiling only
python text_to_image.py --model MODEL --profile-dir ./profiles --no-profile-performance --profile-memory

# Both together (recommended for debugging)
python text_to_image.py --model MODEL --profile-dir ./profiles --profile-memory
```

## Output Files

| File | Viewer |
|------|--------|
| `*_rank0.json.gz` | chrome://tracing or ui.perfetto.dev |
| `*_snapshot.pickle` | https://pytorch.org/memory_viz |
| `*_timeline.html` | Any browser |

---

## Test Checklist

### Unit Tests

- [ ] Run all profiler unit tests
  ```bash
  pytest tests/profiler/ -v
  ```

### Smoke Tests

- [ ] Import test
  ```bash
  python -c "from vllm_omni.profiler import ProfilerConfig, TorchProfiler; print('Import OK')"
  ```

- [ ] Config validation test
  ```bash
  python -c "
  from vllm_omni.profiler import ProfilerConfig
  c1 = ProfilerConfig()
  assert c1.performance == True and c1.memory == False
  c2 = ProfilerConfig(performance=False, memory=True)
  assert c2.performance == False and c2.memory == True
  print('Config OK')
  "
  ```

- [ ] OmniLLM profiler overload test
  ```bash
  python -c "
  from vllm_omni.entrypoints.omni_llm import OmniLLM
  from vllm_omni.profiler import ProfilerConfig
  # Verify start_profile/stop_profile methods exist and override vLLM's
  import inspect
  assert 'start_profile' in dir(OmniLLM)
  assert 'stop_profile' in dir(OmniLLM)
  # Check that methods are defined in OmniLLM, not inherited
  assert 'start_profile' in OmniLLM.__dict__
  assert 'stop_profile' in OmniLLM.__dict__
  print('OmniLLM profiler overload OK')
  "
  ```

### Integration Tests (GPU Required)

#### Test Matrix

| Example | Perf Only | Memory Only | Both | No Profiler |
|---------|-----------|-------------|------|-------------|
| text_to_image | [ ] | [ ] | [ ] | [ ] |
| image_edit | [ ] | [ ] | [ ] | [ ] |
| text_to_video | [ ] | [ ] | [ ] | [ ] |
| image_to_video | [ ] | [ ] | [ ] | [ ] |

#### Quick Test: text_to_image

- [ ] No profiler (baseline)
  ```bash
  python examples/offline_inference/text_to_image/text_to_image.py \
      --model Tongyi-MAI/Z-Image-Turbo \
      --prompt "a cat" \
      --num_inference_steps 4 \
      --output ./test_output.png
  ```

- [ ] Performance profiling only
  ```bash
  mkdir -p ./test_profiles
  python examples/offline_inference/text_to_image/text_to_image.py \
      --model Tongyi-MAI/Z-Image-Turbo \
      --prompt "a cat" \
      --num_inference_steps 4 \
      --profile-dir ./test_profiles \
      --output ./test_perf.png

  ls -la ./test_profiles/*.json.gz
  ```

- [ ] Memory profiling only
  ```bash
  python examples/offline_inference/text_to_image/text_to_image.py \
      --model Tongyi-MAI/Z-Image-Turbo \
      --prompt "a cat" \
      --num_inference_steps 4 \
      --profile-dir ./test_profiles \
      --no-profile-performance \
      --profile-memory \
      --output ./test_mem.png

  ls -la ./test_profiles/*.pickle
  ls -la ./test_profiles/*.html
  ```

- [ ] Both profilers
  ```bash
  python examples/offline_inference/text_to_image/text_to_image.py \
      --model Tongyi-MAI/Z-Image-Turbo \
      --prompt "a cat" \
      --num_inference_steps 4 \
      --profile-dir ./test_profiles \
      --profile-memory \
      --output ./test_both.png
  ```

### Output Verification

- [ ] Trace file is valid gzip
  ```bash
  gunzip -t ./test_profiles/*_rank0.json.gz
  ```

- [ ] Snapshot file is loadable
  ```bash
  python -c "import pickle; pickle.load(open('./test_profiles/*_snapshot.pickle', 'rb')); print('OK')"
  ```

- [ ] Timeline HTML exists
  ```bash
  ls -la ./test_profiles/*_timeline.html
  ```

### Visual Verification (Manual)

- [ ] Open `*.json.gz` in chrome://tracing or ui.perfetto.dev
- [ ] Drag `*.pickle` to https://pytorch.org/memory_viz
- [ ] Open `*.html` in browser

---

## Success Criteria

- [ ] Unit tests pass
- [ ] Image generated
- [ ] Trace file exists (`*.json.gz`)
- [ ] Snapshot file exists (`*_snapshot.pickle`)
- [ ] Timeline file exists (`*_timeline.html`)
- [ ] Console shows "PROFILING RESULTS"
- [ ] Memory stats displayed

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No snapshot file | PyTorch < 2.0 | Upgrade PyTorch |
| Empty timeline | No CUDA allocations | Increase `max_entries` |
| Import error | Missing module | Check `vllm_omni/profiler/__init__.py` |
| OOM during profiling | Profiler overhead | Reduce `max_entries` |
