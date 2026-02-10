# [Feature] Unified Profiler with Online Serving and Stage-Aware Endpoints

## Summary

- Consolidate the scattered diffusion-only profiler (`vllm_omni/diffusion/profiler/`) into a unified `vllm_omni/profiler/` module that works across all stage types (LLM, diffusion, omni-modality)
- Add stage-aware HTTP profiler endpoints (`/start_profile`, `/stop_profile`) for the online API server, following upstream vLLM's API shape and extending it with an optional `stages` parameter for multi-stage pipeline profiling
- Wire up `ProfilerConfig` end-to-end: CLI args → `AsyncOmni` → per-stage workers via `to_dict()`/`from_dict()` serialization
- Add `--profile-dir` CLI argument to all offline inference examples (text-to-image, image-to-video, qwen2.5-omni, qwen3-omni, etc.)

## Changes

### New files
| File | Description |
|------|-------------|
| `vllm_omni/profiler/__init__.py` | Unified profiler package (replaces `vllm_omni/diffusion/profiler/`) |
| `vllm_omni/profiler/config.py` | `ProfilerConfig` dataclass with `to_dict()`/`from_dict()`/`from_any()` serialization |
| `vllm_omni/profiler/torch_profiler.py` | `TorchProfiler` class aligned with upstream vLLM 0.16.0 semantics |
| `vllm_omni/entrypoints/serve/profile/api_router.py` | Stage-aware `/start_profile` and `/stop_profile` HTTP endpoints |
| `tests/profiler/test_config.py` | Unit tests for `ProfilerConfig` |
| `tests/profiler/test_torch_profiler.py` | Unit tests for `TorchProfiler` (CUDA + CPU) |

### Deleted files
| File | Reason |
|------|--------|
| `vllm_omni/diffusion/profiler/base.py` | Replaced by `vllm_omni/profiler/torch_profiler.py` |
| `vllm_omni/diffusion/profiler/torch_profiler.py` | Replaced by `vllm_omni/profiler/torch_profiler.py` |

### Modified files
| File | Change |
|------|--------|
| `vllm_omni/entrypoints/omni.py` | `OmniBase.__init__` accepts `profiler_config`, `start_profile(stages)` / `stop_profile(stages)` methods |
| `vllm_omni/entrypoints/omni_llm.py` | `OmniLLM` accepts `profiler_config`, single-stage `start_profile()` / `stop_profile()` |
| `vllm_omni/entrypoints/omni_stage.py` | Stage workers handle `PROFILER_START`/`PROFILER_STOP` tasks via `TorchProfiler` |
| `vllm_omni/entrypoints/omni_diffusion.py` | Uses unified profiler module |
| `vllm_omni/entrypoints/openai/api_server.py` | Replaces upstream profiler routes with stage-aware versions; converts `profiler_config` for `AsyncOmni` |
| `vllm_omni/diffusion/diffusion_engine.py` | Uses unified profiler module for diffusion engine profiling |
| `vllm_omni/diffusion/worker/diffusion_worker.py` | Uses unified profiler module |
| `vllm_omni/config/__init__.py` | Re-exports `ProfilerConfig` |
| `docs/contributing/profiling.md` | Full documentation for offline and online profiling |
| `examples/offline_inference/*/` | All examples now support `--profile-dir` CLI flag |

## Architecture

### Class Hierarchy

```
ProfilerConfig (vllm_omni/profiler/config.py)     TorchProfiler (vllm_omni/profiler/torch_profiler.py)
       │    to_dict() / from_dict() / from_any()          │    start() / stop() / step() / shutdown()
       │                                                   │
       │  used by all paths below                          │  instantiated in each worker process
       ▼                                                   ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                     Entry Points                             │
  ├────────────────────┬────────────────────┬────────────────────┤
  │  Multi-Stage       │  Single-Stage LLM  │  Single-Stage      │
  │  (Qwen-Omni)       │  (OmniLLM)         │  Diffusion         │
  │                    │                    │  (OmniDiffusion)   │
  │  OmniBase          │  OmniLLM(LLM)     │  OmniDiffusion     │
  │   ├─ Omni          │                    │   └─ DiffusionEngine│
  │   └─ AsyncOmni     │                    │       └─ Workers   │
  └────────────────────┴────────────────────┴────────────────────┘
```

### Multi-Stage Profiling Flow (Qwen2.5-Omni / Qwen3-Omni)

This is the primary flow for online serving and offline omni-modality inference.
Example: `stages=[0]` profiles only the Thinker stage.

```
  Online Serving                                 Offline Inference
  ──────────────                                 ─────────────────

  curl -X POST /start_profile                    omni = Omni(
    -d '{"stages": [0]}'                           model="Qwen/Qwen2.5-Omni-7B",
         │                                         profiler_config=ProfilerConfig(
         ▼                                           profiler="torch",
  api_router.py                                      torch_profiler_dir="./profiles"))
  ProfileRequest{stages:[0]}                         │
         │                                     omni.start_profile(stages=[0])
         ▼                                           │
  AsyncOmni.start_profile(stages=[0])                │
  (async wrapper → calls super())                    │
         │                                           │
         └──────────────┬────────────────────────────┘
                        │
                        ▼
              OmniBase.start_profile(stages=[0])
              │
              │  for stage_id in [0]:
              │    task = {"type": PROFILER_START,
              │            "config": self._profiler_config.to_dict()}
              │    self.stage_list[0].submit(task)
              │
              ▼  (only stage 0 receives task; stages 1,2 are skipped)
  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
  │    Stage 0       │  │    Stage 1       │  │    Stage 2       │
  │   (Thinker)      │  │   (Talker)       │  │   (Code2Wav)     │
  │                  │  │                  │  │                  │
  │  in_q.get()      │  │  (no task)       │  │  (no task)       │
  │       │          │  │                  │  │                  │
  │       ▼          │  │                  │  │                  │
  │  Worker Process  │  │                  │  │                  │
  │  (omni_stage.py) │  │                  │  │                  │
  └───────┬──────────┘  └──────────────────┘  └──────────────────┘
          │
          ▼
  handle_profiler_task_local(task)  ──or──  handle_profiler_task_async(task)
  (sync worker: diffusion stages)          (async worker: LLM stages)
          │
          ├─ config = ProfilerConfig.from_dict(task["config"])
          ├─ profiler = TorchProfiler(config, worker_name="stage-0")
          └─ profiler.start()
          │
  ════════╪══════════════════════════════════════════════════
          │  Requests flow through stage 0 with profiling active
          │  torch.profiler captures CPU/CUDA activity per iteration
  ════════╪══════════════════════════════════════════════════
          │
  ┌───────┴───────── STOP FLOW ─────────────────────────────┐
  │                                                          │
  │  curl -X POST /stop_profile     omni.stop_profile()     │
  │    -d '{"stages": [0]}'              │                   │
  │         │                            │                   │
  │         └────────────┬───────────────┘                   │
  │                      ▼                                   │
  │  OmniBase.stop_profile(stages=[0])                       │
  │  │                                                       │
  │  │  stage.stop_profile()                                 │
  │  │    └─ submit(PROFILER_STOP)                           │
  │  │    └─ out_q.get(timeout=600)  # wait for worker reply │
  │  │                                                       │
  │  └───────────► Worker Process receives PROFILER_STOP     │
  │                      │                                   │
  │                      ▼                                   │
  │                profiler.stop()                           │
  │                      │                                   │
  │                      ├─ Flush trace via tensorboard_trace_handler
  │                      ├─ Write CUDA time stats table      │
  │                      └─ out_q.put({"type":"profiler_result"})
  │                                                          │
  └──────────────────────────────────────────────────────────┘
          │
          ▼
  Output Files (torch_profiler_dir):
  ├── stage-0_*.trace.json.gz     # TensorBoard / Perfetto trace
  └── profiler_out_0.txt          # CUDA time stats (key_averages table)
```

### Single-Stage LLM Flow (OmniLLM)

For single-stage LLM-only models. TorchProfiler runs in-process (no IPC needed).

```
  omni_llm = OmniLLM(profiler_config=ProfilerConfig(...))
       │
  omni_llm.start_profile()
       │
       ├─ TorchProfiler(config, worker_name="llm-rank-0").start()
       │   (created directly in the same process)
       │
       │  ... requests ...
       │
  omni_llm.stop_profile()
       │
       └─ profiler.stop() → trace files written
```

### Single-Stage Diffusion Flow (OmniDiffusion / DiffusionEngine)

For standalone diffusion models. Profiler is distributed to GPU workers via `collective_rpc`.

```
  omni_diff = OmniDiffusion(profiler_config=ProfilerConfig(...))
       │
  omni_diff.start_profile()
       │
       └─ DiffusionEngine.start_profile(config)
            │
            └─ collective_rpc("start_profile", args=(config.to_dict(),))
                 │
                 ├─► DiffusionWorker rank 0: TorchProfiler(config).start()
                 ├─► DiffusionWorker rank 1: TorchProfiler(config).start()
                 └─► ...
                 │
                 │  ... generation ...
                 │
  omni_diff.stop_profile()
       │
       └─ DiffusionEngine.stop_profile()
            │
            └─ collective_rpc("stop_profile")
                 │
                 ├─► Worker rank 0: profiler.stop() → trace files
                 └─► Worker rank 1: profiler.stop() → trace files

  Output Files:
  ├── diffusion-rank-0_*.trace.json.gz
  ├── diffusion-rank-1_*.trace.json.gz
  ├── profiler_out_0.txt
  └── profiler_out_1.txt
```

### Online Serving Config Conversion

When the API server starts, upstream's `--profiler-config` CLI arg is converted to our `ProfilerConfig`:

```
  vllm serve --profiler-config profiler=torch,torch_profiler_dir=./profiles
       │
       ▼
  build_async_omni_from_stage_config(args)
       │
       ├─ upstream_config = args.profiler_config   (vllm.config.ProfilerConfig)
       ├─ our_config = OmniProfilerConfig.from_any(upstream_config)
       │     └─ converts to vllm_omni.profiler.ProfilerConfig
       └─ AsyncOmni(model=..., profiler_config=our_config)
              │
              └─ OmniBase.__init__ stores self._profiler_config
                   │
                   └─ start_profile() serializes via .to_dict() for each stage worker
```

Upstream's profiler routes are replaced at server startup:

```
  app = build_openai_app(args)          # upstream registers /start_profile, /stop_profile
       │
  _remove_route_from_router(app, "/start_profile")   # remove upstream routes
  _remove_route_from_router(app, "/stop_profile")
       │
  attach_profile_router(app)            # register our stage-aware routes
       │                                # (checks app.state.args.profiler_config)
       ▼
  Our /start_profile accepts: {"stages": [0,1,2]} or empty body (all stages)
  Our /stop_profile  accepts: {"stages": [0,1,2]} or empty body (all stages)
```

## Test Plan

### Unit Tests (no GPU needed)

```bash
# 1. ProfilerConfig: defaults, validation, to_dict, from_dict, roundtrip, dir expansion
# 2. ProfilerConfig.from_any(): None, own instance, dict, upstream-like object, profiler=None
# 3. ProfilerConfig re-export from vllm_omni.config
pytest tests/profiler/test_config.py -v

# 4. API router: attach_router conditional (profiler set / None / profiler=None)
# 5. POST /start_profile: no body (all stages), stages=[0], stages=[0,2]
# 6. POST /stop_profile: no body (all stages), stages=[1]
# 7. Verifies engine_client.start_profile/stop_profile called with correct stages arg
pytest tests/profiler/test_api_router.py -v
```

### Unit Tests (CUDA required)

```bash
# 8.  TorchProfiler start/stop lifecycle
# 9.  stop without start is no-op
# 10. double start is no-op
# 11. shutdown stops running profiler
# 12. step() iteration counting
# 13. delay_iterations: profiler starts only after N steps
# 14. max_iterations: profiler auto-stops after N steps
# 15. config-driven settings (record_shapes, memory, flops)
# 16. Trace files (.trace.json.gz) written and non-empty
# 17. CUDA time stats (profiler_out_0.txt) written and non-empty
# 18. worker_name appears in trace filename (e.g. "stage-0")
pytest tests/profiler/test_torch_profiler.py -v
```

### Integration: Offline Diffusion (Text-to-Image)

```bash
# 19. Single-stage diffusion profiling end-to-end
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Tongyi-MAI/Z-Image-Turbo \
    --profile-dir ./profiles/t2i

# Verify:
ls ./profiles/t2i/
# - *.trace.json.gz exists and size > 0
# - profiler_out_*.txt exists and size > 0
python -c "
import glob, os
traces = glob.glob('./profiles/t2i/*.trace.json.gz')
stats = glob.glob('./profiles/t2i/profiler_out_*.txt')
assert len(traces) >= 1, f'No traces: {os.listdir(\"./profiles/t2i\")}'
assert len(stats) >= 1, f'No stats: {os.listdir(\"./profiles/t2i\")}'
assert all(os.path.getsize(f) > 0 for f in traces + stats)
print(f'OK: {len(traces)} traces, {len(stats)} stats files')
"
```

### Integration: Offline Qwen2.5-Omni (3-stage, stage-selective)

```bash
# 20. Multi-stage profiling — stage 0 (Thinker) only
python examples/offline_inference/qwen2_5_omni/end2end.py \
    --model Qwen/Qwen2.5-Omni-7B \
    --profile-dir ./profiles/qwen25

# Verify:
ls ./profiles/qwen25/
# - stage-0_*.trace.json.gz exists (Thinker trace)
# - No stage-1_* or stage-2_* files (only stage 0 was profiled)
python -c "
import glob, os
s0 = glob.glob('./profiles/qwen25/stage-0*')
s1 = glob.glob('./profiles/qwen25/stage-1*')
s2 = glob.glob('./profiles/qwen25/stage-2*')
assert len(s0) >= 1, f'No stage-0 traces: {os.listdir(\"./profiles/qwen25\")}'
assert len(s1) == 0, f'Unexpected stage-1 traces: {s1}'
assert len(s2) == 0, f'Unexpected stage-2 traces: {s2}'
print(f'OK: {len(s0)} stage-0 traces, no stage-1/2 traces')
"
```

### Integration: Offline Qwen3-Omni (3-stage)

```bash
# 21. Multi-stage profiling — Qwen3-Omni variant
python examples/offline_inference/qwen3_omni/end2end.py \
    --model Qwen/Qwen3-Omni-8B \
    --profile-dir ./profiles/qwen3

ls ./profiles/qwen3/
# Expected: stage-0_*.trace.json.gz
```

### Integration: Online Serving Startup

```bash
# 22. Server starts with profiling enabled
python -m vllm_omni.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Omni-7B \
    --profiler-config profiler=torch,torch_profiler_dir=./profiles/online

# Verify server logs contain:
#   "Profiler with mode 'torch' is enabled in the API server..."
```

### Integration: Online Profile All Stages

```bash
# 23. Profile all stages via HTTP (no stages param)
curl -X POST http://localhost:8000/start_profile
# Expected: HTTP 200

# Send requests to exercise all stages
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-Omni-7B",
        "messages": [{"role": "user", "content": "Hello, how are you?"}]
    }'

curl -X POST http://localhost:8000/stop_profile
# Expected: HTTP 200

# Verify traces for ALL stages:
ls ./profiles/online/
# Expected: stage-0_*, stage-1_*, stage-2_* trace files
python -c "
import glob
for i in range(3):
    files = glob.glob(f'./profiles/online/stage-{i}*')
    print(f'stage-{i}: {len(files)} files')
    assert len(files) >= 1, f'Missing stage-{i} traces'
print('OK: all 3 stages have traces')
"
```

### Integration: Online Stage-Selective Profiling

```bash
# 24. Profile only Stage 0 (Thinker)
rm -rf ./profiles/online/*  # clean from previous test

curl -X POST http://localhost:8000/start_profile \
    -H "Content-Type: application/json" \
    -d '{"stages": [0]}'

curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-Omni-7B",
        "messages": [{"role": "user", "content": "Tell me a joke"}]
    }'

curl -X POST http://localhost:8000/stop_profile \
    -H "Content-Type: application/json" \
    -d '{"stages": [0]}'

# Verify ONLY stage-0 traces:
python -c "
import glob
s0 = glob.glob('./profiles/online/stage-0*')
s1 = glob.glob('./profiles/online/stage-1*')
s2 = glob.glob('./profiles/online/stage-2*')
assert len(s0) >= 1, f'Missing stage-0 traces'
assert len(s1) == 0, f'Unexpected stage-1 traces: {s1}'
assert len(s2) == 0, f'Unexpected stage-2 traces: {s2}'
print(f'OK: only stage-0 ({len(s0)} files), no stage-1/2')
"
```

### Integration: Online Profile Talker + Code2Wav

```bash
# 25. Profile stages 1 and 2 only
rm -rf ./profiles/online/*

curl -X POST http://localhost:8000/start_profile \
    -H "Content-Type: application/json" \
    -d '{"stages": [1, 2]}'

# Send requests...
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-Omni-7B",
        "messages": [{"role": "user", "content": "Count to five"}]
    }'

curl -X POST http://localhost:8000/stop_profile \
    -H "Content-Type: application/json" \
    -d '{"stages": [1, 2]}'

# Verify:
python -c "
import glob
s0 = glob.glob('./profiles/online/stage-0*')
s1 = glob.glob('./profiles/online/stage-1*')
s2 = glob.glob('./profiles/online/stage-2*')
assert len(s0) == 0, f'Unexpected stage-0 traces: {s0}'
assert len(s1) >= 1, f'Missing stage-1 traces'
assert len(s2) >= 1, f'Missing stage-2 traces'
print(f'OK: no stage-0, stage-1 ({len(s1)} files), stage-2 ({len(s2)} files)')
"
```

### Negative Tests

```bash
# 26. Server WITHOUT --profiler-config: endpoints should not exist
python -m vllm_omni.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Omni-7B
# (in another terminal)
curl -X POST http://localhost:8000/start_profile
# Expected: 404 Not Found or 405 Method Not Allowed

# 27. Offline: start_profile without profiler_config raises ValueError
python -c "
from vllm_omni.entrypoints.omni_llm import OmniLLM
try:
    llm = OmniLLM.__new__(OmniLLM)
    llm._profiler_config = None
    llm._profiler_instance = None
    llm.start_profile()
    assert False, 'Should have raised ValueError'
except ValueError as e:
    print(f'OK: {e}')
"
```

### Trace Viewing

```bash
# 28. Verify trace files load in Perfetto
#     Upload any .trace.json.gz to https://ui.perfetto.dev/
#     Should render timeline with CPU/CUDA activity
```

### Checklist

- [ ] `pytest tests/profiler/test_config.py -v` — all 14 tests pass (config, from_any, re-export)
- [ ] `pytest tests/profiler/test_api_router.py -v` — all 8 tests pass (endpoints, attach_router)
- [ ] `pytest tests/profiler/test_torch_profiler.py -v` — all 13 tests pass (lifecycle, trace output)
- [ ] Offline text-to-image: traces written, non-empty
- [ ] Offline Qwen2.5-Omni: stage-0 traces only (no stage-1/2)
- [ ] Offline Qwen3-Omni: stage-0 traces
- [ ] Online server starts with profiler warning
- [ ] Online `/start_profile` → 200, `/stop_profile` → 200, all stage traces written
- [ ] Online `{"stages": [0]}` → only stage-0 traces
- [ ] Online `{"stages": [1,2]}` → only stage-1/2 traces
- [ ] Server without `--profiler-config` → 404 on `/start_profile`
- [ ] Offline without `profiler_config` → ValueError
- [ ] Traces load in Perfetto UI
