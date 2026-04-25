# RFC: Diffusion Attention Backend Architecture

| | |
|---|---|
| **Status** | Draft |
| **Author** | @lishunyang12 |
| **Date** | 2026-04-26 |
| **Related** | #3079 (in-flight Blackwell backends), #1568 (RFC: backend extension), #3121 (LTX-2 cuDNN crash), #888 (SpargeAttn) |
| **Supersedes / Extends** | Extends the discussion in #1568 |

---

## Summary

vLLM-OMNI's diffusion attention layer currently uses a flat `if/elif` selector keyed off the `DIFFUSION_ATTENTION_BACKEND` env var, with three backends (`TORCH_SDPA`, `FLASH_ATTN`, `SAGE_ATTN`). PR #3079 adds two more (`CUDNN_ATTN`, `FLASHINFER_ATTN`) and a hardcoded Blackwell auto-route. Issue #1568 has converged on three structural directions: CLI args replacing env vars, YAML per-model defaults, and a unified-Impl-with-platform-dispatch backend pattern.

This RFC proposes the framework that unifies those directions into a single attention backend system, plus four extensions that the current discussion does not yet address:

1. **Capability tiers** — formalize per-SM priority so the framework generalizes beyond `BLACKWELL_CONSUMER` (the only tier validated in #3079)
2. **`has_attn_mask` priority signal** — directly justified by #3079's measurement that cuDNN beats FlashInfer 2× on HunyuanVideo-1.5 specifically because of mask handling
3. **`AttentionType` enum + `supported_types()` filter** — prevents sparse backends from being silently selected for cross-attention layers; lays groundwork for AR/TTS integration
4. **Cross-attention K/V caching for static conditioning** — perf win orthogonal to backend choice

The RFC also defines the integration path for autoregressive attention (TTS, omni-thinker) via a vLLM PagedAttention adapter rather than reimplementation.

---

## Motivation

### Pain points in the current state

The selector lives in `vllm_omni/diffusion/attention/selector.py` and dispatches on `os.environ["DIFFUSION_ATTENTION_BACKEND"]`. This pattern has accumulated several problems:

1. **No measurement-driven backend selection.** Users who don't set the env var get `TORCH_SDPA`, even on hardware where another backend wins by 2×. The auto-route hardcoded in `vllm_omni/platforms/cuda/platform.py` (added by #3079) handles Blackwell but not Hopper, Ada, or Ampere.

2. **No per-attention-type discrimination.** A backend selected globally is used for self-attention, cross-attention, and (eventually) autoregressive attention alike. Sparse backends like SpargeAttn (#888) have spatial-locality assumptions that break on cross-attention with text — but nothing prevents the selector from picking them.

3. **Backend parameters can only be passed via env vars.** As more sophisticated backends land (sparse, FP8, NVFP4), each needs tunables (`sparge_attn_topk`, `sparsity`, `cdf_threshold`). The env-var-soup approach scales badly.

4. **Per-model defaults require code changes.** HunyuanVideo-1.5 wins 2× on cuDNN per #3079's measurements, but a user must remember to set the env var. A pipeline class attribute would still require a Python change for each model added.

5. **Hardware coverage gap.** PR #3079 was tested only on sm_120 RTX Pro 6000 Blackwell. The auto-route logic for Ampere/Hopper is "fall through to FA" — fine as a default, but the framework doesn't formalize what should win on each hardware tier or how to validate it.

### The convergence in #1568

Issue #1568, opened by @jiangmengyu18 and discussed across multiple hardware teams (NVIDIA via @yma11, Intel via @xuechendi, AMD via @hsliuustc0106, NPU via @tjtanaa, @gcanlin), has converged on:

- **Migration from env var to CLI argument** (`--attention-backend`), matching upstream vLLM's pattern. @tjtanaa confirms upstream has deprecated `VLLM_ATTENTION_BACKEND` in favor of the CLI form.
- **YAML-based per-model defaults** with priority `CLI > YAML > code default`. Endorsed by @yma11.
- **Unified-Impl-with-platform-dispatch pattern** following upstream vLLM. @xuechendi proposes one `Backend` per algorithm (e.g. `FlashAttnBackend`) with one `Impl` containing `forward_cuda` / `forward_xpu` / `forward_npu` methods. Hardware-specific operations live in `_hw_ops.py` files (`_aiter_ops.py`, `_xpu_ops.py`, `_ascend_ops.py`). Hardware-specific Backend classes (e.g. `AscendAttention`) require a separate platform RFC justification.
- **Per-backend parameter passing** via CLI / config rather than env vars.

This RFC adopts those four directions verbatim. Where it differs from #1568 is in the four extensions listed in the Summary, which the thread has not addressed.

### Workloads on the horizon

vLLM-OMNI is broadening from pure diffusion to:

- **Omni models** like Qwen3-Omni and LTX-2 that combine diffusion (video) with autoregressive (text/audio) heads in one engine
- **TTS models** like VoxCPM2 (#2594) that are pure autoregressive but produce audio
- **Long-form video generation** (HunyuanVideo, Wan 2.2) where attention seq_len exceeds 30k tokens and sparse attention becomes the only viable path

Each of these has attention patterns that don't fit "bidirectional self-attn on a fixed-shape latent." The framework needs to accommodate them without forking the codebase.

---

## Goals

This RFC aims to deliver a backend system that:

1. **Generalizes per-platform** — priorities and capabilities encoded per `(SM tier, backend, attention type, has_attn_mask)`, with explicit fallback chains.
2. **Distinguishes attention by role** — self-attn, cross-attn, AR-prefill, AR-decode, streaming-decode are different problems and the framework should know that.
3. **Picks the right backend automatically** for the common case, while honoring CLI / YAML overrides for the expert case.
4. **Reuses upstream vLLM** for autoregressive attention rather than reimplementing PagedAttention.
5. **Stays close to upstream vLLM and SGLang patterns** to minimize divergence cost as the upstream patterns evolve.
6. **Lands incrementally** — each PR ships independent value, none requires a big-bang migration.
7. **Provides visibility** — engine init logs backend choices per attention type so operators can debug perf surprises.

## Non-Goals

This RFC does NOT propose:

- A complete LLM serving stack inside vLLM-OMNI. AR backends are integrated via adapter, not vendored. (See "AR/TTS Integration" below.)
- A new sparse attention algorithm. SpargeAttn (#888), STA, VSA — those are separate proposals; this RFC only ensures the framework can host them safely.
- Per-layer manual backend selection in user code. Layers declare their *type*, not their backend. The selector picks.
- Hardware-specific Backend classes for NPU / XPU / ROCm in this RFC. Per @xuechendi's proposal, those should be raised separately with platform-specific justification.

---

## Background

### Upstream vLLM's pattern

vLLM's selector lives in `vllm/platforms/cuda.py::get_attn_backend_cls`. The algorithm is:

1. If user provided a backend (`--attention-backend` or `VLLM_ATTENTION_BACKEND`), call its `validate_configuration()`. Error (don't silently fall back) if invalid.
2. Otherwise, walk every registered backend's `validate_configuration()`. Collect those returning `[]` (valid).
3. Among valid backends, pick the one with highest `priority`.

Each backend declares:
- `validate_configuration(device_capability, head_size, dtype, block_size, …) → list[str]`
- A priority value
- KV cache shape (LLM-specific; not relevant for diffusion)
- Metadata class

vLLM has ~20 backends across `vllm/v1/attention/backends/`, including transformer kernels (`flash_attn`, `flashinfer`, `triton_attn`), state-space (`mamba1_attn`, `mamba2_attn`), MLA, GDN, and TurboQuant.

### SGLang's diffusion pattern (`multimodal_gen`)

SGLang's diffusion attention lives in `python/sglang/multimodal_gen/runtime/layers/attention/`. The code header notes it was forked from FastVideo and adapted from vLLM v0.7.3. Key features:

- 15+ backends including 5+ sparse variants (STA, VSA, V-MoBA, SLA, SparseVideoGen2) and 2 quantized (Sage, Sage3)
- `AttentionMetadata.current_timestep` field — diffusion-specific, used by timestep-aware sparse strategies
- `--attention-backend-config` for parametric backends (sparsity, mask paths)
- Per-layer `supported_attention_backends` filter
- `@cache` on the selector
- `global_force_attn_backend_context_manager` for testing
- Platform abstraction (CUDA, ROCm, XPU, MUSA, MPS, NPU)
- Backend → Impl factory split (FastVideo lineage)

Notably, SGLang does **not** have separate cuDNN or FlashInfer backends for diffusion. They rely on PyTorch SDPA's internal dispatcher.

### Current vLLM-OMNI state (main branch, pre-#3079)

The system is more structured than the "flat if/elif on env var" framing suggests. Concretely:

**Files under `vllm_omni/diffusion/attention/`:**
- `layer.py` — `Attention` nn.Module orchestrating backend selection, parallel strategies, fallback
- `selector.py` — `@cache`'d `get_attn_backend(head_size)` that delegates to platform
- `backends/abstract.py` — `AttentionBackend`, `AttentionImpl`, `AttentionMetadata` ABCs
- `backends/registry.py` — `DiffusionAttentionBackendEnum` with three entries
- `backends/{flash_attn,sdpa,sage_attn,ring_flash_attn,ring_pytorch_attn}.py`
- `parallel/` — Ring / Ulysses / no-op sequence parallel strategies (orthogonal to backend)

**Backend ABC already includes:**
- `get_name()`, `get_impl_cls()`, `get_metadata_cls()`, `get_builder_cls()`, `get_supported_head_sizes()`
- `accept_output_buffer: bool`, `supports_attention_mask() -> bool` class flag

**Impl ABC already dispatches per platform** — this is the @xuechendi pattern from #1568, *already implemented*:
```python
class AttentionImpl(ABC):
    def forward(self, query, key, value, attn_metadata): ...
    def forward_cuda(self, ...): ...
    def forward_hip(self, ...): ...   # defaults to forward_cuda
    def forward_npu(self, ...): ...
    def forward_xpu(self, ...): ...
    def forward_musa(self, ...): ...  # defaults to forward_cuda
```

**Selector already delegates to per-platform code:**
```python
@cache
def get_attn_backend(head_size: int) -> type[AttentionBackend]:
    selected = os.environ.get("DIFFUSION_ATTENTION_BACKEND")
    return current_omni_platform.get_diffusion_attn_backend_cls(selected, head_size)
```

**Per-platform dispatch already in place:**
- `vllm_omni/platforms/cuda/platform.py::get_diffusion_attn_backend_cls` — checks compute capability (SM 8.0 ≤ CC < 10.0 today; #3079 relaxes this), checks `has_flash_attn` package, defaults FA→SDPA
- `vllm_omni/platforms/npu/platform.py` — `mindiesd` availability check
- ROCm / XPU / MUSA platforms — capability + package checks

**`AttentionMetadata` includes diffusion-specific fields:**
- `attn_mask`, `joint_attn_mask` — separate masks for the joint-attention case
- `joint_query`, `joint_key`, `joint_value` — prefix tensors for joint attention (e.g. text conditioning concatenated to image latents)
- `joint_strategy` — "front" or "rear" concatenation order

**Parallel strategies are orthogonal and already wired:**
- `NoParallelAttention`, `UlyssesParallelAttention`, `RingParallelAttention` in `parallel/`
- Each implements `pre_attention()` / `post_attention()` hooks around the backend's forward
- Layer composes a backend with a parallel strategy at construction
- This RFC inherits these unchanged

**What this RFC actually needs to add (vs replace):**

| Capability | Status today | This RFC |
|---|---|---|
| Backend ABC with `get_impl_cls()` | ✅ exists | extend with `validate()` / `priority()` / `supported_types()` |
| Per-platform Impl dispatch (`forward_cuda` etc.) | ✅ exists | unchanged; use as-is |
| `AttentionMetadata` with diffusion fields | ✅ exists | add `current_timestep`, subclass per type |
| Selector caching | ✅ exists (`@cache`) | extend cache key with new dimensions |
| Platform dispatch (`get_diffusion_attn_backend_cls`) | ✅ exists | extend with capability tier, has_attn_mask |
| Sequence-parallel layer composition | ✅ exists | unchanged; new typed layers inherit |
| Backend registry with override | ✅ exists | extend with type-aware filter |
| `validate()` + `priority()` selection | ❌ missing | **add** |
| `AttentionType` enum + per-type filter | ❌ missing | **add** |
| `has_attn_mask` priority signal | ❌ missing | **add** |
| `CapabilityTier` system | ❌ missing | **add** |
| YAML per-model defaults | ❌ missing | **add** |
| CLI `--attention-backend` | ❌ missing (uses env var) | **add** |
| `--attention-backend-config` parametric config | ❌ missing | **add** |
| Cross-attention K/V cache | ❌ missing | **add** |
| AR/TTS via vLLM adapter | ❌ missing | **add** |

This is a much smaller refactor than a clean-slate redesign. The bones are good; we're filling in selection intelligence and surfacing it through CLI/YAML.

### Why vLLM-OMNI is different

Diffusion attention has properties that LLM serving does not:

| | LLM serving | Diffusion |
|---|---|---|
| KV cache | Persistent across decode steps | None |
| Per-call shape | Variable (decode grows seq) | Fixed per resolution |
| Bottleneck | Memory bandwidth | Compute |
| Typical seq_len | 1k–32k | 30k–100k+ for video |
| Attention pattern | Causal | Bidirectional |
| Backend selection time | Engine init (one shot) | Engine init (one shot) |
| Backend per call | Same for all calls | Same for all calls |

The primary takeaway: KV cache machinery doesn't apply, but the multi-backend selection problem does, and the validator+priority pattern transfers cleanly.

For omni / TTS workloads, vLLM-OMNI inherits LLM serving's properties. Rather than reimplement, this RFC integrates upstream vLLM's PagedAttention as one of the registered backends via an adapter.

---

## Proposed Design

### 1. Capability Tier System

Define a `CapabilityTier` enum derived from `torch.cuda.get_device_capability()`:

```python
# vllm_omni/attention/capability.py
from enum import Enum

class CapabilityTier(Enum):
    AMPERE              = "ampere"               # sm_80, sm_86
    ADA                 = "ada"                  # sm_89
    HOPPER              = "hopper"               # sm_90
    BLACKWELL_DC        = "blackwell_dc"         # sm_100, sm_103 (B200, B300)
    BLACKWELL_CONSUMER  = "blackwell_consumer"   # sm_120, sm_121 (RTX 5090, RTX Pro 6000, GB10)
    CPU                 = "cpu"
    UNKNOWN             = "unknown"

def detect_tier() -> CapabilityTier:
    if not torch.cuda.is_available():
        return CapabilityTier.CPU
    cc = torch.cuda.get_device_capability()
    if cc >= (12, 0): return CapabilityTier.BLACKWELL_CONSUMER
    if cc >= (10, 0): return CapabilityTier.BLACKWELL_DC
    if cc >= (9, 0):  return CapabilityTier.HOPPER
    if cc == (8, 9):  return CapabilityTier.ADA
    if cc >= (8, 0):  return CapabilityTier.AMPERE
    return CapabilityTier.UNKNOWN
```

Backends declare priority per `(tier, attention_type, has_attn_mask)`:

```python
class CudnnAttnBackend(AttentionBackend):
    _PRIORITY_TABLE = {
        # (tier, attention_type, has_mask): priority
        (CapabilityTier.BLACKWELL_CONSUMER, DIFF_SELF, True):  90,  # HV-1.5 2x win
        (CapabilityTier.BLACKWELL_CONSUMER, DIFF_SELF, False): 60,
        (CapabilityTier.BLACKWELL_CONSUMER, DIFF_CROSS, True): 80,
        (CapabilityTier.BLACKWELL_DC,       DIFF_SELF, True):  85,  # extrapolation
        (CapabilityTier.BLACKWELL_DC,       DIFF_SELF, False): 60,
        (CapabilityTier.HOPPER,             DIFF_SELF, True):  70,  # extrapolation, mask-heavy still favors cuDNN
        (CapabilityTier.HOPPER,             DIFF_SELF, False): 50,  # FA3 wins unmasked
        (CapabilityTier.AMPERE,             DIFF_SELF, True):  40,  # FA2 fine, cuDNN not validated
        (CapabilityTier.AMPERE,             DIFF_SELF, False): 30,
    }
    
    @classmethod
    def priority(cls, *, attention_type, has_attn_mask, capability_tier, **_):
        return cls._PRIORITY_TABLE.get(
            (capability_tier, attention_type, has_attn_mask), 0
        )
```

**Validation matrix.** Each `(backend, tier)` cell in the priority table must be either measured or marked as extrapolation. Until measured, extrapolated values are conservative. The `benchmarks/diffusion/bench_attention_backends.py` script added in #3079 should be the basis for filling in cells across hardware.

**Within-tier divergence.** Reviewer @frauttauteffasu's measurements on sm_121 show **FlashInfer wins HV-1.5 by 1.41× vs cuDNN's 0.72× normalized**, the opposite of #3079's sm_120 e2e result where cuDNN wins by 2.01×. So `BLACKWELL_CONSUMER` may need to split further (sm_120 vs sm_121) or the priority table may need a third dimension (`gpu_subarch`). Recommend deferring the split until we have enough sm_121 data to justify it; for now, treat sm_121 as `BLACKWELL_CONSUMER` and flag the divergence in the validation matrix.

**cuDNN version requirement.** Native Blackwell SDPA actually requires **cuDNN ≥ 9.7.0** (not 9.5 as currently stated in #3079). cuDNN 9.5/9.6 only cover Hopper and earlier. Validators should check this exact cutoff. FP8 SDPA forward on Blackwell needs cuDNN ≥ 9.13.0; MXFP8/NVFP4 needs 9.20+.

### 2. AttentionType Enum + Per-Type Filter

```python
# vllm_omni/attention/types.py
from enum import Enum, auto

class AttentionType(Enum):
    DIFFUSION_SELF       = auto()    # bidirectional self-attn on latents (FLUX, Wan, LTX-2 video)
    DIFFUSION_CROSS      = auto()    # bidirectional cross-attn (text/audio → video)
    AR_PREFILL           = auto()    # causal, KV cache build (TTS, omni-thinker prefill)
    AR_DECODE            = auto()    # causal, KV cache append
    STREAMING_DECODE     = auto()    # AR_DECODE with latency-tuned kernel
```

Each backend declares which types it supports:

```python
class STABackend(AttentionBackend):
    @classmethod
    def supported_types(cls):
        return {AttentionType.DIFFUSION_SELF}    # spatial-only

class FlashAttnBackend(AttentionBackend):
    @classmethod
    def supported_types(cls):
        return {AttentionType.DIFFUSION_SELF,
                AttentionType.DIFFUSION_CROSS,
                AttentionType.AR_PREFILL}

class VllmPagedAttnAdapter(AttentionBackend):
    @classmethod
    def supported_types(cls):
        return {AttentionType.AR_PREFILL,
                AttentionType.AR_DECODE}
```

The selector filters by type before applying validate+priority. Cross-attention layers cannot silently inherit a self-attention-only backend.

Layer hierarchy mirrors the types:

```python
# vllm_omni/attention/layers/
class AttentionLayer(nn.Module):
    attention_type: AttentionType    # subclass sets this
    
    def __init__(self, *, num_heads, head_dim, num_kv_heads=None,
                 supported_backends=None, backend_config=None, **kwargs):
        super().__init__()
        backend_cls = get_attn_backend(
            attention_type=self.attention_type,
            head_dim=head_dim,
            dtype=current_compute_dtype(),
            capability_tier=detect_tier(),
            has_attn_mask=self._declares_mask(),
            cli_override=resolve_cli_override(),
            yaml_default=resolve_yaml_default(),
            supported_filter=frozenset(supported_backends) if supported_backends else None,
            config_frozen=tuple(sorted((backend_config or {}).items())),
        )
        impl_cls = backend_cls.get_impl_cls(self.attention_type)
        self.impl = impl_cls(
            attention_type=self.attention_type,
            num_heads=num_heads, head_dim=head_dim,
            num_kv_heads=num_kv_heads or num_heads,
            causal=self._causal_default(), config=backend_config,
        )

class DiffusionSelfAttention(AttentionLayer):
    attention_type = AttentionType.DIFFUSION_SELF
    def _declares_mask(self): return False

class DiffusionCrossAttention(AttentionLayer):
    attention_type = AttentionType.DIFFUSION_CROSS
    def _declares_mask(self): return True    # text padding mask always present

class AutoregressiveAttention(AttentionLayer):
    attention_type = AttentionType.AR_DECODE    # or AR_PREFILL via constructor

class StreamingAttention(AttentionLayer):
    attention_type = AttentionType.STREAMING_DECODE
```

Model files construct the right layer subclass for each role. There is no `if backend == "X"` in model code.

### 3. Mask-Aware Priority

The `has_attn_mask: bool` flag (§1 example) is plumbed from the layer through the selector to `Backend.priority()`. This is small but directly justified by #3079 measurements:

- HunyuanVideo-1.5 480p / 33f / 50 steps: TORCH_SDPA = 147.05s, **CUDNN_ATTN = 73.02s**, FLASHINFER_ATTN = 127.84s. **2.01× win for cuDNN.**
- Microbench root cause: PyTorch SDPA's unpinned dispatcher selects `EFFICIENT_ATTENTION` (~25 ms) for masked attention calls instead of cuDNN (~11 ms). Without the mask, both paths tie within noise.

So the rule is: when a layer declares it's wired to a mask source, the selector should prefer backends that handle masks well (cuDNN on Blackwell, eventually FA4 elsewhere). Without the flag, the selector either over-prefers cuDNN globally (losing to FA3 on unmasked self-attn paths) or under-prefers it (losing the HV-1.5 2× win).

### 4. Backend → Impl Factory with Platform Dispatch

**This pattern already exists in main branch.** `AttentionImpl` already has `forward_cuda` / `forward_hip` / `forward_npu` / `forward_xpu` / `forward_musa` methods, and `AttentionBackend` already has `get_impl_cls()`. What this RFC adds is selection metadata on the `Backend` descriptor (`validate()`, `priority()`, `supported_types()`) and the consolidation of per-platform construction logic.

The full ABC, with extensions over the current state highlighted:

```python
# vllm_omni/attention/backends/abstract.py
class AttentionBackend(ABC):
    """Static class. Registry uses get_*() classmethods. Never instantiated."""
    
    @classmethod
    @abstractmethod
    def name(cls) -> str: ...
    
    @classmethod
    @abstractmethod
    def supported_types(cls) -> set[AttentionType]: ...
    
    @classmethod
    @abstractmethod
    def validate(cls, *, attention_type, capability_tier, head_dim, dtype,
                 has_attn_mask=False, config=None, **_) -> list[str]: ...
    
    @classmethod
    @abstractmethod
    def priority(cls, *, attention_type, capability_tier, has_attn_mask,
                 **_) -> int: ...
    
    @classmethod
    @abstractmethod
    def get_impl_cls(cls, attention_type: AttentionType) -> type["AttentionImpl"]: ...


class AttentionImpl(ABC):
    """Per-layer instance. Constructed once at layer __init__."""
    
    @abstractmethod
    def __init__(self, *, attention_type, num_heads, head_dim,
                 num_kv_heads=None, causal=False, softmax_scale=None,
                 dropout_p=0.0, config=None): ...
    
    @abstractmethod
    def forward(self, q, k, v, metadata: AttentionMetadata) -> torch.Tensor: ...
```

Concrete Impl dispatches by platform inside `forward()`:

```python
# vllm_omni/attention/backends/flash_attn.py
class FlashAttnImpl(AttentionImpl):
    def forward(self, q, k, v, metadata):
        if current_platform.is_cuda():
            return self.forward_cuda(q, k, v, metadata)
        elif current_platform.is_xpu():
            return self.forward_xpu(q, k, v, metadata)
        elif current_platform.is_npu():
            return self.forward_npu(q, k, v, metadata)
        elif current_platform.is_rocm():
            return self.forward_rocm(q, k, v, metadata)
        raise NotImplementedError(f"FlashAttn on {current_platform}")
    
    def forward_cuda(self, q, k, v, metadata):
        from vllm_omni.attention.ops._cuda_ops import flash_attn_func
        return flash_attn_func(q, k, v, ...)
    
    def forward_xpu(self, q, k, v, metadata):
        from vllm_omni.attention.ops._xpu_ops import flash_attn_func
        return flash_attn_func(q, k, v, ...)
    # ... npu, rocm
```

Per-platform shared func calls live in `vllm_omni/attention/ops/`:

```
vllm_omni/attention/ops/
├── _cuda_ops.py
├── _xpu_ops.py
├── _ascend_ops.py
└── _aiter_ops.py
```

Each `_<platform>_ops.py` exposes `flash_attn_func`, `cudnn_sdpa_func`, `flashinfer_prefill_func` etc. with shared input arguments. Platform-specific kernel calls and parameter translation live there.

**When does a hardware-specific Backend class (e.g. `AscendFlashAttnBackend`) get justified?** Per @xuechendi's decision rule: if `forward_<platform>` cannot be made compatible because of graph/compile pattern differences or attn_metadata structural differences, file a separate platform RFC arguing why and propose the new Backend class. The default is "unify in Impl, dispatch in forward."

### 5. YAML Per-Model Defaults + CLI

Model configurations declare their preferred default backend. Priority chain: **CLI > YAML > auto**.

```yaml
# configs/hunyuan_video.yaml
model:
  name: HunyuanVideo-1.5
  attention:
    default_backend: CUDNN_ATTN
    backend_config:
      CUDNN_ATTN: {}    # cuDNN-specific tunables, if any
      STA_ATTN:
        sparsity: 0.5
        skip_time_steps: 15
```

```yaml
# configs/wan_2_2.yaml
model:
  name: Wan-2.2-14B
  attention:
    default_backend: null    # let auto-pick decide
```

CLI:

```bash
vllm-omni serve \
  --model HunyuanVideo \
  --attention-backend CUDNN_ATTN \
  --attention-backend-config 'STA_ATTN={"sparsity":0.5}'
```

Env var compatibility: `DIFFUSION_ATTENTION_BACKEND` is honored for one release with a deprecation warning, then removed. This matches upstream vLLM's `VLLM_ATTENTION_BACKEND` deprecation pattern (per @tjtanaa in #1568).

The selector (skeleton):

```python
# vllm_omni/attention/selector.py
@cache
def get_attn_backend(
    attention_type, head_dim, dtype, capability_tier,
    has_attn_mask=False, cli_override=None, yaml_default=None,
    supported_filter=None, config_frozen=None,
) -> type[AttentionBackend]:
    
    # 1. CLI override → must be valid or error
    if cli_override is not None:
        backend = registry.get(cli_override)
        if backend is None:
            raise ValueError(f"Unknown backend: {cli_override}")
        if attention_type not in backend.supported_types():
            raise ValueError(
                f"{cli_override} does not support {attention_type.name}"
            )
        reasons = backend.validate(
            attention_type=attention_type, capability_tier=capability_tier,
            head_dim=head_dim, dtype=dtype, has_attn_mask=has_attn_mask,
            config=dict(config_frozen) if config_frozen else None,
        )
        if reasons:
            raise ValueError(
                f"{cli_override} invalid for {attention_type.name}: {reasons}"
            )
        logger.info("Using CLI-selected backend %s for %s",
                    cli_override, attention_type.name)
        return backend
    
    # 2. YAML default → validate, fall through if invalid
    if yaml_default is not None:
        backend = registry.get(yaml_default)
        if backend and attention_type in backend.supported_types():
            reasons = backend.validate(...)
            if not reasons:
                logger.info("Using YAML-default backend %s for %s",
                            yaml_default, attention_type.name)
                return backend
            logger.warning(
                "YAML default %s invalid for %s on %s: %s. Auto-picking.",
                yaml_default, attention_type.name, capability_tier.value, reasons,
            )
    
    # 3. Auto: filter by type, validate, pick highest priority
    candidates = [
        b for b in registry.list_all()
        if attention_type in b.supported_types()
        and (supported_filter is None or b.name() in supported_filter)
    ]
    valid = []
    for b in candidates:
        reasons = b.validate(
            attention_type=attention_type, capability_tier=capability_tier,
            head_dim=head_dim, dtype=dtype, has_attn_mask=has_attn_mask,
            config=dict(config_frozen) if config_frozen else None,
        )
        if not reasons:
            valid.append((b, b.priority(
                attention_type=attention_type, capability_tier=capability_tier,
                has_attn_mask=has_attn_mask,
            )))
    if not valid:
        raise ValueError(
            f"No valid backend for {attention_type.name} on "
            f"{capability_tier.value} with head_dim={head_dim}, dtype={dtype}"
        )
    chosen = max(valid, key=lambda x: x[1])[0]
    logger.info(
        "Auto-selected %s for %s (tier=%s, head_dim=%d, dtype=%s, has_mask=%s)",
        chosen.name(), attention_type.name, capability_tier.value,
        head_dim, dtype, has_attn_mask,
    )
    return chosen
```

Plus a context manager for testing (`force_backend(name, attention_type=None)`) that overrides for a scope and restores.

### 6. Backend-Specific Configuration

`--attention-backend-config` accepts JSON, YAML file path, or `key=value` pairs scoped per backend:

```bash
# JSON
--attention-backend-config 'STA_ATTN={"sparsity":0.5,"mask_strategy_file_path":"/p/mask.json"}'

# YAML file
--attention-backend-config /etc/vllm-omni/attn_config.yaml

# Multiple backends (comma-separated)
--attention-backend-config 'STA_ATTN={"sparsity":0.5},SAGE_ATTN={"variant":"int8"}'
```

The parsed config is stored on the engine and threaded into `Backend.validate()` and `AttentionImpl.__init__()`. Each backend documents its expected schema in its docstring; future work can add JSON schema validation per backend.

### 7. Cross-Attention K/V Caching

For diffusion cross-attention, the K/V tensors are derived from text/audio encoder outputs that are **constant across all denoising steps for a given request**. Today vLLM-OMNI recomputes K/V projections each step. The cache below avoids that:

```python
class DiffusionCrossAttention(AttentionLayer):
    attention_type = AttentionType.DIFFUSION_CROSS
    
    def __init__(self, *, cache_static_kv: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.cache_static_kv = cache_static_kv
        self._kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    
    def forward(self, q, kv_source, metadata: DiffusionCrossMetadata):
        cache_key = metadata.cache_key
        if self.cache_static_kv and cache_key in self._kv_cache:
            k, v = self._kv_cache[cache_key]
        else:
            k = self.k_proj(kv_source)
            v = self.v_proj(kv_source)
            if self.cache_static_kv and cache_key is not None:
                self._kv_cache[cache_key] = (k, v)
        return self.impl.forward(q, k, v, metadata)
    
    def invalidate_cache(self, cache_key=None):
        if cache_key is None:
            self._kv_cache.clear()
        else:
            self._kv_cache.pop(cache_key, None)
```

The engine invalidates per-request entries on request completion. A global cap (`--cross-attn-cache-mb`) prevents memory blowup. Estimated win: **5–15% per step** on long-step generation (50–500 steps), depending on cross-attn fraction of step time.

This is independent of backend choice and orthogonal to everything else in this RFC, but lands cleanly with the typed-layer pattern.

### 8. AR/TTS Integration via vLLM Adapter

For autoregressive attention (TTS, omni-thinker), this RFC integrates upstream vLLM's PagedAttention rather than reimplementing it. The adapter is registered as a normal backend:

```python
# vllm_omni/attention/backends/adapters/vllm_paged.py
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend as VllmFA,
    FlashAttentionImpl as VllmFAImpl,
)
from vllm_omni.attention.registry import register

@register
class VllmFlashAttnAdapter(AttentionBackend):
    @classmethod
    def name(cls): return "VLLM_FA"
    
    @classmethod
    def supported_types(cls):
        return {AttentionType.AR_PREFILL, AttentionType.AR_DECODE}
    
    @classmethod
    def validate(cls, *, capability_tier, head_dim, dtype, **_):
        # Defer to upstream vLLM's checks
        return _adapt_vllm_validation(VllmFA, capability_tier, head_dim, dtype)
    
    @classmethod
    def priority(cls, *, attention_type, capability_tier, **_):
        # High for AR types; not a candidate for diffusion
        return 80 if attention_type in cls.supported_types() else 0
    
    @classmethod
    def get_impl_cls(cls, attention_type): return VllmFlashAttnAdapterImpl


class VllmFlashAttnAdapterImpl(AttentionImpl):
    """Translates vLLM-OMNI ARMetadata to vLLM's metadata, calls upstream Impl."""
    def __init__(self, **kwargs):
        # ... construct upstream Impl with translated args
        self._inner = VllmFAImpl(...)
    
    def forward(self, q, k, v, metadata):
        vllm_meta = _translate_metadata(metadata)
        return self._inner.forward(q, k, v, vllm_meta)
```

The adapter is ~300–500 LoC, mostly metadata translation. vLLM is a pinned dependency in vLLM-OMNI's `pyproject.toml`; adapter compatibility is enforced per pinned version.

This unblocks VoxCPM2 (#2594) and omni-thinker integration without forking PagedAttention into vLLM-OMNI.

---

## Implementation Plan

Sequenced as 12 PRs after #3079 lands. The plan front-loads the LTX-2 regression unblock, then adds selection intelligence on top of the existing Backend/Impl/Platform structure (which is already most of what we need — see "Current vLLM-OMNI state" above).

| # | PR Title | Scope | Depends On | Risk |
|---|---|---|---|---|
| 0a | (existing) #3079 — Blackwell CUDNN_ATTN + FLASHINFER_ATTN | in flight | — | — |
| 0b | (suggested follow-up to #3079) Fix Codex P1s: batch-mask indexing, additive-mask sentinel, FlashInfer probe | ~150 LoC | #3079 | LOW |
| 1 | Fix LTX-2 cuDNN + torch.compile crash (#3121, supersedes fork PR #54). Detect symbolic-shape compile path; fall back to FlashInfer custom_mask only for that narrow case to avoid HV-1.5 2× regression. | ~80 LoC + regression test | #3079 | MEDIUM (cuDNN hot path) |
| 2 | `CapabilityTier` enum + classifier (no behavior change yet — additive) | ~150 LoC | PR 1 | LOW |
| 3 | Add `has_attn_mask` flag to `Backend.priority()` signature (default branch keeps non-mask byte-identical) | ~250 LoC | PR 2 | LOW-MEDIUM |
| 4 | Per-tier priority tables on existing 5 backends; remove `cc >= 100` special-case | ~300 LoC | PRs 2, 3 | MEDIUM (must reproduce #3079 auto-route on sm_120 + not regress sm_90/89/86) |
| 5 | CLI `--attention-backend` + deprecate `DIFFUSION_ATTENTION_BACKEND` env var (one-release deprecation window) | ~200 LoC | PR 4 | LOW |
| 6 | **RFC sign-off gate** — `AttentionType` enum + `Backend.supported_types()` filter; layers declare type at construction. All current diffusion backends declare `{DIFFUSION_SELF, DIFFUSION_CROSS}` → no behavior change | ~400 LoC | PR 4 + RFC #1568 sign-off from NPU/XPU/ROCm leads | MEDIUM (touches every Attention layer ctor) |
| 7 | YAML per-model defaults + initial YAMLs for HV-1.5, Wan 2.2, FLUX.2, Z-Image, LTX-2 | ~350 LoC + N config files | PRs 5, 6 | MEDIUM (wrong YAML default silently degrades a model — gate each YAML on the e2e bench from #3079) |
| 8 | `--attention-backend-config k=v / JSON / YAML` parametric backend config | ~300 LoC | PRs 5, 7 | LOW-MEDIUM |
| 9 | Cross-attention K/V cache for static text-encoder conditioning | ~450 LoC + reference-seed regression tests | PR 6 | MEDIUM (cache invalidation rules) |
| 10 | **Second sign-off gate** — formalize Backend descriptor / Impl split. The `forward_cuda/forward_hip/forward_npu/forward_xpu/forward_musa` pattern *already exists* in `AttentionImpl`; this PR moves backend-class config (priorities, supported_types) onto `Backend` as a thin descriptor and consolidates per-platform construction. Ships behind a `VLLM_OMNI_USE_NEW_BACKEND_FACTORY=1` flag for one release. | ~700 LoC | PRs 6, 8 + RFC sign-off | HIGH (touches every backend file) |
| 11 | Shared ops layer `vllm_omni/attention/ops/_hw_ops.py` — extract `_maybe_reshape_attn_mask`, `broadcast_to((qo_len, kv_len))`, head-dim padding, dtype coercion. Pure DRY pass after PR 10's split exposes duplication. | ~400 LoC | PR 10 | LOW |
| 12 | vLLM PagedAttention adapter (`VLLM_FA`) for AR_PREFILL / AR_DECODE / STREAMING_DECODE — wraps `BatchPrefillWithPagedKVCacheWrapper` + `plan()` | ~600 LoC + integration test | PRs 6, 10, 11 | MEDIUM |
| 13 | Engine-level `_log_backend_decisions()` startup summary | ~100 LoC | PR 6 | LOW |

**Sequencing notes:**

- **PRs 1–5 are pre-RFC-signoff** — extensions of #3079's design that don't lock in #1568 structural choices. Can land while #1568 is still under discussion.
- **PR 6 is the first gate** — needs explicit ack from NPU + XPU + ROCm reviewers on #1568 before merging.
- **PR 10 is the second gate** — load-bearing structural change. Ship behind a flag, require multi-platform CI green before flipping the default.
- **Production-break risk concentrated in PRs 4, 7, 10**. Each must include the full #3079 e2e bench table (HV-1.5, Wan 2.2, Qwen-Image, FLUX.2, Z-Image, LTX-2) in the PR body, with no regression > 3% vs the #3079 baseline.
- **Pure refactor PRs (2, 11)** are deliberately small and ship alongside value-adding PRs to avoid the "refactor train wreck" pattern.
- PR 7 can split into 7a (loader + schema) and 7b (per-model YAMLs) if any single YAML default proves controversial during review.

**Total scope:** ~4000–4500 LoC plus tests and docs. Smaller than the original "redesign from scratch" estimate because the existing Backend/Impl/Platform structure already covers ~60% of what's needed.

---

## Test Matrix

Each backend should be benchmarked on at least one representative card per CapabilityTier before its priorities are trusted. Suggested cards:

| Tier | Card | Status |
|---|---|---|
| AMPERE | A100 80GB | Need data |
| ADA | RTX 4090 | Need data |
| HOPPER | H100 80GB | Need data |
| BLACKWELL_DC | B200 / B300 | Need data (#nvfp4-hv15-wan22-status mentions B300 NaN regression) |
| BLACKWELL_CONSUMER | RTX Pro 6000 / RTX 5090 | ✅ Covered by #3079 |

Per-tier benchmark sweep:

| Model | Resolution | Steps | Backends to compare | Mask path? |
|---|---|---|---|---|
| HunyuanVideo-1.5 | 480p / 33f | 50 | TORCH_SDPA, FLASH_ATTN, CUDNN_ATTN, FLASHINFER_ATTN, SAGE_ATTN | Yes |
| Wan 2.2 14B | 480p / 33f | 40 | (same) | Light |
| FLUX.2-dev | 1024² | 50 | (same) | Yes (text padding) |
| Z-Image-Turbo | 1024² | 8 | (same) | Light |
| LTX-2.0 | 480p / 97f | 500 | TORCH_SDPA, FLASH_ATTN, FLASHINFER_ATTN | (cuDNN excluded pending #3121) |
| Qwen-Image | 1024² | 50 | (same as HV-1.5) | Yes |

Microbench (`benchmarks/diffusion/bench_attention_backends.py`) should also run per-tier to isolate kernel-level wins from e2e.

CI integration: a smaller subset (one model per tier × top-3 backends) runs nightly. Full sweep runs weekly or per-release.

---

## Migration & Backwards Compatibility

### Env var deprecation

`DIFFUSION_ATTENTION_BACKEND` continues to work for one release after PR 9 lands, with a deprecation warning at engine init. After that release, it's removed. Users must migrate to `--attention-backend`.

### Existing models

Models that don't migrate to typed layers (PRs 5–7) continue to use the legacy `DiffusionAttention` class via a compatibility shim. The shim is removed after all upstream models migrate.

### Backend renaming

No backend renames. `TORCH_SDPA`, `FLASH_ATTN`, `CUDNN_ATTN`, `FLASHINFER_ATTN`, `SAGE_ATTN` keep their current names.

### YAML opt-in

Per-model YAML config is opt-in. Models without a YAML config use auto-pick. No model is forced to ship a YAML.

---

## Reconciling with In-Flight PR #3079

PR #3079 ships material value (Blackwell auto-route, 2× HV-1.5 win, microbench harness) but has five places where its current design choices contradict the directions converged on in #1568. This RFC aims to reconcile rather than block:

1. **CLI vs env var.** PR #3079 still uses `DIFFUSION_ATTENTION_BACKEND` env var and adds two more backend names to it. RFC #1568 (per @tjtanaa, with thumbs-up) has agreed to migrate to `--attention-backend` CLI arg matching upstream vLLM. **Reconciliation:** PR #3079 lands as-is; PR 9 (CLI migration) immediately follows with one-release env var deprecation window.
2. **Coarse-grained backends vs internal dispatch.** PR #3079 introduces `CuDNNAttentionBackend` and `FlashInferAttentionBackend` as separate classes. @yma11 in #1568 (with 2 thumbs-up): "backends should not be classified so detailed... still one `FlashAttnBackend`." @Isotr0py raised the same point on PR #3079: "Can we automatically select SDPA kernel for SDPAImpl instead?" **Reconciliation:** keep PR #3079's separate classes initially since they're already validated; PR 11 (factory split) consolidates the dispatch internally per @xuechendi's `forward_<platform>` pattern.
3. **YAML defaults.** PR #3079 hardcodes Blackwell auto-route in `vllm_omni/platforms/cuda/platform.py`. RFC #1568 wants per-model YAML config. **Reconciliation:** PR 8 ports the hardcoded auto-route to YAML defaults, making the LTX-2 exception declarable rather than code-branched.
4. **Per-model selection.** PR #3079 has only global Blackwell auto-route. @david6666666 raised the LTX-2 regression concern: making cuDNN the Blackwell default crashes LTX-2 (#3121). **Reconciliation:** PR 8's YAML defaults let LTX-2 declare `default_backend: FLASHINFER_ATTN` until #3121 is fully resolved.
5. **Hardware-specific backends.** PR #3079 backends are CUDA-only with no `forward_xpu` / `forward_npu` / `forward_rocm`. RFC #1568 has an open question (raised by @xuechendi, not yet resolved) on whether HW-specific backend classes are a hard no. **Reconciliation:** the factory-split PR (PR 11) provides the platform dispatch surface; non-CUDA implementations land in subsequent PRs as their teams have bandwidth, with @xuechendi's "alternative func_call first, platform Backend with separate RFC otherwise" gate.

Additionally, PR #3079 has three open Codex P1/P2 issues that should be addressed before this RFC's PRs build on its code:

- **Batch-specific masks dropped** in `_pack_mask_for_flashinfer` — `mask[0]` indexing while `mask.dim() > 2` silently selects only batch-0 slice. Breaks CFG and mixed prompt-length batches.
- **Additive masks misinterpreted** — `mask != float("-inf")` treats `-10000` (common additive mask sentinel) as valid → makes FlashInfer attend to padded positions, diverges from SDPA reference.
- **FlashInfer probe only catches `ImportError`** — partial install / ABI mismatch can raise `OSError`/`RuntimeError` and abort startup. Should treat any import-time failure as unavailable.

These should land in PR #3079 itself or in an immediate follow-up before this RFC's framework sequence begins.

---

## Open Questions

1. **Should `vllm_omni/diffusion/attention/` move to `vllm_omni/attention/`?** Promoting out of the diffusion subpath signals that the system serves AR/TTS too. Costs: import-path churn for users.

2. **Internal SDPAImpl dispatch vs separate Backend classes** (raised by @Isotr0py on PR #3079, echoes @yma11 in #1568). Two options:
   - **Option A (PR #3079 today):** `CuDNNAttentionBackend`, `FlashInferAttentionBackend` as siblings to `SDPABackend`. Simple, explicit, validated.
   - **Option B (RFC #1568 direction):** one `SDPABackend` whose `Impl.forward_cuda` internally dispatches to `sdpa_kernel([CUDNN_ATTENTION])` / FlashInfer / cuDNN-direct based on `(tier, has_mask, has_compile)`. Fewer classes, harder to reason about, no separate priority slot.
   
   This RFC takes Option A then consolidates per the factory split (PR 11). Open whether the consolidation should go further and merge the three classes into one `SDPABackend` with deeper internal dispatch — depends on whether the cuDNN/FlashInfer/SDPA distinction stays meaningful at the user-facing CLI.
2. **vLLM version pinning policy for the adapter.** Does vllm-omni already pin a vllm version, or float it? If float, the adapter needs version detection or compatibility shims.
3. **Should sparse backends (STA, VSA, SpargeAttn) be in scope for this RFC or follow-ups?** This RFC ensures they *can* be added safely; it doesn't mandate when. Suggest follow-up RFCs per algorithm.
4. **NPU / XPU / ROCm priority tables.** Do we extrapolate from CUDA priorities or ask hardware teams to fill in their tiers? Suggest the latter — contact @gcanlin / @hsliuustc0106 / @yma11 / @xuechendi to populate their entries.
5. **FA4 integration timeline.** Currently broken on sm_120 (`flash-attn-4 4.0.0b10`, JIT issue). Wait for stable, or wire in optimistically with a feature flag?
6. **Cross-attention cache invalidation.** Per-request invalidation is straightforward. What about CFG-with-changing-conditioning models or batch reuse? Worth specifying explicitly in PR 13.
7. **Should `STREAMING_DECODE` be in this RFC or deferred?** It only matters for TTS latency-bound serving. Defer if VoxCPM2 is offline-batch.

---

## Alternatives Considered

### A. Keep the env-var selector, add tier-based defaults inside `cuda/platform.py`

This is the path of least resistance — no abstraction changes, just smarter defaults. Rejected because:
- Doesn't address backend variety / parameterization needs from #1568
- Doesn't provide per-attention-type discrimination needed for sparse + AR
- Pushes hardware team coordination into one giant `if/elif` tree

### B. Adopt SGLang's `multimodal_gen` attention layer wholesale

Rejected because:
- SGLang's code is FastVideo-derived with idioms (`accept_output_buffer`, `wrap_attention_impl_forward`) that don't fit cleanly
- vLLM-OMNI is an Apache-2.0 project under vLLM org; preferred lineage is vLLM upstream patterns
- SGLang has per-platform Backend classes; @xuechendi's #1568 proposal (which we adopt) prefers unified Impl with platform dispatch

### C. Build a complete LLM serving stack inside vLLM-OMNI

Rejected because:
- PagedAttention is mature, complex, and well-maintained upstream
- Forking it into vLLM-OMNI is a permanent maintenance burden
- The adapter pattern (§8) costs a few hundred LoC and a vLLM version pin

### D. Defer the AttentionType abstraction until sparse backends actually land

Rejected because:
- Sparse backends are explicitly mentioned in #1568 (`SpargeAttentionBackend` per #888, mindiesd sparse variants)
- Adding the type abstraction after sparse lands means migrating sparse backend internals — much bigger change
- The cost of adding AttentionType today is low (new enum, new layer base classes); the cost of adding it later is high

---

## References

### vLLM-OMNI
- PR #3079 — [Blackwell] Add CUDNN_ATTN and FLASHINFER_ATTN backends for diffusion (auto-route)
- Issue #1568 — [RFC]: Discussing the extension of attention backend
- Issue #3121 — LTX-2 cuDNN crash under torch.compile (symbolic head_dim)
- PR #54 (lishunyang12 fork) — [Bugfix] enable CUDNN_ATTN on LTX-2.0 by setting explicit head_dim
- PR #888 — SpargeAttentionBackend
- PR #2594 — VoxCPM2 model support

### Upstream
- vLLM `vllm/platforms/cuda.py::get_attn_backend_cls` — validator+priority pattern
- vLLM `vllm/v1/attention/backends/` — backend implementations (~20)
- SGLang `python/sglang/multimodal_gen/runtime/layers/attention/` — diffusion attention (forked from FastVideo, adapted from vLLM v0.7.3)

### External
- FlashAttention — github.com/Dao-AILab/flash-attention
- FlashInfer — github.com/flashinfer-ai/flashinfer
- SageAttention — github.com/thu-ml/SageAttention
- cuDNN SDPA — docs.nvidia.com/deeplearning/cudnn

---

## Changelog

- **2026-04-26**: Initial draft. Integrated findings from current-state audit (existing `forward_cuda/forward_hip/forward_npu/forward_xpu/forward_musa` per-platform Impl dispatch), PR #3079 + #1568 deep dive (5 contradictions to reconcile, 3 Codex P1s to address, frauttauteffasu sm_121 divergence data, Isotr0py architectural question), hardware compatibility matrix (cuDNN 9.7 not 9.5; FA4 broken on sm_120 specifically; FlashInfer cutlass/fa3/trtllm-gen unsupported on sm_120/121), and migration sequence planning (12 PRs, two RFC sign-off gates at PR 6 and PR 10).
