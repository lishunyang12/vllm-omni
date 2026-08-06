# CPU Offloading for Diffusion Models

## Overview

vLLM-Omni provides three offloading strategies to reduce accelerator memory
usage for diffusion models:

1. **Model-level (sequential) offloading** swaps whole DiT and encoder
   components between the host and accelerator.
2. **Layerwise (blockwise) offloading** streams one transformer block at a
   time on a single rank.
3. **Distributed layerwise offloading (DLO)** adds shared double buffers and
   either reconstructs DP-sharded blocks with AllGather or streams the
   standard loader's rank-local weights without AllGather.

All three strategies use pinned host memory for faster transfers. They are
mutually exclusive; distributed layerwise takes priority over layerwise, and
layerwise takes priority over model-level offloading.


## Model-level (Sequential) Offloading

### How It Works

Model-level offloading implements mutual exclusion between DiT transformer and encoder modules using pre forward hooks:

- **When encoders run**: DiT transformer is offloaded to CPU
- **When DiT runs**: Encoders are offloaded to CPU, if more than one dit models, only one loaded on GPU, others get offloaded to CPU.
- **VAE**: Stays resident on GPU

Before each module's forward pass, the hook automatically moves it to GPU while offloading the other module group to CPU. Transfers use pinned memory for speed.

### Usage

**Python API:**
```python
from vllm_omni import Omni

m = Omni(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers", enable_cpu_offload=True)
```

**CLI:**
```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --enable-cpu-offload
```

### To Support a Model

Implement the `SupportsComponentDiscovery` protocol to declare which
submodules serve as pipeline components (used by offloading, HSDP
sharding, and other framework features):

```python
from typing import ClassVar
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery

class MyPipeline(nn.Module, SupportsComponentDiscovery):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "vision_model"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []  # optional

    def __init__(self):
        super().__init__()
        self.transformer = ...     # DiT — stays on GPU during denoising
        self.text_encoder = ...    # Encoder — offloaded to CPU during denoising
        self.vision_model = ...    # Encoder — offloaded to CPU during denoising
        self.vae = ...             # VAE — always on GPU
```

- `_dit_modules`: attribute names of denoising submodules (kept on GPU
  during the diffusion loop).
- `_encoder_modules`: attribute names of encoder/vision submodules
  (offloaded to CPU during the diffusion loop).
- `_vae_modules`: attribute names of VAE(s) (always kept on GPU, not
  part of the mutual exclusion hooks).
- `_resident_modules`: attribute names of small submodules that must
  stay on GPU during layerwise offloading (e.g. embedders, connectors).
  Optional — defaults to `[]`.

All attribute names support dotted paths for nested submodules
(e.g. `"pipe.transformer"`, `"bagel.time_embedder"`).

Both DiT and encoder lists are needed because the offload hooks use
mutual exclusion: when one group runs, the other moves to CPU.

### Limitations
- Cold start latency increases
- Adds overhead from CPU-GPU transfers between encoder and denoising phases
- Support single GPU only for now


### Component offloading for split models (e.g. Cosmos3)

Some models split their transformer into mutually-exclusive *components* that run
in different phases of a single forward pass rather than as separate pipeline
components -- e.g. Cosmos3's understanding (reasoner) component runs once per
generation while the generation (generator) component runs every denoising step.
Such models have no separate text encoder to swap against, so the transformer
owns a small model-local offload path and wraps each phase with
`with self._offload_context(name):`

```python
class Cosmos3VFMTransformer(nn.Module):
    def forward(self, ...):
        with self._offload_context("reasoner"):
            ...  # understanding pass, runs once
        with self._offload_context("generator"):
            ...  # denoising pass, runs every step
```

Model-level offloading then keeps exactly one component GPU-resident at a time
(the other on CPU), reusing the same `SequentialOffloadHook` `.to()` movers. The
pipeline opts in by exposing `enable_omni_model_cpu_offload` (which drives the
transformer's `enable_model_cpu_offload` and pins the VAE). Layerwise offloading
works for these models too -- each component declares its own block container via
`_layerwise_offload_blocks_attrs`.


## Layerwise (Blockwise) Offloading

### How It Works

Layerwise offloading keeps only one transformer block on GPU at a time.

As each block completes, the next block is prefetched to GPU while the current block is freed. The pre and forward hooks utilized by layerwise offloading apply a separate CUDA stream (`copy_stream`) to overlap weight transfer with computation, and retain flattened tensors in pinned CPU memory for block parameters re-materialization. Encoders, VAE, and non-block DiT modules (embeddings, norms) always stay on GPU.

**Execution Flow:**

| Block | Pre-forward Hook | Forward | Post-forward Hook |
|-------|------------------|---------|-------------------|
| block-0 | Prefetch block-1 (async) | Compute block-0 | Free block-0 |
| block-1 | Prefetch block-2 (async) | Compute block-1 | Free block-1 |
| ... | ... | ... | ... |
| block-(n-1) | **Prefetch block-0** (async) | Compute block-(n-1) | Free block-(n-1) |

Each transformer block has a `LayerwiseOffloadHook` that prefetches the next block before forward and frees the current block after forward.

Layerwise offloading is primarily recommended for large **video generation models** where the compute cost per block is high enough to effectively overlap with memory prefetch operations. For example, Wan2.2 T2V and I2V pipelines.

### Usage

**Python API:**
```python
from vllm_omni import Omni

# Text-to-video
m = Omni(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers", enable_layerwise_offload=True)

# Or image-to-video
m = Omni(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers", enable_layerwise_offload=True)
```

**CLI:**
```bash
# Text-to-video
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --enable-layerwise-offload

# Or image-to-video
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni --enable-layerwise-offload
```

### To Support a Model

Models must define the blocks attribute name for layerwise offloading:

```python
class WanTransformer3DModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]  # Attribute names containing transformer blocks

    def __init__(self):
        self.blocks = nn.ModuleList([...])  # Transformer blocks
```

For models with multiple block types:

```python
class Flux2Transformer2DModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]
```

### Limitations
- Cold start latency increases because offloaded components must be moved to CPU
  during setup; layerwise offload may add extra weight consolidation and pinning
  work.
- Performance depends on compute cost and H2D bandwidth as well
- Support single GPU only for now


## Distributed Layerwise Offloading

Distributed layerwise offloading extends block streaming to multi-device
deployments. It has two distinct execution modes:

| Mode | Host weights per rank | Device communication | Typical parallelism |
|---|---|---|---|
| **AllGather** (default) | One DP shard (`1 / dp_size`) | H2D plus per-block AllGather | DP multi-concurrency |
| **Rank-local** (`--dlo-no-use-allgather`) | The standard loader's local tensors, including TP/HSDP shards | H2D only | TP, SP, or HSDP |

In both modes, a fixed pair of device buffers alternates between the current
and next block. H2D transfer—and AllGather when enabled—runs on dedicated
streams and overlaps block computation.

### Execution flow

```text
Compute stream:  [block N]          [block N+1]          [block N+2]
H2D stream:      [H2D N+1]         [H2D N+2]
AllGather:       [AG N+1]          [AG N+2]              # AllGather mode only
Buffer slots:    slot 0: block N   slot 1: block N+1
```

Additional behavior:

- AllGather mode stores only one equal-sized DP shard of each block in pinned
  host memory and reconstructs the full block before each forward.
  This is a per-rank reduction: shards on ranks within the same host still sum
  to one full DiT checkpoint. DP concurrency reduces memory per worker/device,
  not the node's aggregate host-weight floor.
- Rank-local mode sets DLO's internal DP size to one. It does not add another
  shard; each worker streams exactly what the standard model loader produced.
- Plan-declared encoders can use ordinary rank-local layerwise hooks, while
  VAEs and other stage components can be loaded only around encode/decode.
- `--dlo-resident-layers N` keeps the first `N` plan-selected DiT blocks on
  the accelerator during denoising, then releases them before VAE decode.
  It is available only in rank-local mode.

### Usage

AllGather mode with four concurrent DP ranks:

```bash
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4
```

Rank-local mode with TP2, using MiniMax H3 as an example:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --num-gpus 2 \
  --tensor-parallel-size 2 --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 20 --enforce-eager
```

See the [MiniMax H3 RTX 5090 recipe](../../../recipes/MiniMaxAI/MiniMax-H3-5090.md)
for validated one- and two-GPU consumer configurations.

Python API:

```python
from vllm_omni import Omni

model = Omni(
    model="/path/to/model",
    enable_distributed_layerwise_offload=True,
    dlo_use_allgather=False,
)
```

### CLI flags

| Flag | Description | Default |
|---|---|---|
| `--enable-distributed-layerwise-offload` | Enable DLO | `false` |
| `--data-parallel-size N` | DP ranks used for host-weight sharding in AllGather mode | `1` |
| `--dlo-use-allgather` | Reconstruct DP-sharded blocks with AllGather | `true` |
| `--dlo-no-use-allgather` | Stream standard-loader rank-local tensors independently | `false` |
| `--dlo-resident-layers N` | Stage the first `N` plan-selected DiT blocks for the denoise phase; rank-local only | `0` |

### Weight-loading paths

AllGather mode uses a direct loader for models that expose checkpoint-key
remapping:

1. Construct an opted-in DiT on the meta device, or convert it to meta before
   checkpoint binding, so random initialization weights do not remain resident.
2. Attach checkpoint tensors as `safe_open().get_tensor()` mmap views.
3. Load checkpoint-backed persistent buffers and restore computed,
   non-persistent buffers.
4. Apply model-declared mmap layout transforms before sharding.
5. Copy only the current DP shard into pinned host storage and call post-load
   validation.

This direct path bypasses the model's ordinary weight-loader callbacks. A model
with a checkpoint layout adapter can attach an `mmap_weight_transform` to the
affected parameter; otherwise it must opt out and use the regular loader.
MiniMax H3 uses this hook for its grouped-QKV reorder. TP-specific sharding
callbacks remain unsupported, so direct mmap still requires TP1.

Rank-local mode deliberately keeps the regular loading pipeline:

1. Safetensors are iterated with mmap-backed `safe_open` views. If `eager`
   safetensors loading was requested, DLO changes it to `lazy`; multithreaded
   whole-shard loading is disabled.
2. Registered checkpoint adapters run on each source tensor.
3. The model's `load_weights()` method performs layout conversion and invokes
   standard TP-aware callbacks. MiniMax H3 uses this step for grouped-QKV
   reorder and fused-MLP gate/up packing before TP sharding.
4. DLO flattens and pins only the resulting rank-local tensors.
5. H3's text encoder also reads one mmap tensor at a time, and its VAEs use
   Transformers low-memory loading.

Mmap avoids an eager private copy of the whole checkpoint. The pinned tensors
retained for block streaming still consume process RSS.

### OffloadPlan (declarative topology metadata)

Models can declare block topology and stage lifecycles with `OffloadPlan`:

```python
from dataclasses import replace

from torch import nn

from vllm_omni.diffusion.offloader import OffloadPlan

class MyPipeline(nn.Module):
    _dit_modules = ["transformer"]
    _offload_plan = OffloadPlan(
        block_attrs={"transformer": ("blocks",)},
        offload_submodules={"context_encoder": "layers"},
        resident_dit_paths=frozenset({"transformer"}),
        encoder_block_attrs={"text_encoder": ("layers",)},
        on_demand_component_paths=frozenset({"text_encoder", "vae"}),
    )

    def get_offload_plan(self):
        # A modular pipeline can derive residency from the partition it loaded.
        active = frozenset(
            path for path in self._dit_modules if hasattr(self, path)
        )
        return replace(self._offload_plan, resident_dit_paths=active)
```

`block_attrs` and `offload_submodules` replace heuristic block discovery.
`encoder_block_attrs` declares rank-local encoder stacks.
`on_demand_component_paths` declares components whose pipeline lifecycle
stages them only when needed. `resident_dit_paths` limits the resident-layer
knob to selected DiTs; a combined pipeline can keep separate resident groups
and load only the group selected for the request.

When no plan is declared, DLO falls back to
`_layerwise_offload_blocks_attrs` and heuristic attribute discovery.

### DP multi-concurrency

When `--data-parallel-size > 1` and AllGather is enabled, the scheduler can
process a different request on each DP rank while synchronizing only
request-independent weight shards.

All participating requests must set the same explicit `num_inference_steps`.
Rank-local mode has no collective and therefore no concurrent-request
lockstep requirement.

### Limitations

- AllGather mode rejects online quantization because flattened dtype groups
  cannot preserve quantized weight/scale layouts.
- The direct AllGather mmap path rejects tensor parallelism greater than one
  because it bypasses TP-aware weight-loader callbacks. Rank-local mode
  supports existing TP shards.
- HSDP plus AllGather is rejected to prevent double sharding. Rank-local mode
  can stream HSDP-local tensors.
- `dlo_resident_layers` requires rank-local mode and a model-declared
  `resident_dit_paths` selection.
- DP multi-concurrency requires an explicit, identical inference-step count
  on every participating request.

**Module Discovery**

The offloader discovers pipeline components in two ways:

1. **Protocol-based** (preferred): If the pipeline implements
    `SupportsComponentDiscovery`, its `_dit_modules`, `_encoder_modules`,
    `_vae_modules`, and `_resident_modules` class variables are used
    directly.  All attribute names support dotted paths (e.g.
    `"pipe.transformer"`, `"bagel.time_embedder"`) for nested submodules.

2. **Fallback attribute scan**: Otherwise, the offloader scans for
    well-known attribute names:
    - **DiT modules**: `transformer`, `transformer_2`, `dit`, `sr_dit`, `language_model`, `transformer_blocks`, `model`
    - **Encoders**: `text_encoder`, `text_encoder_2`, `text_encoder_3`, `image_encoder`
    - **VAE**: `vae`, `audio_vae`

**Hook System**

All offloading backends use vLLM-Omni's hook registry system (`HookRegistry` and `ModelHook`) to register pre/post forward callbacks on modules, enabling automatic swapping without modifying model code.

**Backend Architecture**

```
OffloadBackend (base class)
├── ModelLevelOffloadBackend → uses SequentialOffloadHook (.to() swap)
│                              (delegates to a pipeline's enable_omni_model_cpu_offload
│                               for split models like Cosmos3)
├── LayerWiseOffloadBackend → uses LayerwiseOffloadHook
│                          (single-GPU, full weights on host)
└── DistributedLayerwiseOffloadBackend → uses DistributedLayerwiseOffloadHook
                                         (DP-sharded AllGather or rank-local weights)
```

Factory function `get_offload_backend()` selects the appropriate backend based on
configuration.

For split models, `ModelLevelOffloadBackend.enable()` detects a pipeline's
`enable_omni_model_cpu_offload` hook and delegates to it; Cosmos3 then swaps its
reasoner/generator components inside the model forward pass.


## Supported Models

| Architecture | Example Models | DiT Class | Model-Level Offload | Layerwise Offload | Distributed Layerwise Offload | Blocks Attrs (Layerwise specific) |
|--------------|----------------|-----------|---------------------|-------------------|-------------------------------|-----------------------------------|
| Flux2Pipeline | `black-forest-labs/FLUX.2-dev` | `Flux2Transformer2DModel` | ✓ | ✓ | - | `"transformer_blocks"`, `"single_transformer_blocks"` |
| LongCatImagePipeline | `meituan-longcat/LongCat-Image` | `LongCatImageTransformer2DModel` | - | ✓ | - | `"transformer_blocks"`, `"single_transformer_blocks"` |
| NextStep11Pipeline | `stepfun-ai/NextStep-1.1` | `NextStepModel` | - | ✓ | - | `"layers"` |
| OvisImagePipeline | `AIDC-AI/Ovis-Image-7B` | `OvisImageTransformer2DModel` | - | ✓ | - | `"transformer"` |
| QwenImagePipeline | `Qwen/Qwen-Image` | `QwenImageTransformer2DModel` | ✓ | ✓ | - | `"transformer_blocks"` |
| StableDiffusionXLPipeline | `stabilityai/stable-diffusion-xl-base-1.0` | `SDXLUNet2DConditionModel` | ✓ | ✓ | - | `"down_blocks"`, `"up_blocks"` |
| StableDiffusion3Pipeline | `stabilityai/stable-diffusion-3.5-medium` | `SD3Transformer2DModel` | - | ✓ | - | `"transformer_blocks"` |
| Wan22I2VPipeline | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | `WanTransformer3DModel` | ✓ | ✓ | - | `"blocks"` |
| Wan22Pipeline | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `WanTransformer3DModel` | ✓ | ✓ | - | `"blocks"` |
| SoulXSingerPipeline / SoulXSingerSVCPipeline | `Soul-AILab/SoulX-Singer` | `DiffLlama` (`cfm_decoder.model.diff_estimator`) | ✓ | ✓ | - | `"layers"` |
| BagelPipeline | `ByteDance-Seed/BAGEL-7B-MoT` | `Qwen2MoTModel` | - | ✓ | - | `"layers"`, `"customized modules"` |
| MiniMaxH3Pipeline | `MiniMaxAI/MiniMax-H3` | `MiniMaxH3DiTModel` | ✓ | ✓ | ✓ (rank-local) | `"blocks"` plus plan-declared encoder stacks |
| Cosmos3OmniDiffusersPipeline | `nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Super` | `Cosmos3VFMTransformer`, `Cosmos3LanguageModel` | ✓ | ✓ | ✓ | `"layers"`, `"gen_layers"` |

**Notes:**

- Model-level offloading applies to pipelines that declare their DiT and
  encoder components.
- Layerwise offloading requires discoverable transformer block lists.
- DLO can use the generic topology fallback, but direct AllGather mmap loading
  additionally requires checkpoint-key remapping. Advanced encoder, VAE,
  resident-layer, and multi-DiT staging should be declared with `OffloadPlan`.
- See the [Cosmos3 DLO recipe](../../../recipes/cosmos3/Cosmos3-DistOffload.md)
  for AllGather examples and the
  [MiniMax H3 RTX 5090 recipe](../../../recipes/MiniMaxAI/MiniMax-H3-5090.md)
  for rank-local TP examples.
