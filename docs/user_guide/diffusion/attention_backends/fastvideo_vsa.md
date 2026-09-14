# FastVideo VSA

Variable Sparse Attention (VSA) selects key/value blocks from a video token
grid. It changes the attention computation and is not a bitwise-lossless
replacement for dense attention.

## Supported models

| Model / checkpoint | Required adapter | Tasks | Sequence parallelism |
| --- | --- | --- | --- |
| `FastVideo/FastWan2.2-TI2V-5B-Diffusers` | None | T2V, I2V | Disabled |
| `MiniMaxAI/MiniMax-H3` | FastH3 VSA | T2VA | Disabled or pure Ulysses |

The backend requires CUDA, FP16/BF16, and non-causal self-attention with equal
Q/K/V sequence lengths and head counts. Other models require an explicit
integration; installing the kernel alone does not establish support.

## Installation

In the vLLM-Omni environment, install the
[published kernel wheel](https://pypi.org/project/fastvideo-kernel/0.3.4/):

```bash
uv pip install --only-binary=:all: "fastvideo-kernel==0.3.4"
```

This wheel setup requires Linux, Python 3.12, and glibc 2.34 or newer
(x86-64 or aarch64). The full FastVideo framework and additional environment
variables are not required.

## Enable the backend

```bash
vllm serve FastVideo/FastWan2.2-TI2V-5B-Diffusers --omni \
  --diffusion-attention-backend FASTVIDEO_VSA
```

Top-k defaults to 64. Set `--fastvideo-vsa-topk K` to change it. Per-role
configuration follows the shared [attention configuration](../attention_backends.md#configuration);
place `fastvideo_vsa_topk` in the corresponding role's `AttentionSpec`.

For H3, use the [FastH3 VSA recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#fasth3-vsa-serving),
which loads the required adapter and specifies the four-step request.

## Choose top-k

Top-k is a positive block count per query, not a sparsity percentage. Block
counts depend on resolution, duration, and model geometry.

| Route | Selection and boundary behavior |
| --- | --- |
| Wan | 256-token video blocks. `K > num_blocks` falls back to SDPA. At equality, native checkpoints use SDPA; DMD checkpoints retain VSA. |
| H3 | 64-token video blocks. Retains `min(K, video_blocks)` video blocks plus all prefix keys; prefix queries remain dense. Full selection retains VSA and its learned gate. |

Use the logged block count when choosing K. Smaller K can reduce computation
and quality; overhead can limit the speedup. Compare with the checkpoint's
reference implementation at the same shape, seed, and sampling schedule.

## Checkpoint behavior

Learned compression gates are loaded from checkpoint weights, not selected
by a user flag. Native Wan and distilled checkpoints retain their respective
schedules. A sparse-distilled adapter requires its VSA computation; forcing
it through dense attention does not recover the base model.

## Verify routing and fallback

| Log | Interpretation |
| --- | --- |
| `route=VSA` / `route=VSA_ALL_BLOCKS` | Wan sparse / all-block VSA route |
| `FASTVIDEO_VSA H3 routing` | H3 route and prefix/video block counts |
| `route=SDPA` / `FASTVIDEO_VSA falling back to SDPA` | Dense execution; the warning states the reason |

Routing messages precede the kernel call. Confirm request completion and
inspect subsequent fallback warnings. The H3 recipe identifies the expected
token-refiner fallback. CUDA accelerator faults propagate to the worker.
