# SubBlock Sage Attention

`SUBBLOCK_SAGE_ATTN` is an **opt-in, experimental, lossy** sparse self-attention backend for MiniMax-H3 on
**SM120** GPUs (RTX PRO 6000 Blackwell). It trades output fidelity for speed, and nothing changes unless you select it.
It requires FlashInfer with the SM120 CAKE Sage kernels (`flashinfer.cute_dsl.sparse.bsa_attn_sm120`).

A training-free router pools query and key tokens into sub-blocks, scores every 64-token block pair with a
log-sum-exp, and keeps the top `1 - sparsity` key blocks for each 64-token query block. FlashInfer's CAKE Sage
kernel (INT8 Q/K, FP8 V) then attends to the selected blocks only. Query blocks that contain non-video rows (text,
visual conditions, audio) stay dense.

The backend delegates to the platform's exact BF16 attention (cuDNN) for everything it does not handle: cross
attention, the first `start_step` denoise steps, layers in `skip_layers`, sequences shorter than `min_seq_len`,
non-BF16 or non-128 head sizes, and calls without a packed video layout. Other GPUs fail at startup with a clear
message instead of falling back silently.

## Configuration

| Key | Valid values | Meaning |
| --- | --- | --- |
| `sparsity` | `(0, 1)` | Upper bound on the dropped key-block fraction per query block, rounded up to the kernel's budget granularity of 8 blocks; default `0.75` |
| `start_step` | integer, `>= 0` | Number of early denoise steps kept dense; default `10` |
| `end_step` | integer, `>= 0` | Number of final denoise steps kept dense; default `0` |
| `min_seq_len` | integer, `>= 64` | Shorter sequences stay dense; default `4096` |
| `skip_layers` | selector such as `"15"` or `"0-3,38"` | DiT blocks kept dense |
| `n_q`, `n_k` | `1`, `2`, `4`, `8` | Query/key sub-blocks per 64-token block in the router score; default `4` |

Select it for the self-attention role only, so cross attention keeps the platform default:

```bash
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --diffusion-attention-config '{"per_role":{"self":{"backend":"SUBBLOCK_SAGE_ATTN",\
    "subblock":{"sparsity":0.75,"start_step":10}}}}'
```

A more conservative setting that keeps later steps dense and exempts a sensitive layer:

```bash
  --diffusion-attention-config '{"per_role":{"self":{"backend":"SUBBLOCK_SAGE_ATTN",\
    "subblock":{"sparsity":0.5,"start_step":20,"skip_layers":"15"}}}}'
```

## Quality warning

The output differs from BF16 attention. Sparsity is the dominant source of error; Sage quantization alone is
1.4–2.9 % relative L2 per attention call, and adding it on top of sparsity changes the error very little.
On one RTX PRO 6000 (49 steps, three MiniMax-H3 clips, BF16 cuDNN as the reference):

| Setting | s / clip | Speedup | Video PSNR (dB) | Video SSIM | Audio cosine |
| --- | --- | --- | --- | --- | --- |
| BF16 cuDNN (reference) | 495.5 | 1.00x | exact | 1.00 | 1.00 |
| `sparsity 0.75, start_step 10` | 327.7 | 1.51x | 22.2 / 18.6 / 22.5 | 0.78 / 0.78 / 0.88 | 0.62 / 0.77 / 0.95 |
| `sparsity 0.5, start_step 20, skip_layers 15` | 393.7 | 1.26x | 27.8 / 26.9 / 24.2 | 0.88 / 0.89 / 0.92 | 0.77 / 0.84 / 0.98 |

Dense early steps fix the overall composition before sparsity applies, so lowering `start_step` is not
recommended. Review the generated video and audio before using any setting.

## Limitations

- SM120 only; SM90 and SM100 are rejected at startup.
- BF16 activations, head size 128, non-causal self-attention with a packed video layout.
- Single-GPU and Ulysses sequence parallel run the exchange in BF16; the quantize-before-all-to-all exchange is not
  part of this backend.
