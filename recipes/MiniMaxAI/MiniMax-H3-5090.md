# MiniMax-H3 on RTX 5090

This recipe uses BF16 weights, tiled VAE decode, tensor parallelism where a
second GPU is available, and distributed layerwise offload (DLO). It is a
memory-first serving configuration; lower resident counts reduce HBM use and
increase CPU-to-GPU transfer time.

## Capacity requirements

| Resource | One RTX 5090 | Two RTX 5090s |
| --- | ---: | ---: |
| GPU HBM | 32 GiB | 32 GiB per GPU |
| Checkpoint storage | 135 GiB per partition | 135 GiB per partition |
| Available system RAM | 200 GiB minimum | 200 GiB minimum |
| Recommended system RAM | 384 GiB | 384 GiB |

`FL2VA` and `Ref2VA` are separate 135 GiB checkpoint partitions. Start one
server at a time. DLO keeps rank-local weights in pinned host memory; increasing
`--dlo-resident-layers` improves latency but does **not** reduce host RAM in the
current implementation because resident layers retain pinned CPU master copies.

## One RTX 5090: 1344x768, 5 seconds

Use 12 resident DiT layers. A 50-step B300 allocation test with this exact
single-rank topology peaked at 26.50 GiB; re-measure peak HBM on the target
card before increasing the resident count.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
  --usp 1 --ring 1 --vae-patch-parallel-size 1 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 12 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

## Two RTX 5090s: 1344x768, 5 seconds

Use TP2 and 20 resident DiT layers. The two-rank B300 capacity run peaked at
27,726 MiB per rank for this shape and 50 steps. This is a memory/correctness
proxy, not a consumer-GPU latency claim.

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
  --usp 1 --ring 1 --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 20 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

For Ref2VA, stop the FL2VA server and restart the same command with
`/path/to/MiniMax-H3/Ref2VA`. Ref2VA reference video count and prompt length
can increase activation memory; begin with one request at a time.

## Reproducible two-GPU E2E

The repository runner exercises T2VA, FL2VA, and both Ref2VA modes, validates
the generated MP4 streams, and records per-second GPU usage:

```bash
RUN_ROOT=/path/to/run-root MODEL_ROOT=/path/to/MiniMax-H3 \
GPU_IDS=0,1 PROFILE=rtx5090 \
bash examples/offline_inference/minimax_h3/run_h3_2gpu_all_tasks.sh
```
