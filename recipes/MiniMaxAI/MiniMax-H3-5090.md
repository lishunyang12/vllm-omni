# MiniMax-H3 on RTX 5090

[Model guide](MiniMax-H3.md) · [Deployment choices](MiniMax-H3.md#choose-a-deployment) · [HTTP API](MiniMax-H3.md#http-api-examples)

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

> **Modular H3:** after #5720 lands, preserve this recipe's one-partition
> behavior with `--task-type fl2va` or `--task-type ref2va`.

## One RTX 5090: 1344x768, 5 seconds

Use 12 resident DiT layers. A 50-step B300 allocation test with this exact
single-rank topology peaked at 26.50 GiB; re-measure peak HBM on the target
card before increasing the resident count.

```bash
vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code \
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
vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code \
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

## RTX 5090 target-hardware validation

At vLLM-Omni commit `ae6577ea`, one full 50-step T2VA request completed on
2 x RTX 5090 without OOM:

| Shape    | Frames        | Client E2E | Sampled peak/GPU       | Output validation                                            |
| -------: | ------------: | ---------: | ---------------------: | -----------------------------------------------------------: |
| 1344x768 | 124 at 24 FPS | 8 min 38 s | approximately 22.6 GiB | H.264 video + 32 kHz stereo AAC; full `ffmpeg` decode passed |

This is a single end-to-end validation run, not a warmed multi-run latency
benchmark. The sampled `nvidia-smi` peak is also not a CUDA allocator
high-water mark. The environment used vLLM 0.26.0, vLLM-Omni
`0.26.1.dev14+gae6577ea`, and PyTorch 2.11.0+cu130. The
[run record](https://github.com/lishunyang12/vllm-omni-rankings/blob/dcd06d7e83cb069842535918c0169ee9f3f29ba0/scripts/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20260805000034_86_237.png)
captures the environment, output contract, elapsed time, and sampled peak.

Before the target run, both profiles were exercised on two B300 ranks as an
allocation and correctness proxy. At 1344x768, 124 frames, and 50 steps, the
20-layer profile peaked at 27,726 MiB per rank. At 1024x576, the 12-layer
profile peaked at 18,888 MiB per rank in a 5-step capacity run. The resident
and fully streamed placements produced identical decoded video-frame and audio
hashes for the same shape, step count, prompt, and seed. The B300 result does
not establish RTX 4090 PCIe latency; treat the 4090 profile as a conservative
starting point until it is measured on that GPU.

To run T2VA, FL2VA, image+audio Ref2VA, and two-video Ref2VA in order, validate
every MP4's H.264/AAC streams, and retain live server and GPU-memory logs:

```bash
RUN_ROOT=/path/to/run-root \
MODEL_ROOT=/path/to/MiniMax-H3 \
GPU_IDS=0,1 \
PROFILE=rtx5090 \
bash examples/offline_inference/minimax_h3/run_h3_2gpu_all_tasks.sh
```

The script selects 20 resident layers for `PROFILE=rtx5090` and 12 for
`PROFILE=rtx4090`; `DLO_RESIDENT_LAYERS=N` overrides either default.
