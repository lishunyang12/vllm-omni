# MiniMax-H3 on four RTX PRO 5000 Blackwell GPUs

This recipe runs MiniMax-H3 in BF16 on four 72 GiB RTX PRO 5000
Blackwell GPUs. It keeps the model resident, uses TP2 to shard the weights,
Ulysses2 to shard the attention sequence, and tiled VAE decode across all four
GPUs. CPU offload and distributed layerwise offload are not required.

## Capacity requirements

| Resource | Requirement |
| --- | ---: |
| GPUs | 4 x RTX PRO 5000 Blackwell |
| GPU HBM | 72 GiB per GPU |
| Checkpoint storage | 135 GiB per partition |
| Available system RAM | 200 GiB minimum |
| Recommended system RAM | 384 GiB |

`FL2VA` and `Ref2VA` are separate checkpoint partitions. Start one server at a
time on a host sized for the minimum system-memory requirement.

## PCIe topology and GPU order

RTX PRO 5000 does not provide NVLink. Before starting the server, select four
GPUs on the same NUMA node and identify the two closest PCIe pairs:

```bash
nvidia-smi topo -m
nvidia-smi nvlink -s
```

On the validated host, physical GPUs `(0,1)` and `(2,3)` are the two `PXB`
pairs on NUMA node 0. The order `CUDA_VISIBLE_DEVICES=0,2,1,3` maps the
Ulysses groups to those local pairs. Do not copy these IDs blindly: reproduce
the same relationship on the target host. For the second NUMA node of the
validated host, the equivalent order is `4,6,5,7`.

## Four-GPU serving configuration

The validated baseline uses TP2 x Ulysses2, text-encoder TP4, VAE patch
parallelism 4, and explicit cuDNN BF16 attention. Selecting the backend
explicitly keeps the recipe independent of platform-default backend changes.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0,2,1,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
numactl --cpunodebind=0 --membind=0 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --tensor-parallel-size 2 \
  --usp 2 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN
```

Do not add `--enforce-eager` for the performance run. Warm the server once
before measuring so regional compilation is outside the measured request.

For Ref2VA, stop the FL2VA server and restart the same command with
`MODEL="${MODEL_ROOT}/Ref2VA"`.

## Target-hardware validation

The configuration was exercised on a PCIe-only, dual-socket host with four
selected RTX PRO 5000 GPUs on one NUMA node. The run used PyTorch 2.11.0+cu130,
CUDA 13.0, driver 580.95.05, 1344x768 output, 124 frames, and two warmups.

| Measurement | Result |
| --- | ---: |
| Maximum externally sampled peak | 69,219 MiB (67.60 GiB) per GPU |
| T2VA worker-reported peak | 65,314 MiB |
| First-frame FL2VA worker-reported peak | 66,780 MiB |
| T2VA maximum GPU kernel-time deviation | 0.56% |
| First-frame FL2VA maximum GPU kernel-time deviation | 0.09% |

The measured peak leaves about 4.1 GiB below the reported 73,415 MiB device
capacity. Re-measure memory for longer reference inputs, concurrency greater
than one, or a different output shape. The recorded run is a five-step
profiling validation; it is not a production 50-step latency claim.

## T2VA request example

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"

curl -sS --max-time 1800 -X POST "${API_URL}" \
  -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```
