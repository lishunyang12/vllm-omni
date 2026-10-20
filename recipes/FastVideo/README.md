# FastH3

FastH3 has three distinct releases in vLLM-Omni. Choose the checkpoint and
recipe for the task you want to run; each release keeps its trained schedule
and attention policy. Hardware determines the attention provider and deployment settings.

| Release | Task | Weights | Transformer forwards | Video/audio shifts | Attention | Recipe |
| --- | --- | --- | ---: | --- | --- | --- |
| **FastH3** | T2VA: text to video and audio | Four-step Preview V1 adapter fused into native H3 | 4 | Native H3 12/3; request overrides rejected | Dense or VSA variant; VSA uses fixed top-k | [FastH3](FastH3.md) |
| **FastH3 V2** | T2VA: text to video and audio | Full 8-Step V2 checkpoint | 8 | 10/3 | 64-token VSA, 80% sparsity | [FastH3 V2](FastH3-V2.md) |
| **FastH3 Omni** | Ref2VA: references plus text to video and audio | Full Omni 8-Step V1 checkpoint | 8 | 12/3 | Logical 128-token VSA, 90% sparsity, separate reference-video budgets | [Single card](<FastH3 RefOmni Single Card.md>) · [Multi-GPU](FastH3-Omni.md) |

T2VA requires no reference media. Omni Ref2VA requires at least one image or
video and can also use audio references. All three use the existing
`MiniMaxH3Pipeline` and video HTTP API. A deployment loads one release;
request fields cannot switch an adapter deployment into V2 or Omni.

The step count in Omni is the number of transformer forwards. In particular,
V2 uses `num_inference_steps=8` even though its FastVideo example counts nine
sigma nodes. Do not apply the four-step adapter to either full checkpoint.

## Hardware and attention providers

| Release / variant | Execution route | Qualification |
| --- | --- | --- |
| FastH3 Dense | A dense backend supported by H3 and the installed CUDA platform | Historical B300 measurements; choose a backend for the target GPU |
| FastH3 VSA / FastH3 V2 | cuDNN BF16 tile64 in the eight-step profile; automatic FlashInfer/FastVideo selection remains available | SM120 kernel regressions; V2 full-model qualification pending |
| FastH3 Omni | cuDNN native tile128 in the eight-step profile; automatic FastVideo tile128 on SM100/SM103 or FlashInfer block64 on SM120/SM121 remains available | SM120 kernel and cuDNN HTTP generation checks on 8× PRO 5000; cross-provider quality equivalence unverified |

The model names, sampling contracts, and HTTP requests are independent of the
chosen provider. Install a compatible kernel and size the parallel deployment
for the GPU memory and interconnect. Provider availability alone does not
establish full-model quality, throughput, or cross-provider equivalence.
cuDNN also provides tile128 on SM90. That provider capability does not establish
full-model qualification on an untested GPU.

The BF16 Omni recipe also documents an optional
[released NVFP4 VAE decoder](FastH3-Omni.md#optional-nvfp4-video-decoder).
That decoder option does not establish support for the separate OmniRef FP8,
OmniRef NVFP4-QAD, or FastH3-QAD-v2 full checkpoints.

The [single-card guide](<FastH3 RefOmni Single Card.md>) brings together
RTX 4090, RTX 5090, DGX Spark, and RTX PRO 6000 Blackwell configurations.
It covers installation, server commands, generation, decoder selection,
and memory settings in one place.

## Performance

See the [8× RTX PRO 5000 Omni benchmark](FastH3-Omni.md#benchmark-8-nvidia-rtx-pro-5000)
for the workload, parallel configuration, measured latency, memory accounting,
and software-version scope. These Ref2VA measurements do not describe V2 T2VA
performance.

## Shared documentation

- [Native MiniMax H3](../MiniMaxAI/MiniMax-H3.md): base-model inputs and API fields.
- [VSA backend](../../docs/user_guide/diffusion/attention_backends/fastvideo_vsa.md): provider selection.
- [ComfyUI integration](../../docs/features/comfyui.md): generic video workflows.
- [All community recipes](../README.md).
