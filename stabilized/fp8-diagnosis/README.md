# Why FP8 is slower for these Rubin Qwen/Wan requests

The original latency arithmetic is correct, and a fresh full-request check reproduces the slowdown. The FP8 GEMMs are faster, but the surrounding dynamic quantization, separate bias operations and host dispatch outweigh the savings. The balance differs between Qwen and Wan; no runtime performance fix is claimed by this investigation.

**Placement correction:** Qwen-Image uses four Rubin GPUs. Wan2.2 uses **one**, because the pinned default `vllm_omni/deploy/wan2_2_ti2v.yaml` fixes `devices: "0"`. Requested four-GPU kwargs were ineffective. All original Wan modes use this same placement. [Telemetry audit of the original 40 cases](../execution-topology.json) and the diagnostic worker traces corroborate it. No four-GPU Wan score was measured.

## Full-request latency

Seconds; five warmups per case. The n=20 P50/P95 remain the performance results. The separate n=3 check is replication evidence, not a new stabilized score. One fixed prompt/seed per model, 50 steps, eager, concurrency one.

| Model | GPUs | Mode | Original n=20 P50 / P95 s | Fresh n=3 P50 s |
|---|---:|---|---:|---:|
| qwen_image | 4 | BF16 | 7.569 / 7.705 | 7.519 |
| qwen_image | 4 | MXFP8 | 14.433 / 15.404 | 14.202 |
| qwen_image | 4 | Blockwise FP8 | 13.465 / 13.650 | 13.505 |
| wan22 | 1 | BF16 | 23.891 / 23.906 | 23.861 |
| wan22 | 1 | MXFP8 | 26.875 / 26.903 | 26.873 |
| wan22 | 1 | Blockwise FP8 | 28.309 / 28.321 | 28.306 |

[Raw replication records](replication.csv) · [Validation](validation.json) · [Original log/data recheck](original-data-recheck.json).

All six processes exited successfully. All first/last measured pixel hashes match their original precision-mode outputs. Recorded source hashes, model revisions and runtime package versions match the original run. There were no autotune/compile messages after the original final warmup in these six logs. Qwen used the same cuDNN attention backend for all precisions; MXFP8 selected `FlashInferCutedslMxfp8LinearKernel`, and blockwise selected `DeepGemmFp8BlockScaledMMKernel`. This does not indicate repeated weight quantization: weights are quantized at load, activations during linear execution.

## Operator evidence

One two-step diagnostic trace per rank after two short warmups. Values below are **GPU kernel-duration sums in ms**, averaged across Qwen’s four ranks; Wan has only rank 0. They are not request latency and must not be added to CPU wall time. All ranks and raw traces are published; the table does not select a favorable rank.

| Model | Mode | Matched DiT GEMM work ms | Quantization / scales ms | Separate bias-add ms | Bias calls per rank |
|---|---|---:|---:|---:|---:|
| qwen_image | BF16 | 35.372 | 0.000 | 0.000 | 0 |
| qwen_image | MXFP8 | 21.412 | 5.280 | 17.942 | 1920 |
| qwen_image | Blockwise FP8 | 24.615 | 14.366 | 18.030 | 1920 |
| wan22 | BF16 | 284.773 | 0.000 | 0.000 | 0 |
| wan22 | MXFP8 | 106.038 | 32.099 | 321.964 | 960 |
| wan22 | Blockwise FP8 | 138.251 | 58.724 | 321.719 | 960 |

GEMM scope: Qwen covers all 1,920 quantized-scope DiT linear calls per rank; BF16 includes associated split-K reduction kernels. Wan covers the 720 large image-branch calls (M=27,280), excluding text KV and conditioning projections that share BF16 shapes. Quantization and bias columns cover all quantized linears, including Wan’s small text branch. The 240 Wan MXFP8 text GEMMs add only 1.615 ms, so this scope distinction cannot explain the bias overhead. DeepGEMM scale packing is counted as quantization, not GEMM.

**Qwen:** FP8 lowers GEMM GPU work, but separately launches dynamic quantization and 1,920 bias additions per rank. BF16 uses `aten::addmm` with bias fused. Small text-branch M values are 5 and 36; image M is 1,024 per rank. The blockwise trace also spends more CPU operator time in compiled-region dispatch, Dynamo cache lookup and the DeepGEMM binding. The isolated control below supports a substantial host-dispatch contribution for MXFP8. NCCL duration includes waiting for other ranks and does not by itself establish a network bottleneck.

**Wan:** the large image-branch GEMMs improve from 284.773 ms to 106.038 ms (MXFP8) or 138.251 ms (blockwise), but about 322 ms of separate bias-add work appears in either FP8 mode, plus dynamic quantization/scaling. That additional memory pass is a concrete reason faster GEMMs do not produce a faster full request here.

Profiling perturbs timing: blockwise has a longer profiled Qwen pipeline than MXFP8, while the unprofiled full-request order is reversed. Use the n=20 timings for performance ranking. These traces identify overhead; they do not assign an exact fraction of the full-request slowdown to every cause.

## Isolated MXFP8 host-dispatch control

Synthetic BF16 weights and Qwen-observed shapes, on the same Rubin allocation after all model runs. Each value is median mean-call time across seven rounds of 100 synchronized calls, in microseconds. The MXFP8 wrapper includes activation quantization, GEMM and bias. CUDA Graph replay reduces Python dispatch; it is not an end-to-end model graph or a validated production optimization.

| M / N / K | BF16 eager µs | MXFP8 eager µs | BF16 graph µs | MXFP8 graph µs |
|---|---:|---:|---:|---:|
| 5 / 12288 / 3072 | 7.32 | 85.87 | 8.51 | 10.81 |
| 36 / 12288 / 3072 | 7.02 | 81.94 | 8.10 | 11.69 |
| 1024 / 12288 / 3072 | 29.70 | 81.45 | 32.65 | 43.98 |
| 1024 / 3072 / 12288 | 31.82 | 73.87 | 33.10 | 32.44 |

The M=5 case drops from 85.87 µs eager to 10.81 µs graph replay for MXFP8. Python call profiles show input validation, runner construction, autotuner cache lookup, quantization wrappers and output allocation on the per-call path. Cache lookup is not fresh autotuning. cProfile itself adds overhead, so its function timings are attribution evidence rather than unprofiled latency estimates. [All rounds and Python profiles](microbench).

## Next changes justified by this evidence

1. Fuse bias into the FP8 GEMM epilogue, or otherwise remove the extra full output-tensor pass; prioritize Wan.
2. Reduce repeated Python validation/runner/cache work for fixed shapes; prioritize Qwen’s small text linears.
3. Evaluate retaining BF16 for small linears and validate whole-model graph execution, with the same quality and latency protocol.

These are optimization candidates, not implemented or measured fixes. Rubin operator attribution is not a GB200 profile, and it does not explain an SGLang comparison. Existing FP8 quality differences remain; performance changes must retain the SSIM/PSNR/LPIPS and generated-output checks.

## Reproduce and serve

[Diagnostic instructions](reproduce/README.txt) · [Diagnostic scripts ZIP](reproduce.zip) · [Analysis code](reproduce/analyze_traces.py) · [Raw profiles/metadata](results) · [Logs](logs).

[Online serve script](../reproduce/serve_online.sh) · [Pinned image bootstrap and online request examples](../reproduce/ONLINE-SERVE.txt). The online wrapper was syntax/CLI-validated; this investigation did not add an HTTP latency benchmark.

Measured source: `a6a35c86b13db636a07173755b241544dff4c082`. [PR #10390](https://github.com/vllm-project/vllm-omni/pull/10390) merged as `b576ed001032a78d565960d4b99740251a9daa8b`. Pinned container: `vllm/vllm-openai:cu134-nightly-1648fefab91204f3b15bf8062217316f07f2c4bb`. Rubin SM107, driver 620.70.06; [full versions](evidence/versions.json). Slurm job 3242146 completed with exit code 0 and released its allocation.
