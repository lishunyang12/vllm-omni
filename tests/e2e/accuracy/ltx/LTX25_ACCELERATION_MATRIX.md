# LTX-2.5 offline acceleration matrix

`run_ltx_acceleration_matrix.py` runs every experiment through the existing
`run_ltx_reference.py` inference runner and measures it with the existing
`compare_ltx_outputs.py` utility. It does not duplicate pipeline or metric
logic.

Use the official Diffusers one-stage output as `--reference-dir`. The driver
snapshots the request, fixes prompt/seed/shape/steps across every case, runs one
unreported warmup plus three measured repetitions, and writes both JSON and CSV
summaries. Every measured output is compared with Diffusers and with the eager
Omni output. Quality gates are applied against each case's declared comparison
baseline (normally eager; Ulysses2/4 for the VAE-patch cases), so an existing
Diffusers/Omni or SP parity gap is not misattributed to another acceleration.
Self-baseline comparisons emit schema-compatible identity metrics directly;
they do not reread all frames or run LPIPS. Comparisons between different
artifact directories still run the full metric utility.
Every Omni run pins `CUDNN_ATTN`, including eager, to hold attention numerics
constant across the matrix.

## Verified vLLM 0.27 smoke status

The August 2026 smokes used one measured repeat per case. They establish
feasibility and catch regressions, but are not stable-throughput benchmarks.
Quality is evaluated against the matched eager CUDNN artifact with the
checked-in gate. "Functional" means that the configured path completed a real
121-frame request and wrote video and audio artifacts; it does not mean that
the path passed quality gates, accelerated generation, or is recommended for
deployment.

| Case | Runtime status | Functional | Generation | Speedup vs matched eager | Peak GPU memory | Quality vs eager | Recommended use |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| eager | completed | yes | 4.25 s | 1.00x | 78,016 MB | baseline | production reference |
| regional_compile | completed | yes | 19.80 s | 0.21x | 78,018 MB | fail | no |
| tp2 | completed | yes | 4.26 s | 1.00x | 59,710 MB | fail | no |
| ulysses2 | completed | yes | 7.97 s | 0.53x | 78,016 MB | fail | no |
| ring2 | failed | no | N/A | N/A | N/A | not evaluated | no on this SM120 stack |
| hsdp2 | completed | yes | 5.75 s | 0.79x | 60,942 MB | strict parity | memory-constrained two-GPU runs only |
| fp8 | completed | yes | 4.52 s | 0.97x | 59,790 MB | fail | no; memory experiment only |
| layerwise_offload | completed | yes | 11.30 s | 0.39x | 43,466 MB | eager-reference parity | single-GPU capacity fallback |
| distributed layerwise offload | not run | unknown | N/A | N/A | N/A | not evaluated | pending DP2 AllGather smoke |
| cache_dit | provisional run completed | provisional | 7.66 s | 0.58x | 78,504 MB | identity in provisional run | no; rerun after adapter fix |

The failed gates are material. Regional compile produced SSIM 0.75662, PSNR
18.31 dB, LPIPS 0.14478, and audio cosine 0.47061 against eager, while also
running 4.66x slower. TP2 produced SSIM 0.83606 and Ulysses2 produced SSIM
0.84551; neither accelerated generation. Therefore regional compile, TP2, and
Ulysses2 are experiments only for LTX-2.5 in this branch. Ring2 did not produce
an artifact because the installed FlashAttention Hopper binary has no SM120
kernel image.

HSDP2 is functional and lossless in this smoke: its output was identical to
the matched eager artifact (SSIM 1.0, PSNR 120 dB, LPIPS 0, audio cosine 1.0).
It reduced peak memory from 78,016 MB to 60,942 MB (21.9%) but made generation
1.27x slower and cold end-to-end latency 2.65x slower. It is a capacity option,
not a latency optimization.

Online FP8 quantization is also functional, and the log confirms the
Cutlass FP8 scaled-matmul and Quack fused-bias paths were selected. It reduced
peak memory from 78,016 MB to 59,790 MB (23.4%), but did not improve this cold
generation run. More importantly, it failed the quality gate with SSIM
0.72023, PSNR 17.79 dB, LPIPS 0.17234, and audio cosine 0.39093 against the
matched eager artifact. FP8 must not be recommended for LTX-2.5 until its
layer exclusions or quantization policy are calibrated and retested.

The verified `layerwise_offload` case is ordinary single-process layerwise CPU
offload, not Distributed Layerwise Offload (DLO). It offloaded all 48 video
transformer blocks, reduced peak memory from 78,016 MB to 43,466 MB (44.3%),
and reproduced the eager run's official-reference scores, but generation was
2.57x slower and cold end-to-end latency was 4.33x slower. Use it only when
single-GPU capacity is the limiting factor. A real DLO test still requires
`data_parallel_size=2`, `enable_distributed_layerwise_offload=True`,
`dlo_use_allgather=True`, and `dlo_resident_blocks=0`; no functional or
performance claim is made for that path yet.

The first Cache-DiT run logged successful DBCache attachment and completed an
artifact, but it was identical to eager while generation was 1.73x slower and
peak memory increased slightly. This is evidence that the integration can be
entered, not that useful cache skipping occurred. Cache-DiT numbers in this
document are provisional until the LTX block adapter is fixed and the same
request is rerun. TP4, Ulysses4, Ring4, VAE cases, model offload, true DLO, and
combined stacks remain unverified and must not be claimed as validated.

```bash
cd <repo>
source <repo>/.venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python tests/e2e/accuracy/ltx/run_ltx_acceleration_matrix.py \
  --model <model> \
  --request tests/e2e/accuracy/ltx/ltx25_distilled_request.json.example \
  --reference-dir <output>/diffusers-one-stage-t2v \
  --output-root <output>/acceleration-matrix \
  --gpus 4,6,5,7
```

Inspect commands without loading the model or touching GPUs:

```bash
python tests/e2e/accuracy/ltx/run_ltx_acceleration_matrix.py \
  --model <model> \
  --reference-dir <output>/diffusers-one-stage-t2v \
  --output-root /tmp/ltx25-matrix-dry-run \
  --gpus 4,6,5,7 \
  --dry-run
```

Use `--cases eager,regional_compile,ulysses2` for a subset. Eager is inserted
automatically; any declared matched baseline is also inserted. Failed cases do
not stop later cases. `--resume` reuses successful run directories and continues
an interrupted matrix.

The default matrix covers eager, regional compile, TP2/4, Ulysses2/4, Ring2/4,
HSDP2, VAE tiling, VAE patch2, model and layerwise CPU offload, FP8, Cache-DiT,
and an experimental compile+Ulysses4+VAE-patch4 stack. A case being listed in
the matrix is a reproducible test plan, not a support or quality claim. VAE
patch2 intentionally shares the Ulysses2 process group and should be compared
with `ulysses2`, not with single-GPU eager, when attributing decode speed. Cache-DiT uses a
conservative eight-step profile and remains a lossy experiment.

CFG parallel is skipped because the distilled recipe is positive-only.
TeaCache is unsupported rather than merely unbenchmarked. The generic backend
looks up the exact transformer class in both `EXTRACTOR_REGISTRY` and
`_MODEL_COEFFICIENTS`; `LTX2VideoTransformer3DModel` is present in neither.
Without a model-specific extractor it cannot obtain the modulated input and
residual boundaries used for cache decisions, and without five calibrated
polynomial coefficients it cannot map relative-L1 changes to the residual
change estimate. Passing arbitrary coefficients would address only the second
failure and would not make the hook correct. GGUF is skipped because there is
no official LTX-2.5 GGUF checkpoint or validated component adapter.

`TRTLLM_ATTN` is also unsupported for the current LTX-2.5 transformer. LTX-2.5
uses Q/K/V head dimension 64, while the current vLLM-Omni TensorRT-LLM backend
advertises head size 128 and the installed kernels reject 64 during dummy
warmup. Use `CUDNN_ATTN` for the production path; changing the model shape is
not a valid workaround. FA3/Ring FlashAttention is likewise unsupported only
on this tested SM120 stack: the installed Hopper extension has no SM120 kernel
image. This is an environment-specific result, not a claim about every FA3
build or GPU architecture.

Raw measured artifacts are retained. Warmup `video.npy` and `audio.npy` are
removed after a successful warmup to limit disk use; pass
`--keep-warmup-artifacts` to retain them. The commands, logs, resolved
acceleration config, metadata, and status remain in each warmup directory.
