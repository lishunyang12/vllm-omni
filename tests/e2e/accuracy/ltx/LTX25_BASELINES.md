# LTX-2.5 accuracy and latency baselines

These commands use the Diffusers PR #14447 merge commit and the local
`Lightricks/LTX-2.5-Diffusers` snapshot. The verified integration environment
is vLLM 0.27.0, vLLM-Omni 0.27.0.dev106, PyTorch 2.13.0+cu130, and Diffusers
0.40.0.dev0. Do not combine the older vLLM-Omni 0.26 and vLLM 0.24 packages
when reproducing these results.

All runs save raw `video.npy`, `audio.npy`, and `metadata.json`; MP4 encoding
is intentionally outside the accuracy and latency comparison. `CUDNN_ATTN` is
the default and recommended production backend for this integration.
`TORCH_SDPA` remains an explicit, reproducible accuracy baseline.

## Environment

```bash
cd <repo>
source <repo>/.venv/bin/activate

export LTX25_MODEL=<model>
export DIFFUSERS_REF=<diffusers-repo>
export PYTHONPATH="$DIFFUSERS_REF/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export REQUEST=tests/e2e/accuracy/ltx/ltx25_distilled_request.json.example
export OUTPUT_ROOT=<output>
```

`DIFFUSERS_REF` must be at commit
`7564fb016dabda0c943416190fc92398c50b1b20`.

Confirm the aligned runtime before a measured run:

```bash
python - <<'PY'
import torch, vllm, vllm_omni

print("torch", torch.__version__)
print("vllm", vllm.__version__)
print("vllm-omni", vllm_omni.__version__)
PY
```

## Diffusers model-card baselines

Distilled one-stage T2V (960x544 output):

```bash
python tests/e2e/accuracy/ltx/run_ltx_reference.py \
  --backend diffusers \
  --mode distilled-one-stage \
  --request "$REQUEST" \
  --model "$LTX25_MODEL" \
  --output-dir "$OUTPUT_ROOT/diffusers-one-stage-t2v"
```

Distilled one-stage I2V uses the same request and an explicit first frame:

```bash
python tests/e2e/accuracy/ltx/run_ltx_reference.py \
  --backend diffusers \
  --mode distilled-one-stage \
  --request "$REQUEST" \
  --image /absolute/path/to/first-frame.png \
  --model "$LTX25_MODEL" \
  --output-dir "$OUTPUT_ROOT/diffusers-one-stage-i2v"
```

Distilled two-stage T2V runs stage 1 at 960x544, performs x2 latent
upsampling, and applies the official three-sigma tail at 1920x1088:

```bash
python tests/e2e/accuracy/ltx/run_ltx_reference.py \
  --backend diffusers \
  --mode distilled-two-stage \
  --request "$REQUEST" \
  --model "$LTX25_MODEL" \
  --output-dir "$OUTPUT_ROOT/diffusers-two-stage-t2v"
```

Full/SFT T2V loads `transformer_full/`, restores the official dynamic
scheduler, and uses its 30-step defaults:

```bash
python tests/e2e/accuracy/ltx/run_ltx_reference.py \
  --backend diffusers \
  --mode full \
  --request "$REQUEST" \
  --model "$LTX25_MODEL" \
  --output-dir "$OUTPUT_ROOT/diffusers-full-t2v"
```

The diffusion decoder is not in this runnable matrix. It is an alternate,
stochastic decode path rather than a denoising baseline and requires the
external `kernels`/NATTEN runtime. Compare convolutional decode first; add a
separate decoder experiment once that dependency is installed and pinned.

## vLLM-Omni comparison

Run the same one-stage request through the native pipeline:

```bash
python tests/e2e/accuracy/ltx/run_ltx_reference.py \
  --backend omni \
  --mode distilled-one-stage \
  --request "$REQUEST" \
  --model "$LTX25_MODEL" \
  --model-class-name LTX2Pipeline \
  --omni-attention-backend CUDNN_ATTN \
  --output-dir "$OUTPUT_ROOT/omni-one-stage-t2v"
```

For first-frame I2V, add the same `--image` argument used by the Diffusers run.
For two-stage T2V, use `--mode distilled-two-stage` and
`--model-class-name LTX2DistilledPipeline`. The runner converts the model-card
stage-1 size to Omni's final-output size convention.

Full/SFT T2V is supported through the dedicated `transformer_full/` component
profile:

```bash
python tests/e2e/accuracy/ltx/run_ltx_reference.py \
  --backend omni \
  --mode full \
  --request "$REQUEST" \
  --model "$LTX25_MODEL" \
  --model-class-name LTX2FullPipeline \
  --omni-attention-backend CUDNN_ATTN \
  --output-dir "$OUTPUT_ROOT/omni-full-t2v"
```

Full/SFT currently supports T2V only. The official LTX-2.5 recipe defines I2V
for the distilled one-stage pipeline, not Full or distilled two-stage.

To collect the SDPA baseline, repeat an Omni command with
`--omni-attention-backend TORCH_SDPA`. Do not use `TRTLLM_ATTN` for LTX-2.5:
the transformer uses Q/K/V head dimension 64, while the current vLLM-Omni
backend advertises head size 128 and the installed TensorRT-LLM kernels reject
64. Both one-stage I2V and two-stage T2V stop during dummy warmup before an
artifact is produced. `CUDNN_ATTN` accepts this shape and remains the production
backend.

## Verified vLLM 0.27 smoke results

The following are cold, single-run checks, not throughput claims:

| Recipe | Omni backend | Result | Generation | Peak GPU memory |
| --- | --- | --- | ---: | ---: |
| Distilled one-stage T2V matrix baseline | CUDNN_ATTN | completed | 4.25 s | 78,016 MB |
| Distilled one-stage I2V | TORCH_SDPA | completed | 7.99 s | 78,650 MB |
| Distilled two-stage T2V | TORCH_SDPA | completed | 19.04 s | 113,506 MB |
| Full/SFT T2V | TORCH_SDPA | completed | 51.47 s | 79,708 MB |
| One-stage I2V / two-stage T2V | TRTLLM_ATTN | blocked before generation | N/A | N/A |

Against the corresponding current-environment Diffusers runs, the SDPA
one-stage I2V result measured SSIM 0.99155, PSNR 36.48 dB, LPIPS 0.00599, and
5.52x generation speedup. The SDPA two-stage T2V result measured SSIM 0.81394,
PSNR 19.27 dB, LPIPS 0.16219, and 4.22x generation speedup. These parity
numbers are recorded as evidence; the two-stage quality gap must not be
presented as a passed release gate.

## Acceleration decisions

The detailed commands and raw results are documented in
[`LTX25_ACCELERATION_MATRIX.md`](LTX25_ACCELERATION_MATRIX.md). The distinction
between functional and recommended is important: completing a real request
does not imply a quality pass or a latency improvement.

| Feature | Functional status | Current result | Recommendation |
| --- | --- | --- | --- |
| HSDP2 | verified | eager-identical; 21.9% lower peak memory; 1.27x slower generation | use only for two-GPU capacity relief |
| FP8 | verified path, failed quality gate | 23.4% lower peak memory; 0.97x generation speedup; SSIM 0.72023 | not release-recommended |
| ordinary layerwise offload | verified | 44.3% lower peak memory; 2.57x slower generation; eager-reference parity | single-GPU capacity fallback |
| DP2 / Distributed Layerwise Offload | pending | DP2 AllGather smoke has not run | no functional or performance claim yet |
| Cache-DiT | provisional path only | adapter attached, but output was eager-identical and 1.73x slower | rerun after adapter fix before publishing final numbers |
| TeaCache | unsupported | no LTX extractor and no calibrated coefficient profile | do not enable |
| TRTLLM_ATTN | unsupported | head dimension 64 is rejected during dummy warmup | use CUDNN_ATTN |
| FA3 / Ring FlashAttention | unsupported in this SM120 environment | installed Hopper extension has no SM120 kernel image | do not enable on this tested stack |

The verified ordinary layerwise run used `enable_layerwise_offload=True`; it
must not be described as DLO. Distributed Layerwise Offload is a separate
multi-process configuration using `enable_distributed_layerwise_offload=True`
and an AllGather or broadcast weight path.

TeaCache fails two source-of-truth checks. `LTX2VideoTransformer3DModel` has no
entry in the TeaCache `EXTRACTOR_REGISTRY`, so the hook cannot define the
modulated-input signal, block execution, and residual reconstruction for this
joint audio/video transformer. It also has no entry in
`_MODEL_COEFFICIENTS`, so there is no calibrated five-coefficient polynomial
for converting relative-L1 changes into the cache decision estimate. Supplying
guessed coefficients does not solve the missing extractor or establish
quality.

## Metrics

```bash
python tests/e2e/accuracy/ltx/compare_ltx_outputs.py \
  --reference-dir "$OUTPUT_ROOT/diffusers-one-stage-t2v" \
  --candidate-dir "$OUTPUT_ROOT/omni-one-stage-t2v" \
  --output "$OUTPUT_ROOT/one-stage-t2v-metrics.json"
```

The JSON contains per-frame SSIM, PSNR, and LPIPS with mean/p95/max
summaries; audio relative-L2 and cosine similarity; load/generation/e2e
latencies and speedups; stage timings; and peak GPU memory. Timings are cold
single-process measurements. Run one unreported warmup plus at least three
measured repetitions when reporting stable throughput.

The checked-in release gates are the offline runner, comparison utility, and
acceleration-matrix driver:

- `run_ltx_reference.py` writes deterministic raw outputs and metadata.
- `compare_ltx_outputs.py` computes SSIM, PSNR, LPIPS, audio, latency, and
  memory comparisons.
- `run_ltx_acceleration_matrix.py` snapshots each case and applies declared
  quality gates.

They intentionally remain offline scripts rather than heavyweight model-loading
pytest cases.
