Reproduce PR #10513 H3 first-forward measurements

Use two SM90+ GPUs and an NVLink connection. Allow 300+ GiB local checkpoint cache.
The validated image is:
  vllm/vllm-openai:cu134-nightly-1648fefab91204f3b15bf8062217316f07f2c4bb

Copy this directory into an empty work directory as work/bundle. Mount work at /work
in the image, expose exactly two GPUs, and use host IPC. Run:
  bash /work/bundle/bootstrap.sh /work

bootstrap.sh obtains pinned runtime source 6f5a99768f66 and Sana f4035e0a976d,
installs constrained dependencies while preserving image torch/vLLM/Triton/CUDA,
and downloads the immutable MiniMax-H3 snapshot 0919fb684102ef115126612f3fbb84d1027e7317.
The production code is identical to fix commit 731ea29270e2 (that fix changes two tests).
Native FL2VA and Ref2VA partitions are required; the script downloads both.

The official 768p request JSON files are copied verbatim from that snapshot's
scripts/readme/reproducible-768p-{t2va,fl2va,ref2va}-request.sh scripts.
Reference images/video/audio are fetched from their original request URLs and hashed.
The real pipeline produces encoded prompt and condition tensors; no synthetic latents.

Protocol: BF16 full 50-layer DiT, TP1/Ulysses2, text encoder TP2, tiled VAE PP2,
request seed 0, first conditional denoising step of a 50-step schedule. Replays use
identical forward arguments. Three warmups and nine measured calls; synchronized
wall latency is the maximum over both ranks. CUDA-event timings are retained too.
Weights, encoding, warmup, sampler advancement and VAE decoding are excluded.
A deliberate H3_FORWARD_BENCHMARK_COMPLETE exception stops the pipeline after the
measured calls. Success requires both rank reports, finite outputs and unchanged
input hashes. Other failures do not generate a successful summary.

The default runner uses model-level CPU offload when GPU memory is below 110 GiB;
this accommodates GH200 96GB. Weight transfers happen before the measured forward;
the complete DiT is resident throughout its repeated calls. This capacity setting
is explicitly recorded and must match within each Dense/SOL pair.

The optional run_ref2va_conn1.sh is a separate diagnostic: it applies
CUDA_DEVICE_MAX_CONNECTIONS=1 equally to Dense and SOL, runs only Ref2VA and writes
new *-conn1 directories. It does not overwrite the default experiment. Do not mix
results from the two environment settings.

Outputs: results/<task>-<backend>/{protocol.json,rank-0.json,rank-1.json,summary.json,run.log}.
Do not rerun into a directory containing rank reports; use a fresh results directory.
The comparison helper rejects A/B pairs with different input hashes or parallelism.
Nine-sample medians are primary; first-three medians are retained for comparison to
the separately user-provided three-sample SM120 table. The exact SM120 script was
not supplied, so this is a reconstruction of its stated protocol.

This measures forward latency only. It does not validate end-to-end latency,
video/audio quality, SSIM, PSNR, or LPIPS, and does not produce output videos.
