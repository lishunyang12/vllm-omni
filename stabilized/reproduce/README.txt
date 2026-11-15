Repeated latency benchmark for PR #10390
========================================

Expose four compatible NVIDIA GPUs (SM100, SM103 or SM107), the pinned container
below and approximately 600 GB of writable storage. This benchmark runs the
same model revisions, prompts, seeds, dimensions and steps as the quality
matrix. Each GPU family needs its own work directory and result bundle.

From the directory containing prepare_latency.sh:

  mkdir -p /path/to/latency-data
  docker run --rm --gpus all --ipc=host \
    -v "$PWD:/validation:ro" \
    -v /path/to/latency-data:/work \
    --entrypoint bash \
    vllm/vllm-openai:cu134-nightly-1648fefab91204f3b15bf8062217316f07f2c4bb \
    /validation/prepare_latency.sh /work

Protocol: five discarded warmup requests, then exactly 20 measured requests
in the same resident engine, concurrency one. All samples are retained.
Each request recreates the generator with the manifest seed. Initialization,
frame validation, PNG saving and metrics are outside the measured interval.
The synchronous Omni.generate call returns completed outputs. This is offline
API wall latency, not HTTP client latency. H3 runs its full t2va pipeline.
Persistent kernel caches remain enabled; result caching is not configured.

A fresh process loads each model/precision case. Base BF16 and PR BF16 controls
are included alongside PR MXFP8 and existing fp8_per_block. Parallelism remains
DiT TP1/Ulysses4, H3 encoder TP4 and other encoders TP1, eager execution.
All GPU families use the same model and variant order in protocol.json.

GPU-count correction: the original kwargs request four GPUs for every case,
but Wan2.2 default deployment fixes devices to GPU0. Its measured runs used
one GPU; H3 and Qwen used four. Original scripts/metadata retain requested
kwargs for provenance. See execution-topology.json for all observed counts.

A case passes the predeclared stability check only if all of these hold:
  - sample standard deviation / mean <= 5%
  - absolute change between first-half and second-half medians <= 5%
  - P95 / P50 <= 1.10
These are operational checks for this run, not statistical confidence bounds.
Failed checks remain visible and are not removed from the report. P50/P95 use
linear interpolation. One fixed prompt/seed per model limits generalization.

Outputs are under /path/to/latency-data/stabilized:
  results/<model>/<variant>/metadata.json   All warmup/measured samples
  results/<model>/<variant>/frame-*.png     First measured output preview frames
  logs/                                   Per-case execution logs
  evidence/                               Runtime, topology and GPU telemetry
  runs.json                               All process outcomes, including failures

After collection, summarize without GPU dependencies:

  python3 summarize_stabilized.py \
    --run Rubin=/path/to/rubin/stabilized \
    --run GB200=/path/to/gb200/stabilized \
    --output /path/to/summary

The summarizer independently recomputes percentiles and stability checks and
verifies them against the recorded summaries. It writes CSV and JSON and
prints a table. Keep raw metadata alongside that table. Model files and source
revisions are immutable; reuse a work directory only for the identical protocol.
Reruns skip completed cases and retain failed attempts in failed-attempts/.

Do not compare these latencies to another engine unless that engine has been
measured with the same workload, output boundary and warmup policy.

H3-Image confirmation protocol (15 warmups, 50 measured requests):

  cp confirmation-protocol.json protocol.json

Then run prepare_latency.sh with a NEW work directory, or with empty stabilized
results while retaining the original result bundle separately. Use this protocol
on both GPU families and retain all four modes, including the base BF16 control.
The initial 5-warmup/20-sample data remains part of the report.

Provenance note: the executed confirmation protocol's two descriptive strings
(cache_policy and sampling) inherited the initial 25/20 counts. Its numeric
fields were 15 warmups and 50 measurements; the runner uses those numeric
fields. Raw executed protocols and request records are preserved unchanged.
The reproduction copy corrects only those descriptions. benchmark_case.py is
byte-identical to the executed script. The published orchestration adds a
resume guard that rejects completed results from a different protocol.

Optional figure rendering from the published evidence directory:

  python3 -m pip install matplotlib==3.10.8
  python3 plot_stabilized.py --report /path/to/evidence/stabilized \
    --output /path/to/latency

This writes PNG and SVG. Bars are P50, whiskers are the observed P5-P95 range,
not confidence intervals. Plot generation reads raw samples and checks its
percentiles against latency.json; it does not require GPUs or model weights.

Online HTTP serving:
  See ONLINE-SERVE.txt for the pinned Docker command and serve_online.sh.
  Online quantization means load-time weight quantization; these serving
  scripts do not change the offline timing definition of the published table.
