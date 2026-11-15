Rubin FP8 slowdown diagnostic reproduction

From this directory, using a NEW output directory:

  mkdir -p /path/to/diagnostic-data
  docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
    -v "$PWD:/validation:ro" -v /path/to/diagnostic-data:/work \
    --entrypoint bash \
    vllm/vllm-openai:cu134-nightly-1648fefab91204f3b15bf8062217316f07f2c4bb \
    /validation/prepare_latency.sh /work

Qwen uses four GPUs. Wan uses only GPU 0 via its pinned default deployment.
The byte-identical executed benchmark records requested four-GPU kwargs even
for Wan; these are not effective placement. Preserve this known behavior to
reproduce these results. Online serving explicitly requests one GPU for Wan.

Per model/precision: five full warmups, three full 50-step measured requests,
two two-step warmups, then a single two-step CPU/CUDA trace. C1, eager.
The isolated microbenchmark follows all model measurements, without overlap.
The portable bootstrap differs only in downloading the two relevant models,
exiting without the cluster collector, and running analysis/microbenchmark.
Source, dependencies, benchmark_case.py, shapes and sampling remain pinned.

The original job reused the prior same-node compilation/autotune caches. A
fresh directory starts cold; all full warmups remain excluded. Cache loading
and the cache hash are recorded by microbench.py; the original cache is not
distributed. Autotuning may select different tactics in another environment.

Known executed bookkeeping bug: metadata profile.files is empty because its
glob expects trace.json rather than trace_rankN.json.gz. Raw trace files exist.
analyze_traces.py enumerates the correct names and records their hashes.
Use a fresh work directory for reruns: the original runner's resume condition
checks full measurements rather than profile completion.

To regenerate analysis from the published raw traces, from fp8-diagnosis/:

  python3 reproduce/analyze_traces.py --results results --output analysis
  python3 reproduce/verify_diagnosis.py . ..

The audit needs the adjacent stabilized/initial records. Python standard
library suffices for analysis/audit. Model execution uses the pinned image.
These three-request checks and two-step traces do not replace the n=20 table.
