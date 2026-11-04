# Hy-Nomas history and interaction extensions

> **Status: initial implementation and follow-up design.** The C1 archive is
> opt-in and needs real-model/distributed qualification. C2–C8 and D1–D3 below
> are implementation research, not advertised support.

## Scope and inspected revisions

This extends [the C/D roadmap in #9525](https://github.com/vllm-project/vllm-omni/issues/9525)
for developers serving controllable, long-running Hy-Nomas video sessions.
The initial change starts with C1, following the owner's stated C1/C2 order.
Keep BF16, full history, and eight sampling steps as defaults.

- Model source: [#9338 at b3eec672](https://github.com/wtz2333/vllm-omni/tree/b3eec6720ba7f13112050608afb3aa00611a34fd).
  This revision has native DiT/PixelUMM modules, full/recent history, and
  segmented KV publication. Older #9338 evidence applies to its stated
  revisions; importing the latest source does not requalify those results.
- Official source: [Hy-Nomas at 0bd24056](https://github.com/Tencent-Hunyuan/Hy-Nomas/tree/0bd240561567c0245d8ab8695c962337a8ea77fa).
  Reference paths below are relative to this pinned source tree.
- Fork base: `mooncake2026/vllm-omni:feat/hy-nomas` at `5d2cac68` contained
  the earlier integration proposal, without model code. The first commit ports
  the #9338 integration diff, including empty-frame metadata delivery for the
  fork's video-only encoder, without unrelated main updates. The second commit
  adds C1 and this research.

The [model integration contract](../module/diffusion/diffusion_model_integration.md)
and [AR session contract](realtime_ar_diffusion.md) own runtime boundaries.
The [recipe](../../../recipes/Tencent/Hy-Nomas.md) owns runnable configuration.

## C1: bounded CPU archive

### Ownership and representation

`HyNomasSession.history_archive` owns CPU copies of each newly committed chunk
when `ar_history_mode=recent` and `ar_history_archive_max_bytes > 0`.
The archive contains all completed chunks, including ones still resident on
GPU. A record contains:

- absolute latent start/count and spatial tokens per latent frame;
- segment ID, appended text-token count, and action-text offset;
- per-layer, local-head K/V tensors, already position encoded;
- the original visibility mask, before later segments hide old action text.

Video rows precede text rows. `frame_rows(position)` maps an absolute latent
position to its video slice; text starts at `frame_count * tokens_per_frame`.
Text is appended only when the baseline commits it, including recaching after
recent-history eviction. Transient conditioning rows are excluded.
No GPU segment handle is retained in the CPU archive.

`ARDiffusionSegmentedKVState.pending_layer_views()` borrows initialized current
rows after every layer is staged. The model copies those views before staging
can be overwritten. It does not inspect allocator internals or change the
attention backend, retained set, RoPE positions, or sampling RNG.

### Publication and failure

1. Run the existing required KV-commit forward and stage every layer.
2. Validate the GPU retention set and the archive's prospective byte/frame use.
3. Copy current K/V and mask to CPU, preparing an unpublished record tuple.
4. Use the existing SP error agreement before GPU publication.
5. Commit GPU retention, then publish the prepared archive tuple.

Archive copies are blocking and once per layer/chunk. This adds D2H work;
no speedup is claimed. An archive copy/quota failure participates in the
existing rank error check. Local or peer failure discards pending CPU/GPU
state without publishing it. Reset, close, cancellation, and runner failure
cleanup clear the session archive. This reuses the baseline publication
contract; it does not provide recovery after process/device loss during commit.

### Bounds and limit behavior

`ar_history_archive_max_bytes` limits committed plus pending CPU tensor payload
per session **per rank**, including K, V, and masks. The default `0` disables
archival. `ar_history_archive_max_frames` separately bounds latent positions
and record count; its default is `4096`. Python object/tuple overhead is
bounded by record count but is not included in the tensor-byte metric.

Before copying a chunk, both prospective limits must fit. Otherwise the request
fails explicitly and the runner releases the session. There is no silent
archive truncation or policy downgrade. A deployment should reserve up to
`session_capacity * SP_world_size * archive_max_bytes` host tensor memory,
plus models, Python metadata, decoder/output buffers, and other service state.
The archive quota is separate from the GPU admission ledger and does not
reserve host RAM physically or guarantee that allocation cannot fail.

Chunk metadata reports `archive_payload_bytes` and `archive_latent_frames`
for the local rank; both are zero when archival is disabled. These are payload
counters, not total process RSS or aggregate multi-rank measurements.

An archive is not a resumable checkpoint: RNG, decoder windows, conditioning
objects, interaction queues, and client timeline state are not captured here.
C7/C8 must add and budget those before advertising recovery.

## C2: history sampling follow-up

The official `hymm/models/dit/modules/uniform_history_cache.py` provides
`history_frame_indices`, `ArchivedSegment`, `commit_uniform_history`, and
`_rebuild_history`. Its archive grows with generation; copying that storage
policy directly would violate C1's explicit quota contract.

| Policy | Reference selection | Integration requirement |
| --- | --- | --- |
| `uniform` | Endpoint-inclusive integer grid over all previous frames | Restore selected archived video and associated segment text. |
| `recent_mix_uniform` | Keep `K//2` recent frames and sample the remaining budget from the older prefix | Deduplicate positions and preserve chronological order. |
| `temporal_context` | Mirrored power-spaced distances from the newest frame | Preserve reference rounding, endpoints, and configurable positive finite gamma. |

These policies count sampled video frames and add associated text; baseline
`recent` has a shared video/text token budget. C2 must declare GPU capacity for
text and pending rows explicitly rather than treating these budgets as equal.

A changed sampling grid can select a frame already evicted from GPU. C2 therefore
needs transactional archive-to-page restoration, a new retained set, and aligned
visibility/action masks. It must preserve the archived absolute positions rather
than renumbering them to physical row indices. No new selection mode is enabled
by C1. Compare all three policies against the same pinned oracle, selected rows,
noise, and budgets before the 120-second and W6 quality qualification.

## Remaining C/D work

| Item | Verified reference / current gap | Proposed first boundary and required evidence |
| --- | --- | --- |
| C3 INT8 KV | `hymm/models/dit/modules/int8_kv_cache.py` stores integer payload plus FP32 scales and dequantizes for attention. Omni currently owns floating-point pages/staging. | Add storage and actual-read support together. Count scales and BF16 staging; compare fixed-capacity memory separately from fixed-memory capacity and latency. Test eviction, reset, failures, and sustained error. |
| C4 QVG | `qvg_kv_cache.py` uses `quant-videogen` PRQ payload/codebooks and separate text storage. | Follow the qualified C3 lifecycle contract; account for encode/decode workspaces, metadata, isolated RNG, and archived payload. Independently measure quality/memory/latency. |
| C5 INT8 DiT | `int8_stored_bf16_linear.py` loads W8 storage and computes `F.linear` with dequantized BF16 weights. Omni rejects quantized Hy-Nomas configuration. | Validate checkpoint keys/shapes/scales and actual kernel execution. W8 storage alone does not establish INT8 matrix multiplication or speedup. Qualify W6 and controls. |
| C6 FP8 PixelUMM | `hymm/models/vae/pixelumm_decoder/prequantized.py` validates serialized weights/scales and installs FP8 backend modules. | Match native Omni decoder names and backend geometry. Compare identical-latent decode first, then full streams; measure conversion overhead and real FP8 engagement. |
| C7 undo | `prepare_history_rollback` / `apply_history_rollback` in `uniform_history_cache.py`; reference recent-only history rejects rollback. | C1 is only the KV prerequisite. Snapshot RNG, conditioning, decoder, masks/scales and positions; negotiate client timeline replacement. Reject unavailable history before mutation. |
| C8 pause/resume | Official web/job-store paths manage live jobs; archived KV alone is insufficient to restore a session. | Bound pause TTL, identity, quotas and cleanup within one process. Release GPU pages; restore full generation/media state with correct PTS. Explicitly reject process-restart recovery initially. |
| D1 Nomas-PE | Official PE checkpoint, `apps/seg_ar_gateway.py`, prompt templates and enhancer adapters; Omni already accepts direct text/structured prompts. | Optional service before generation admission, with separate timing and explicit timeout/schema errors. Compare direct submission of the resulting enhanced text. |
| D2 LoRA | `scripts/train/merge_lora.py` validates A/B names/shapes and merges `(alpha/rank) * B @ A` into the distilled checkpoint. Omni currently rejects `lora_path`. | Start with a specified adapter and strict native-to-Diffusers mapping or validated merged weights. Verify zero effect and official active-adapter parity; leave quantized combinations unsupported until tested. |
| D3 browser controls | Official `apps/frontend/src/` has camera interaction; Omni `hy_nomas/interaction.py` already accepts camera events. | Adapt a browser UI to the existing protocol. Test press/release, blur, disconnect, coalescing and effective chunks, including a slow reader; prevent stale held controls. |

C1 → C2 is the initial sequence; C7/C8 need additional complete-state contracts.
C3 can proceed against A4 independently of C2; C4 follows C3. C5/C6 are separate
quantization efforts. D1–D3 can build on #9338 independently, with D3 reusing
A4 serving qualification. No C/D combination is implicitly qualified.

## Validation boundary

CPU tests cover archive payload independence, exact accounting, position mapping,
quota rejection, missing-layer reads, peer/local rollback, default-off behavior,
output equality with substituted learned components, and session reclamation.
The ordinary model and segmented-KV regression suites remain applicable.

Before promoting this draft, run real BF16 SP1/SP2 generation with archive off/on,
identical noise and controls, repeated GPU-history eviction, injected copy/rank
failures, and close/reset/cancellation. Report D2H time, actual host RSS and per-rank
payload alongside GPU pages/staging. The RFC's 120-second service matrix and W6
are separate gates; CPU tests do not establish them.
