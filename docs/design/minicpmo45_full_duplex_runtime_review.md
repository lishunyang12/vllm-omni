# MiniCPM-o 4.5 Native Full-Duplex Runtime Review Guide

## 1. Purpose and Current Status

This document describes the full-duplex runtime work in PR #3907 and the
follow-up migration from the legacy `--stage-configs-path` deployment to the
standard deploy-config pipeline.

The implemented and remotely verified checkpoint is:

- MiniCPM-o 4.5 owns the `listen`/`speak` decision.
- An audio-only client can commit speech without sending `response.create`.
- One WebSocket session can complete at least three clean, distinct input turns.
- Stage0 conversation context survives clean turn boundaries.
- Stage1 TTS and Token2Wav state is reset at the model turn boundary.
- Text, audio, response, and scheduler lifetimes are separated.
- Duplex identity is carried as one `DuplexFence` across serving, engine,
  orchestrator, scheduler output, and stage handoff paths.
- The native duplex deployment uses `--deploy-config` and does not depend on the
  legacy MiniCPM streaming stage-config file.

This checkpoint does not claim:

- automatic or VAD-driven barge-in;
- scheduler-native KV append or a first-class persistent KV lease;
- bounded KV for minute-scale conversations;
- production multi-session capacity or concurrency guarantees.

The interruption reducer and epoch fence exist as a common contract, but
MiniCPM-o 4.5 does not currently provide a validated automatic interruption
source. A clean multi-turn native duplex checkpoint must not be presented as a
validated barge-in implementation.

## 2. Base and Compatibility

The review branch is rebased on the current local `origin/main` used for this
checkpoint:

```text
origin/main: ff6f0da271bd1c3658c3e46e4e2cdc4db17219a5
runtime checkpoint before deploy migration:
             4159a910ceac24ab63f8b9ea799e0406ae56c279
```

`origin/main` is an ancestor of the branch. The implementation was validated in
the remote CUDA environment with vLLM 0.24-compatible APIs. The full-duplex
package remains under `vllm_omni.experimental` because the scheduler append and
production lifetime contracts are not upstream-stable APIs yet.

## 3. Design Principles

### 3.1 Separate logical lifetimes

The previous implementation mixed five different lifetimes. The revised design
defines them explicitly:

| Lifetime | Identity / owner | Ends when |
| --- | --- | --- |
| WebSocket session | `session_id` | client closes or runtime fails |
| interruption generation | `epoch` | interruption/context rebuild |
| user turn | `turn_id` | the next committed user input allocates a new turn |
| OpenAI response | `response_seq` / `response_id` | model `turn_eos` or failure |
| engine stage resource | fence-derived request ID | epoch/session close or stage failure |

An OpenAI response is not the scheduler request. Clean turns reuse session-level
engine resources while each committed user turn gets a new external response.
This is what allows `response.done` to close one assistant reply without
destroying the session or the Stage0 conversational context needed by the next
turn.

### 3.2 One cross-layer identity

`vllm_omni.experimental.fullduplex.core.identity.DuplexFence` is the canonical
cross-layer identity:

```python
DuplexFence(
    session_id: str,
    epoch: int,
    turn_id: int,
    response_seq: int,
)
```

The fence is attached to typed engine messages, orchestrator request state,
stage bindings, multimodal metadata, output events, cursors, and teardown
operations. Missing fence metadata is an error on typed duplex paths. A valid
old fence is stale output and is dropped; an impossible newer or mismatched
fence fails loudly.

Clean turns advance `turn_id` and `response_seq` while preserving `epoch`.
Interruption advances `epoch` atomically and invalidates all prior output.

### 3.3 Keep model policy out of serving

MiniCPM-o samples native `listen` and `speak` tokens. Serving must not make the
model speak by rewriting probabilities, forcing a token, or converting a
`listen` decision into a response. The generic runtime performs transport and
lifecycle operations; the MiniCPM adapter maps model outputs to typed events.

The implementation therefore removed the successful-run dependency on:

- `listen_prob_scale=0.0` as a forced-speak workaround;
- serving-side `force_listen`/`force_speak` turn policy;
- client `response.create` in `auto_response` mode;
- punctuation or TTS segment completion as the assistant turn boundary.

### 3.4 Isolate the current scheduler workaround

Current vLLM does not expose the scheduler-native append/session-KV primitive
needed by the target architecture. The existing resumable request and scheduler
data-plane behavior is isolated behind
`experimental.fullduplex.engine.OmniDuplexEnginePort` and typed engine methods.

The core lifecycle, OpenAI projection, and MiniCPM adapter do not depend on
placeholder-token accounting. A future native append implementation can replace
the engine port without changing the domain events or response lifecycle.

## 4. Package Boundaries

```text
vllm_omni/experimental/fullduplex/
  core/
    identity.py       immutable `DuplexFence`
    events.py         typed domain events and effects
    state.py          pure session/turn/response reducer
    runtime.py        effect execution and task ownership
    ports.py          engine and event-sink protocols
    playback.py       generated/sent/played/committed cursor
  engine/
    omni.py           current orchestrator/scheduler data-plane adapter
    intermediate.py   typed Stage0 -> Stage1 payload helpers
    worker.py         loaded-model runner/provider integration
  openai/
    protocol.py       duplex protocol schema and session registry
    realtime_session.py Realtime input/output schema projection
    websocket.py      reader/writer queues and task ownership
    serving.py        session controller and engine/protocol orchestration
    realtime.py       typed reducer event projection
    history.py        response lifecycle ledger
    data_plane.py     per-fence output cursors
    audio.py          audio format conversion
  minicpmo45/
    policy.py         token names, framing, and scheduler accounting rules
    input.py          PCM chunk buffering and commit accounting
    adapter.py        model events and serving-side session preparation
    stage0.py         native thinker/listen/speak runtime
    stage1.py         talker/Token2Wav runtime
    worker.py         stage provider hooks
    runtime.py        MiniCPM adapter exports
  joyvl/
    ...               second model package using the same core contracts
```

The generic package owns lifecycle and identity. Model token IDs, MiniCPM audio
unit sizing, reference-audio setup, Stage0 state, and Stage1 handoff rules stay
inside `minicpmo45`.

`openai/serving.py` remains the largest integration module because it projects
the existing `/v1/duplex` and `/v1/realtime?duplex=1` protocols onto the engine
adapter. It no longer contains the model Stage0/Stage1 implementation, audio
codecs, WebSocket actor, or Realtime schema state. Further reduction of this
controller is possible, but is not required to replace the current engine port.

## 5. State Machine and Normal Turn Flow

The pure reducer uses these turn phases:

```text
IDLE
  -> INPUT_STREAMING
  -> TURN_COMMITTED
  -> AWAITING_MODEL
  -> RESPONDING
  -> IDLE
```

The normal native audio flow is:

1. `input_audio_buffer.append` is decoded and normalized to 16 kHz
   `pcm_f32le`.
2. `MiniCPMO45PcmAppendBuffer` emits only complete model audio units. It tracks
   per-turn `had_input` and `had_speech` independently from residual bytes.
3. Incremental units are appended with `final=False` to the stable fenced
   engine session.
4. `input_audio_buffer.commit` advances the logical turn and reserves one
   response when `auto_response` is enabled and the turn had speech.
5. Auto-response does not depend on `flush()` returning residual PCM. A commit
   exactly on a model-unit boundary is still a valid committed speech turn.
6. The model continues its native loop and samples `listen` or `speak`.
7. The first `speak` transition creates exactly one external response.
8. Stage0 text/hidden-state handoffs drive Stage1, which streams PCM/audio
   deltas.
9. TTS segment end closes the segment only.
10. Model `turn_eos` closes audio/content/response exactly once and resets
    turn-local Stage1 state.
11. The WebSocket session and Stage0 conversational state remain available for
    the next input commit.

In `auto_response=true`, the client sends audio append and commit events only.
Sending a manual `response.create` at the same time would enable two response
drivers and is intentionally rejected instead of being timed around an active
response.

## 6. Stage0 and Stage1 Data Flow

### 6.1 Stage0 ownership

Stage0 owns the model-native continuous state:

- streaming audio encoder state;
- thinker/talker model state used for native listen/speak decisions;
- conversational KV and model context across clean turns;
- current turn-ended latch;
- accumulated Stage0 TTS conditioning for the active turn.

Stage0 state is session-scoped unless explicitly documented as turn-local.
Clean `turn_eos` must not erase conversational context. An epoch change may
rebuild Stage0 from playback-committed history.

### 6.2 Stage0 to Stage1 handoff

`stage_input_processors/minicpmo_4_5_omni.py` converts the Stage0 multimodal
output into the existing Stage1 input shape. The handoff carries the complete
accumulated TTS condition for the active turn because Stage1 tracks a consumed
cursor. It does not treat a new handoff as a new conversation turn.

Handoff metadata includes flat scalar keys such as:

- `meta.duplex_epoch`
- `meta.duplex_turn_id`
- `meta.segment_end`
- `meta.turn_end`
- `meta.tts_is_last_chunk`

Flat metadata avoids the previous nested/flat merge mismatch and passes through
the output processor's explicit metadata handling.

### 6.3 Stage1 ownership

Stage1 owns turn-local speech generation state:

- consumed-token cursor into the cumulative handoff;
- talker LM turn state;
- Token2Wav token buffer and vocoder stream state;
- audio offset and text/audio alignment metadata.

These are reset on model turn end or a fenced interruption. They are not reset
at punctuation, a TTS segment boundary, or every engine output batch.

### 6.4 Token2Wav continuity fix

The prior path could finalize or clear Token2Wav at ordinary punctuation
segments. That broke multi-segment speech and could replay an earlier turn's
tail when the next turn was short or empty.

The revised contract distinguishes:

- `segment_end`: one TTS segment is complete; retain turn state;
- `turn_end`: the model sampled its turn EOS; finalize and clear turn state.

The consumed cursor and cumulative handoff are reset together at the turn
boundary. Stage0 KV is deliberately preserved. This prevents both directions
of the bug: prior-turn leakage and cross-turn amnesia.

## 7. Response and Protocol Lifecycle

The Realtime projection enforces one lifecycle per fenced response:

```text
response.created
response.output_item.added
response.content_part.added
response.audio.delta / transcript delta ...
response.audio.done
response.content_part.done
response.output_item.done
response.done
```

`ResponseLifecycleLedger` and per-response terminal sets make terminal events
idempotent. `response.created` is keyed by the fenced response rather than by a
raw string grep count, which avoids both actual duplicate creation and false
diagnosis from nested event payload text.

Text and audio use per-fence cursors. A new turn cannot reuse the previous
turn's cumulative text cursor, audio offset, or text/audio marks. Late output
after `response.done` is rejected or stale-dropped depending on its fence.

Playback acknowledgement tracks four monotonic positions: generated, sent,
played, and committed. Only playback-committed assistant history is eligible
for reconstruction after interruption. This contract is present even though
automatic barge-in is out of scope for this checkpoint.

## 8. Interruption Contract and Current Barge-In Scope

The reducer defines one interruption transition:

1. capture the old fence;
2. increment `epoch`;
3. cancel old-fence work;
4. stale-drop late old-fence output;
5. reset Stage1 with the old fence;
6. rebuild Stage0 from playback-committed history;
7. accept new output only under the new fence.

There is no second barge-in-specific response state machine in the core design.
A future VAD, model signal, or explicit client control can emit
`InterruptRequested` through the same transition.

MiniCPM-o 4.5 currently exposes model-owned `listen`/`speak`, but the validated
official loop does not provide a reliable, explicit automatic barge-in event.
For that reason, this PR does not enable or claim automatic/VAD barge-in. The
fencing mechanics are unit-tested infrastructure, not an E2E capability claim.

## 9. Engine and Scheduler Integration

The current engine adapter exposes typed fenced operations:

- open duplex session;
- append duplex input;
- signal turn lifecycle;
- stream fenced output;
- cancel a fence;
- close session resources.

The orchestrator binds one replica per stage for a session/epoch and derives
stage request IDs from the fence. It rejects missing fence state and keeps
Stage0 -> Stage1 output identity intact.

This is still a compatibility implementation over current scheduler requests.
It does not claim a core KV lease or scheduler-native append. The capability
surface reports that distinction explicitly so callers cannot infer a stronger
runtime guarantee from a successful demo.

The MiniCPM duplex deployment selects the synchronous AR scheduler for both
stages. This is deliberate, not a general recommendation. Remote validation
showed that the async scheduler path admitted overlapping lifecycle work and
produced four responses for a three-turn scenario. The validated native
append/drain contract currently requires deterministic synchronous scheduling
with `active_stream_window: 1`. Async scheduling can be re-enabled only after a
separate scheduler-level contract and E2E test prove one response per commit.

## 10. Deploy-Config Migration

### 10.1 Why migrate

The legacy launch required:

```text
--stage-configs-path \
  vllm_omni/model_executor/stage_configs/minicpmo45_2gpu_streaming.yaml
```

That bypassed the current pipeline/deploy composition and left duplex session
mode outside the standard deployment schema.

The replacement is:

```text
--deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml
```

The deploy overlay reuses the registered `minicpmo_4_5` pipeline and only
overrides duplex-specific runtime behavior:

```yaml
base_config: minicpmo_4_5.yaml
pipeline: minicpmo_4_5
session_mode: duplex
active_stream_window: 1

stages:
  - stage_id: 0
    async_scheduling: false
  - stage_id: 1
    async_scheduling: false
    default_sampling_params:
      max_tokens: 4096
      extra_args:
        stop_token_names: ["<|im_end|>"]
```

`DeployConfig.session_mode` is propagated into every merged `StageConfig` and
its OmegaConf representation. Non-duplex deploys default to `session_mode:
turn`, preserving existing behavior.

`--trust-remote-code` remains an explicit launch option. MiniCPM-o 4.5 requires
it, but a model-specific deploy file must not silently weaken the global CLI
trust boundary.

### 10.2 Migration files

- Added `vllm_omni/deploy/minicpmo_4_5_duplex.yaml`.
- Added new-schema Stage1 replica overlays for 3, 4, and 8 GPU layouts.
- Added `session_mode` parsing/propagation in
  `vllm_omni/config/stage_config.py`.
- Removed the duplicate legacy `minicpmo45_*.yaml` `stage_args` configs. The
  standard 2, 3, and 8 GPU layouts use the existing `minicpmo_4_5*.yaml`
  deploy configs; replica and duplex variants are thin overlays on those
  configs.
- Updated `examples/online_serving/minicpmo/README.md`.
- Added real deploy composition contract tests in
  `tests/test_config_factory.py`.

## 11. Review Map by Subsystem

### Core lifecycle and identity

- `vllm_omni/experimental/fullduplex/core/identity.py`
- `vllm_omni/experimental/fullduplex/core/events.py`
- `vllm_omni/experimental/fullduplex/core/state.py`
- `vllm_omni/experimental/fullduplex/core/runtime.py`
- `vllm_omni/experimental/fullduplex/core/playback.py`
- `vllm_omni/experimental/fullduplex/core/ports.py`

### OpenAI Realtime and WebSocket projection

- `vllm_omni/experimental/fullduplex/openai/serving.py`
- `vllm_omni/experimental/fullduplex/openai/realtime_session.py`
- `vllm_omni/experimental/fullduplex/openai/websocket.py`
- `vllm_omni/experimental/fullduplex/openai/protocol.py`
- `vllm_omni/experimental/fullduplex/openai/history.py`
- `vllm_omni/experimental/fullduplex/openai/data_plane.py`
- `vllm_omni/experimental/fullduplex/openai/audio.py`
- `vllm_omni/entrypoints/openai/api_server.py`
- `vllm_omni/entrypoints/openai/serving_chat.py`

### Engine, orchestrator, scheduler, and worker path

- `vllm_omni/experimental/fullduplex/engine/omni.py`
- `vllm_omni/experimental/fullduplex/engine/intermediate.py`
- `vllm_omni/experimental/fullduplex/engine/worker.py`
- `vllm_omni/engine/messages.py`
- `vllm_omni/engine/async_omni_engine.py`
- `vllm_omni/engine/orchestrator.py`
- `vllm_omni/engine/output_processor.py`
- `vllm_omni/engine/stage_pool.py`
- `vllm_omni/core/sched/omni_ar_scheduler.py`
- `vllm_omni/worker/gpu_ar_model_runner.py`
- `vllm_omni/worker/gpu_model_runner.py`
- `vllm_omni/worker/mixins.py`

### MiniCPM-o model and bridge path

- `vllm_omni/experimental/fullduplex/minicpmo45/`
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py`
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_llm.py`
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py`
- `vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py`
- `vllm_omni/inputs/data.py`
- `vllm_omni/inputs/preprocess.py`
- `vllm_omni/utils/mm_outputs.py`
- `vllm_omni/data_entry_keys.py`

### Demo and verification

- `examples/online_serving/minicpmo/realtime_duplex_demo.py`
- `examples/online_serving/minicpmo/realtime_web/`
- `tests/fullduplex/`
- `tests/entrypoints/openai/test_duplex_protocol.py`
- `tests/entrypoints/openai_api/test_duplex_handler.py`
- `tests/entrypoints/test_async_omni_duplex.py`
- `tests/entrypoints/test_duplex_fence_propagation.py`
- `tests/engine/test_duplex_runtime.py`
- `tests/worker/test_native_duplex_hooks.py`
- `tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py`

## 12. Main Bugs Fixed

The runtime work closes these observed failure classes:

1. Auto-response and manual `response.create` both driving one turn.
2. Commit auto-response depending on residual bytes returned by `flush()`.
3. Model forced to speak by serving-side probability/token policy.
4. Stage output audio dropped by a base pooling early-return path.
5. Flat and nested metadata representations losing turn identity.
6. TTS segment end incorrectly treated as assistant turn end.
7. Stage1 consumed cursor, cumulative handoff, or Token2Wav state leaking into
   the next turn.
8. Stage0 state being cleared too aggressively and losing conversation memory.
9. Empty or short turns replaying a prior turn's text/audio tail.
10. Response lifecycle closing the external response while also destroying the
    persistent session request.
11. Missing fence metadata silently disabling stale-output protection.
12. Model-specific code and WebSocket/protocol code accumulating in one generic
    serving module.
13. Legacy deploy configuration bypassing the standard pipeline composition.

## 13. Verification Evidence

All pytest, E2E, ASR, and audio-quality work for this checkpoint is run on the
remote H20 environment, not on the local macOS Python environment.

### 13.1 Runtime regression before deploy migration

```text
265 passed
log: /tmp/remote_gpu_logs/6aa1bc29.log
```

Prior clean multi-turn E2E and ASR evidence:

```text
/tmp/remote_gpu_logs/f9acd944.log
/tmp/remote_gpu_logs/6ee3df18.log
```

### 13.2 Deploy migration tests

Focused deploy/session contract:

```text
2 passed
/tmp/remote_gpu_logs/3a138f97.log
```

Config impact classes:

```text
36 passed
/tmp/remote_gpu_logs/23974d95.log
```

Final MiniCPM native duplex regression suite after the deploy migration:

```text
265 passed
/tmp/remote_gpu_logs/cc7aaf15.log
```

Scheduler selection RED/GREEN:

```text
RED, async scheduler selected:
  /tmp/remote_gpu_logs/11e25a35.log
GREEN, synchronous scheduler selected:
  /tmp/remote_gpu_logs/f116b812.log
```

The full config-factory file was not accepted as evidence because it stalled in
an unrelated remote model auto-detection test before reaching this change:

```text
/tmp/remote_gpu_logs/1a327e3a.log
```

### 13.3 New deploy-config E2E

Server startup:

```text
/tmp/remote_gpu_logs/1db6b37e.log
```

Three-turn E2E:

```text
PASS
/tmp/remote_gpu_logs/032bb67b.log
artifacts: /tmp/minicpmo_e2e_pr3907_deploy_config_sync_20260711
```

Observed protocol counts:

```text
response.created: 3
response.done: 3
response.audio.done: 3
response.audio.delta: 15
cancel/listen/stale terminal residue: 0
```

Observed response transcripts:

```text
1. 你好呀，有什么我可以帮到你的吗？
2. 哎，不是说好不聊八卦的吗？
3. 哈，那我们就不聊八卦了嘛。
```

Whisper large-v3 ASR was run against the three per-response WAV artifacts on the
remote H20 host:

```text
/tmp/remote_gpu_logs/01e659b0.log

response_01.wav: 你好呀,有什么我可以帮到你的吗?
response_02.wav: 诶不,是说好不聊八卦的吗?
response_03.wav: 那我们就不聊八卦了嘛
```

All three files contain intelligible Chinese speech and agree semantically with
the protocol transcripts. This is an audio-content sanity check, not a
large-corpus MOS or speaker-similarity claim.

## 14. Reviewer Reproduction

### 14.1 Start the server

```bash
python3 -m vllm_omni.entrypoints.cli.main serve \
  openbmb/MiniCPM-o-4_5 \
  --omni \
  --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8099
```

Wait for the server to report readiness before starting the client.

### 14.2 Run a distinct-input scenario

```bash
python3 examples/online_serving/minicpmo/realtime_duplex_demo.py \
  --url 'ws://127.0.0.1:8099/v1/realtime?duplex=1' \
  --model openbmb/MiniCPM-o-4_5 \
  --input-wav /path/to/turn1.wav \
  --turn-input-wav /path/to/turn2.wav \
  --turn-input-wav /path/to/turn3.wav \
  --require-distinct-inputs \
  --require-audio \
  --output-dir /tmp/minicpmo45_duplex_review
```

The client must not send `response.create` or a force-barge-in event. Review the
per-response WAV files in the output directory and verify:

- three unique response IDs;
- one `response.created`, `response.audio.done`, and `response.done` per turn;
- transcript delta concatenation equals transcript done;
- no previous-turn suffix in the next response;
- audio exists before done and no audio arrives after done;
- no missing-fence, stale-guard-inert, timeout, forced-speak, or forced-listen
  fallback in the server log.

### 14.3 Focused unit tests

```bash
pytest -q \
  tests/fullduplex \
  tests/engine/test_duplex_runtime.py \
  tests/entrypoints/openai/test_duplex_protocol.py \
  tests/entrypoints/test_duplex_fence_propagation.py \
  tests/fullduplex/minicpmo45 \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py

pytest -q \
  tests/test_config_factory.py::TestStageConfig \
  tests/test_config_factory.py::TestDeployConfigLoading
```

## 15. Review Priorities

Reviewers should focus on these invariants rather than only the demo output:

1. A clean commit advances turn/response once but preserves epoch and Stage0
   context.
2. `auto_response` has one response driver and never requires
   `response.create`.
3. Every data-plane output and terminal event carries the same complete fence.
4. Segment end cannot close a response or clear turn-local state.
5. Turn end clears all Stage1 turn-local accumulators together.
6. Clean turn reset does not clear Stage0 conversational KV.
7. An empty/short turn cannot replay prior text or audio.
8. A stage failure or session close releases all fenced request and replica
   bindings.
9. The deploy overlay preserves default non-duplex behavior.
10. Capability reporting does not claim scheduler-native append, core KV lease,
    automatic barge-in, or production concurrency.

## 16. Follow-Up Work

The next architectural tiers are intentionally separate from this checkpoint:

- add an upstream scheduler-native append/session-KV primitive and replace the
  compatibility engine port;
- select and validate an automatic interruption source for MiniCPM-o;
- run the existing epoch interruption contract through full E2E audio fencing;
- implement bounded/windowed conversational KV for minute-scale sessions;
- add multi-session admission, fairness, failure, and capacity tests;
- add a larger audio-quality corpus with ASR and MOS comparison against the
  official Hugging Face loop.

These are not hidden blockers for reviewing the clean multi-turn runtime, but
they remain blockers for a production-level general full-duplex claim.
