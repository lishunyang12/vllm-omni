# Track A (continuous full-duplex) — status & handoff

Goal: make the clean event-sourced Track A path (`experimental/fullduplex/core` FSM +
`OmniDuplexEnginePort`) drive MiniCPM-o 4.5 continuous full-duplex, opt-in via
`/v1/realtime?duplex=1&runtime=fsm`, so the official OpenBMB demo frontend (through
`official_demo_bridge_worker.py`) runs on it. Track B (`openai/serving.py`) is the
validated path and works end-to-end.

## ✅ ROOT CAUSE OF THE NaN CRASH — FOUND & FIXED

The prior theory in this doc (missing ref-audio context → uninitialized NaN) was **wrong**.
Verified with tensor-level instrumentation on the live model, comparing Track A vs Track B on
the same server/model/input:

- The session config reaching the model is complete: `extra_body.ref_audio_data` present,
  decodes to a valid waveform, and the reference-audio context embeddings build cleanly
  (`get_audio_hidden_states -> (60, 4096)`, no NaN). `ctx_embeds=13` was a red herring — it is
  the **list length** (12 text-token elements + 1 ref-audio tensor of 60 rows = 72 embedding
  rows), matching the scheduler reserve of 72. Ref audio was never the problem.
- The NaN was isolated to the **incoming streaming audio chunk**: `mel_feat nan/inf=True
  absmax=nan` on Track A, but `absmax=1.5` clean on Track B — for the *same* input bytes and
  the *same* stage0 code.

**Actual cause:** the FSM inbound adapter ignored the session-level `input_audio_format`. The
OpenAI realtime dialect (and the OpenBMB demo/bridge) declare the input encoding **once** on
`session.update` (`input_audio_format: "pcm16"`) and send every `input_audio_buffer.append`
with **no per-frame format**. `_extract_audio_payload` defaulted the missing format to
`pcm_f32le`, so the pcm16 bytes were **never converted** and were later reinterpreted as
float32 (`np.frombuffer(dtype=float32)`), producing garbage → NaN mel features → NaN input
embeddings → `CUDA device-side assert (probability tensor contains inf/nan)` → EngineDeadError
on the first speak forward. Track B works because its serving path honors the session-level
`input_audio_format` and converts pcm16→f32le.

**Fix (in tree):**
- `openai/inbound.py`: `_extract_audio_payload(frame, *, default_format)` and
  `RealtimeInboundAdapter(input_audio_format=...)` — a missing per-frame format now falls back
  to the session-declared format, so pcm16 streams convert correctly. +2 regression tests
  (`test_inbound.py`).
- `openai/handler.py`: reads `session.input_audio_format` from the opening `session.update`
  and threads it into `run()` → `RealtimeInboundAdapter`.

**Validated live:** with the fix, Track A no longer NaNs — mel features are clean, input
embeddings are finite, the engine stays alive, and the model runs listen decisions across
multiple chunks and **transitions to speak** after trailing silence. (Test harness: scratchpad
`direct_fsm_probe.py`, driving `/v1/realtime?duplex=1&runtime=fsm` with pcm16 like the bridge.)

## ⏭ NEXT BLOCKER — output normalization for the speak path

With the NaN fixed, the next gap is the **output side**. The FSM port's `output_mapper`
(`MiniCPMO45ModelEventAdapter.map_output`) expects a pre-normalized flat dict
(`is_listen`/`audio_data`/`text`), but the real engine output is a `RequestOutput`/
`OmniRequestOutput` whose `multimodal_output` carries `{duplex_prompt_token_ids, finished,
latent, meta, text}` — **no `is_listen`, no `audio_data`**. The listen/speak decision must be
derived (Track B's `_data_plane_native_decision`: from `meta` special-token-ids + completion
token-ids), and the spoken audio comes from a **separate stage-1 TTS output** that must be
decoded into `audio_data`.

Progress this session: `OmniDuplexEnginePort._extract_mm_output` / `_normalize_engine_output`
now robustly unwrap `multimodal_output` from the engine output (mirroring the Track B
extraction chain) and pass domain events / mappings through untouched (all 206 unit tests
pass). What remains is to port the **classification + audio extraction** from Track B's
`_native_results_from_data_plane_output` (serving.py, ~600 lines): native listen/speak
decision, stage-1 audio-chunk decode, turn/epoch staleness drop, and segment text-delta dedup.
Current symptom without it: a stage-0 listen output is mis-classified as speak and the reducer
rejects it (`illegal ModelTextDelta transition ... model was not speaking`).

## Fallback / working demo today

Track B works end-to-end. The official demo runs via the bridge with `&runtime=fsm` dropped.
The path-alias fix in `official_demo_bridge_worker.py` (`/v1/worker/duplex`) makes the official
frontend work for Track B now.

## Test-server recipe

Scratchpad `start_test_server.sh` (GPUs 5,6, port 8100, ModelScope weights,
`--gpu-memory-utilization 0.30`) + `direct_fsm_probe.py` (drives pcm16 appends, saves any reply
audio). Reload cycle ~2 min (weights cached). Full-duplex unit tests: `pytest tests/fullduplex`.
