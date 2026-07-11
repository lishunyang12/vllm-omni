# vLLM-Omni · MiniCPM-o 4.5 Online Demo

Gradio-based web UI for **MiniCPM-o 4.5** served via `vllm-omni`'s
OpenAI-compatible endpoints.

The UI supports:

- **Inputs**: text prompt + optional image, audio (file or mic), video.
- **Outputs**: text + speech (WAV player).

## 1. Start the backend server

Pick a deploy config that matches your GPU layout:

| config | GPUs | TP | Notes |
|---|---|---|---|
| `minicpmo_4_5.yaml` | 2 | 1 | Thinker on GPU0, talker+t2w on GPU1. |
| `minicpmo_4_5_3gpu.yaml` | 3 | 2 | Thinker 2-way TP on GPU0/1, talker+t2w share GPU2. |
| `minicpmo_4_5_8x4090.yaml` | 8 | 4 | Thinker 4-way TP on GPU0-3, talker+t2w on GPU4. |
| `minicpmo_4_5_3gpu_stage1_replicas.yaml` | 3 | 1 | Thinker on GPU0, two talker+Token2wav replicas on GPU1/2 for concurrent text+audio serving. |
| `minicpmo_4_5_4gpu_stage1_replicas.yaml` | 4 | 1 | Thinker on GPU0, three talker+Token2wav replicas on GPU1/2/3. |
| `minicpmo_4_5_8x4090_stage1_replicas.yaml` | 8 | 4 | Thinker 4-way TP on GPU0-3, four talker+Token2wav replicas on GPU4-7. |

Then:

```bash
vllm-omni serve openbmb/MiniCPM-o-4_5 \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

For production or internal networks where Hugging Face downloads are slow, pass
a local ModelScope-downloaded checkpoint path instead of `openbmb/MiniCPM-o-4_5`.

### TTS throughput notes

MiniCPM-o 4.5's remote-code `MiniCPMTTS.generate()` currently runs as a
single-request whole-waveform path, so the deploy configs keep Stage1
`max_num_seqs: 1`. Use the `*_stage1_replicas.yaml` configs to scale concurrent
text+audio throughput horizontally.

```bash
vllm-omni serve /path/to/MiniCPM-o-4_5 \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_4gpu_stage1_replicas.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

Talker/token2wav runtime behavior uses checked-in defaults rather than
MiniCPM-specific environment variables. If these knobs need to be exposed, add
them through a first-class stage/model config so deployments have one clear
configuration surface.

Request-level reference audio is cached by content hash before it is passed to
Token2wav. This keeps repeated requests with the same voice prompt from
thrashing Token2wav's prompt cache while still resetting the cache when the
reference audio changes.

### Experimental `/v1/duplex` WebSocket

`/v1/duplex` is the experimental duplex session runtime entry point. It keeps a
WebSocket session actor with separate control/input/output queues, tracks the
active request, epoch, stale-output filtering, playback commit cursor, barge-in
cancellation, and runtime-control acknowledgements. Generic non-native sessions
can still fall back to normal chat streaming requests; MiniCPM-o 4.5 native
sessions use the scheduler data-plane path described below.

Start the server with the duplex-specific deploy config. The regular
`minicpmo_4_5.yaml` deploy does not opt into duplex sessions and keeps the
non-streaming Stage1 token budget.

```bash
vllm-omni serve openbmb/MiniCPM-o-4_5 \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

MiniCPM-o 4.5 also has an explicit experimental native mode:

```json
{"extra_body": {"minicpmo45_native_duplex": true}}
```

That mode is not enabled just because the model name is MiniCPM-o 4.5. The
current MiniCPM-o 4.5 native audio-append path is scheduler data-plane based:
open/close/signal remain control-plane events, while audio appends are submitted
to a stable per-session/epoch stage request with runner-owned
`model_intermediate_buffer` payload and `resumable=True` streaming updates.
Stage0 output is forwarded through the existing stage pipeline instead of
placing hidden states in a worker-control RPC payload. This gives the current
prototype long-lived stage request IDs for the active epoch, but it is still not
a core persistent KV lease. Stage0 may only run when the loaded vLLM model
runner exposes a scheduler/KV-backed
`duplex_forward_with_runner_context` hook, but that is still a model-runner
forward boundary rather than the RFC's long-lived request lifecycle. Internal
legacy worker-control fallback diagnostics are not emitted as public API
fields by default.

The worker path wraps the already loaded MiniCPM-o stage model. It must not load
a second full `AutoModel.from_pretrained(...).to("cuda")` copy for native duplex.
The Stage0 LLM path requires a scheduler-backed
`duplex_forward_with_runner_context` hook before it can claim
`runner_kv_backed=true`; the hook must return that metadata explicitly. There
is no single-sequence eager decoder fallback in the native Stage0 path; if the
runner hook is absent, the stage reports an explicit error instead of pretending
to be KV-backed.

The `/v1/realtime?duplex=1` endpoint uses the same `DuplexSessionActor` runtime
and translates the main OpenAI-style realtime client events used by MiniCPM-o
4.5 live streaming:
`session.update`, `conversation.item.create`, `input_audio_buffer.append`,
`input_audio_buffer.commit`, `input_audio_buffer.clear`, `response.create`,
`response.cancel`, `output_audio_buffer.clear`, and `session.close`. `pcm16`
input is converted at the serving boundary to the model-native `pcm_f32le`
append payload. Silent or low-RMS PCM chunks do not start speech and are routed
to listen/no-response behavior instead of scheduling a model response. Server
events include the current Realtime-style `conversation.item.added`,
`conversation.item.created`, `conversation.item.done`,
`conversation.item.deleted`, `conversation.item.truncated`,
`input_audio_buffer.speech_started/stopped`,
`input_audio_buffer.committed`, `response.created`,
`response.output_item.added/done`, `response.content_part.added/done`,
`response.output_audio.delta/done`, `response.audio.delta/done`,
`response.output_audio_transcript.delta/done`, `response.done`, and
`rate_limits.updated`. Legacy vLLM-Omni aliases such as
`response.output_item.created`, `response.audio_transcript.*`, and
`response.text.*` are emitted only when `legacy_audio_events=true` or
`vllm_omni_legacy_events=true` is set.
Connections that pass `model=` in the Realtime URL open a default session
immediately; `session.update` can then patch that session without acting as the
first open event. Output format negotiation accepts both flat string formats
and nested `audio.input/output.format` objects for PCM/WAV.
`session.update` preserves common Realtime fields including `tools`,
`tool_choice`, `metadata`, `turn_detection`, `input_audio_transcription`,
`audio`, and `tracing`, and unknown JSON-safe session fields are reflected back
in `session.updated`. It is still not a byte-perfect OpenAI Realtime
compatibility layer for every event field.

If a response contains `runtime_control.unsupported_count > 0` or a
`runtime.control` event with `unsupported_count > 0`, one or more stage-native
duplex hooks returned their no-op fallback. Treat it as capability information,
not as a failed chat/TTS response. Playback cursor now gates assistant text
history commit using model/audio delta text-duration marks when available, and
Realtime `conversation.item.truncate` uses those same marks before falling back
to proportional truncation for models that do not expose alignment metadata.
Native persistent KV lease remains future work by design.

In multi-replica deployments, duplex session admission is bound to one replica
per stage for the session/epoch. A failed replica is evicted from new admission,
its bound request/session state is cleaned up, and healthy replicas in the same
stage remain available. If the final healthy replica in a stage dies, the
orchestrator still fails the stage because there is no remaining admission
target.

## 2. Launch the Gradio demo

```bash
bash examples/online_serving/minicpmo/run_gradio_demo.sh

# Or run the python entry point directly:
python examples/online_serving/minicpmo/gradio_demo.py \
    --minicpmo45-api-base http://localhost:8099/v1 \
    --minicpmo45-model openbmb/MiniCPM-o-4_5 \
    --port 7862
```

Open `http://<host>:7862` in a browser.

## 3. Run the Realtime duplex scenario demo

After the server is running, use the scenario client to validate the native
duplex semantics end to end:

```bash
python examples/online_serving/minicpmo/realtime_duplex_demo.py \
    --url ws://localhost:8099/v1/realtime?duplex=1 \
    --model openbmb/MiniCPM-o-4_5 \
    --input-wav /path/to/input_16k_mono_pcm16.wav \
    --output-dir /tmp/minicpmo_realtime_duplex_demo
```

The script streams sequential clean speech turns and relies on `auto_response`;
it never sends `response.create` or a serving-side barge-in signal. Pass a
different `--turn-input-wav` for each later turn and use
`--require-distinct-inputs` to reject repeated audio content and cross-turn
transcript tail reuse. It fails on incomplete response/audio lifecycle events,
stale output, cancellation, transcript delta/done mismatch, or missing audio
when `--require-audio` is set.

## Notes

- **TTS trigger**: the demo sets
  `extra_body.chat_template_kwargs.use_tts_template=True`, which appends
  `<|tts_bos|>` to the assistant prefix.
- Uncheck **"Generate speech output (TTS)"** to get text-only responses
  (faster).
- The audio output is the raw WAV returned by the stage-1 talker +
  Token2Wav; sample rate is 24 kHz.
- Video input is forwarded as a base64 `video_url` entry; the server needs
  decord/torchvision to decode it.
- `MINICPMO45_PROFILE_LOGS=1` logs `llm2tts`, Talker decode, and Token2wav
  timings for profiling.
