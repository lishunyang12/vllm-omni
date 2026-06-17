# Full-duplex runtime

A model-agnostic control plane for full-duplex realtime sessions: input keeps
arriving while the assistant produces output, and a barge-in interrupts in-flight
output. The lifecycle lives here once; pipelines plug in via one seam.

```
client ── realtime events ──▶ transport (/v1/duplex, /v1/realtime?duplex=1)
                                  │  decoded events
                                  ▼
                            DuplexRuntime ──▶ DuplexSession   (state, epoch, barge-in)
                                  │
                                  ▼
                            DuplexAdapter   ◀── the only seam
                              ├── adapters/joyvl.py        proactive video + text out (external speech)
                              └── adapters/minicpmo45.py   fused audio: listen/speak + native TTS  (future)
```

- **`session.py`** — `DuplexSession`: id / response index / **epoch** / playback cursor; `begin_response` / `barge_in` / `is_stale`.
- **`runtime.py`** — `DuplexRuntime`: the event loop. Output produced under a stale epoch (after a barge-in) is dropped, so long responses stay interruptible.
- **`adapter.py`** — `DuplexAdapter`: `capabilities / on_input / should_respond / respond / on_barge_in / on_playback_ack`. Adapters own *model policy only*.
- **`protocol.py`** — the realtime event vocabulary transports map to/from.

The data planes differ by adapter and are **not** shared: a fused audio model
(MiniCPM-o) runs its native listen/speak + TTS handoff; JoyVL reuses
`InteractionBrain` and keeps speech external (ASR/TTS bridges). Only the
session/lifecycle/event control plane is common.
