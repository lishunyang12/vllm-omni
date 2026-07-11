# Experimental Full-Duplex Runtime

This package contains the experimental session-oriented runtime used by the
MiniCPM-o 4.5 native duplex path and the JoyVL example integration.

The current architecture, lifecycle invariants, MiniCPM Stage0/Stage1 data
flow, deploy configuration, validation evidence, and reviewer reproduction
steps are documented in:

[`docs/design/minicpmo45_full_duplex_runtime_review.md`](../../../docs/design/minicpmo45_full_duplex_runtime_review.md)

## Package boundaries

```text
core/       typed identity, events, reducer, runtime, ports, playback cursor
engine/     current vLLM-Omni scheduler/orchestrator adapter
openai/     Realtime protocol projection and WebSocket transport
minicpmo45/ MiniCPM-o model policy and Stage0/Stage1 runtime adapters
joyvl/      JoyVL model-specific implementation
```

`core` is model-agnostic. Model token IDs, input framing, and stage state belong
in the model package. Scheduler request details belong in `engine`. OpenAI event
names and audio codecs belong in `openai`.

## Scope

The verified MiniCPM-o checkpoint supports model-owned listen/speak,
auto-response, and clean multi-turn native audio streaming. Automatic/VAD
barge-in, scheduler-native append, bounded long-session KV, and production
multi-session concurrency remain follow-up work.
