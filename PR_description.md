# [Feature][TTS] Streaming Text Input for Qwen3-TTS via WebSocket

## Summary

Add a WebSocket endpoint `/v1/audio/speech/stream` that accepts text input incrementally (e.g., from a real-time STT pipeline), buffers and splits at sentence boundaries, and generates audio per sentence using the existing TTS pipeline.

This enables real-time text-to-speech workflows where text is produced progressively (speech-to-text, LLM token streaming, live captions) and audio needs to be generated as soon as complete sentences are available, rather than waiting for the entire input.

**Scope:** Streaming text *input* only. Each sentence produces a complete audio response. Streaming audio *output* (chunked PCM) is tracked separately in PR #1189.

## Motivation

The current `/v1/audio/speech` REST endpoint requires the full text upfront. In real-time pipelines (e.g., STT → LLM → TTS), text arrives incrementally. Without streaming input support, clients must either:
1. Wait for the entire text before calling TTS (high latency), or
2. Manually implement sentence buffering and make multiple REST calls (complex, no session state).

This PR solves both issues with a single WebSocket session that handles buffering, sentence detection, and per-sentence generation automatically.

## WebSocket Protocol

**Transport:** WebSocket (industry standard — used by OpenAI Realtime API, ElevenLabs, Azure TTS)

### Client → Server

```jsonc
// 1. Session config (sent once, first message)
{"type": "session.config", "voice": "Vivian", "task_type": "CustomVoice", "language": "Auto"}

// 2. Text chunks (sent incrementally, any number of times)
{"type": "input.text", "text": "Hello, how are you? "}

// 3. End of input (flushes remaining buffer)
{"type": "input.done"}
```

### Server → Client

```jsonc
// Per-sentence: metadata → binary audio → completion
{"type": "audio.start", "sentence_index": 0, "sentence_text": "Hello, how are you?", "format": "wav"}
<binary WebSocket frame: audio bytes>
{"type": "audio.done", "sentence_index": 0}

// Session complete
{"type": "session.done", "total_sentences": 3}

// Non-fatal error (session continues)
{"type": "error", "message": "..."}
```

## Changes

### New Files

| File | Description |
|------|-------------|
| `vllm_omni/entrypoints/openai/text_splitter.py` | `SentenceSplitter` — incremental sentence boundary detector. Regex-based splitting at English `.!?` + whitespace and CJK fullwidth `。！？，；`. Configurable `min_sentence_length` (default 2, CJK-friendly). |
| `vllm_omni/entrypoints/openai/serving_speech_stream.py` | `OmniStreamingSpeechHandler` — WebSocket session handler. Manages config validation, idle/config timeouts (30s/10s), per-sentence audio generation, and error resilience (one sentence failure doesn't kill the session). |
| `examples/online_serving/qwen3_tts/streaming_speech_client.py` | Python WebSocket client example. Supports `--simulate-stt` mode (word-by-word with configurable delay), all 3 task types (CustomVoice, VoiceDesign, Base), saves per-sentence audio files. |
| `tests/entrypoints/openai_api/test_text_splitter.py` | Unit tests for `SentenceSplitter`: English/Chinese/mixed splitting, incremental accumulation, flush behavior, edge cases. |
| `tests/entrypoints/openai_api/test_serving_speech_stream.py` | WebSocket integration tests: session lifecycle, multi-sentence, incremental text, flush-on-done, empty input, invalid config, invalid JSON, unknown message types, generation failure recovery. |

### Modified Files

| File | Description |
|------|-------------|
| `vllm_omni/entrypoints/openai/serving_speech.py` | **Refactor:** Extracted `_generate_audio_bytes(request) → (bytes, media_type)` from `create_speech()`. The REST endpoint delegates to it; the WebSocket handler reuses it per sentence. No behavior change for existing callers. |
| `vllm_omni/entrypoints/openai/protocol/audio.py` | Added `StreamingSpeechSessionConfig` Pydantic model for WebSocket session configuration (mirrors `OpenAICreateSpeechRequest` fields minus `input`). |
| `vllm_omni/entrypoints/openai/api_server.py` | Added `@router.websocket("/v1/audio/speech/stream")` route and `OmniStreamingSpeechHandler` initialization in `omni_init_app_state()`. |
| `examples/online_serving/qwen3_tts/README.md` | Added streaming text input documentation section with protocol spec, parameters, and usage examples. |

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **WebSocket** (not SSE/HTTP chunked) | Bidirectional: client sends text incrementally, server sends audio back. Industry standard for real-time TTS (OpenAI, ElevenLabs, Azure). |
| **Sentence-level chunking** (not token/word) | Natural speech boundaries produce coherent audio. Avoids artifacts from splitting mid-sentence. Production-ready granularity. |
| **`min_sentence_length=2`** | Prevents splitting on lone punctuation (`.`) while supporting short CJK sentences like `你好！` (3 chars). |
| **`_generate_audio_bytes()` extraction** | Clean separation of concerns. REST endpoint wraps in `Response`; WebSocket sends raw bytes. No code duplication. |
| **Per-sentence error resilience** | If generation fails for one sentence, an error is sent but the session continues for remaining sentences. |
| **Idle + config timeouts** | Prevents resource leaks from abandoned connections (30s idle, 10s for initial config). |

## Test Plan

- [ ] `pytest tests/entrypoints/openai_api/test_text_splitter.py` — sentence splitter unit tests
- [ ] `pytest tests/entrypoints/openai_api/test_serving_speech_stream.py` — WebSocket integration tests
- [ ] `pytest tests/entrypoints/openai_api/test_serving_speech.py` — existing REST endpoint tests (verify refactor is non-breaking)
- [ ] Manual test with live server:
  ```bash
  # Start server
  ./examples/online_serving/qwen3_tts/run_server.sh CustomVoice

  # Run streaming client
  python examples/online_serving/qwen3_tts/streaming_speech_client.py \
      --text "Hello world. How are you? I am fine." \
      --simulate-stt
  ```
