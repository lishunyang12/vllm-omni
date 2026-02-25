# Speech API

vLLM-Omni provides an OpenAI-compatible API for text-to-speech (TTS) generation using Qwen3-TTS models.

Each server instance runs a single model (specified at startup via `vllm serve <model> --omni`).

## Quick Start

### Start the Server

```bash
# CustomVoice model (predefined speakers)
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

### Generate Speech

**Using curl:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English"
    }' --output output.wav
```

**Using Python:**

```python
import httpx

response = httpx.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English",
    },
    timeout=300.0,
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

**Using OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.audio.speech.create(
    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    voice="vivian",
    input="Hello, how are you?",
)

response.stream_to_file("output.wav")
```

## API Reference

### Endpoint

```
POST /v1/audio/speech
Content-Type: application/json
```

### Request Parameters

#### OpenAI Standard Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | The text to synthesize into speech |
| `model` | string | server's model | Model to use (optional, should match server if specified) |
| `voice` | string | "vivian" | Speaker name (e.g., vivian, ryan, aiden) |
| `response_format` | string | "wav" | Audio format: wav, mp3, flac, pcm, aac, opus |
| `speed` | float | 1.0 | Playback speed (0.25-4.0) |

#### vLLM-Omni Extension Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | string | "CustomVoice" | TTS task type: CustomVoice, VoiceDesign, or Base |
| `language` | string | "Auto" | Language (see supported languages below) |
| `instructions` | string | "" | Voice style/emotion instructions |
| `max_new_tokens` | integer | 2048 | Maximum tokens to generate |

**Supported languages:** Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

#### Voice Clone Parameters (Base task)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ref_audio` | string | null | Reference audio (URL or base64 data URL) |
| `ref_text` | string | null | Transcript of reference audio |
| `x_vector_only_mode` | bool | null | Use speaker embedding only (no ICL) |

### Response Format

Returns binary audio data with appropriate `Content-Type` header (e.g., `audio/wav`).

### Voices Endpoint

```
GET /v1/audio/voices
```

Lists available voices for the loaded model.

```json
{
    "voices": ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
}
```

## Streaming Text Input (WebSocket)

The `/v1/audio/speech/stream` WebSocket endpoint accepts text incrementally and generates audio per sentence as boundaries are detected. This is useful for real-time pipelines where text arrives progressively (e.g., LLM token streaming, STT output).

> **Note:** This is streaming text *input* only. Each sentence produces a complete audio response.

### Quick Example

```python
import asyncio
import json
import websockets

async def stream_tts():
    async with websockets.connect("ws://localhost:8091/v1/audio/speech/stream") as ws:
        # 1. Session config
        await ws.send(json.dumps({
            "type": "session.config",
            "voice": "Vivian",
            "response_format": "wav",
        }))

        # 2. Send text incrementally
        for word in ["Hello ", "world. ", "How ", "are ", "you? "]:
            await ws.send(json.dumps({"type": "input.text", "text": word}))

        # 3. Signal end of input
        await ws.send(json.dumps({"type": "input.done"}))

        # 4. Receive audio per sentence
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                print(f"Audio: {len(msg)} bytes")
            else:
                data = json.loads(msg)
                print(data)
                if data.get("type") == "session.done":
                    break

asyncio.run(stream_tts())
```

### WebSocket Protocol

**Client → Server:**

| Message | Description |
|---------|-------------|
| `{"type": "session.config", ...}` | Session configuration (sent once, first message) |
| `{"type": "input.text", "text": "..."}` | Text chunk (sent any number of times) |
| `{"type": "input.done"}` | End of input, flushes remaining buffer |

**Server → Client:**

| Message | Description |
|---------|-------------|
| `{"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}` | Audio generation starting for a sentence |
| Binary frame | Raw audio bytes |
| `{"type": "audio.done", "sentence_index": 0}` | Audio complete for a sentence |
| `{"type": "session.done", "total_sentences": N}` | Session complete |
| `{"type": "error", "message": "..."}` | Non-fatal error (session continues) |

### Session Config Parameters

All REST API parameters are supported, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voice` | string | None | Speaker voice name |
| `task_type` | string | None | CustomVoice, VoiceDesign, or Base |
| `language` | string | None | Language code |
| `instructions` | string | None | Voice style instructions |
| `response_format` | string | "wav" | Audio format per sentence |
| `speed` | float | 1.0 | Playback speed (0.25-4.0) |
| `max_new_tokens` | int | None | Max tokens per sentence |
| `ref_audio` | string | None | Reference audio (Base task) |
| `ref_text` | string | None | Reference text (Base task) |
| `split_granularity` | string | "sentence" | Text splitting granularity |

### Split Granularity

Controls how text is split into chunks for audio generation:

| Granularity | Boundaries | Use case |
|-------------|-----------|----------|
| `"sentence"` (default) | English `.!?` + whitespace, CJK `。！？` | Best prosody, recommended for most use cases |
| `"clause"` | All of the above + CJK commas `，` and semicolons `；` | Lower latency, more frequent but shorter audio chunks |

Text without a sentence boundary is buffered until `input.done` triggers a flush.

## Examples

### CustomVoice with Style Instruction

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "I am so excited!",
        "voice": "vivian",
        "instructions": "Speak with great enthusiasm"
    }' --output excited.wav
```

### VoiceDesign (Natural Language Voice Description)

```bash
# Start server with VoiceDesign model first
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello world",
        "task_type": "VoiceDesign",
        "instructions": "A warm, friendly female voice with a gentle tone"
    }' --output designed.wav
```

### Base (Voice Cloning)

```bash
# Start server with Base model first
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is a cloned voice",
        "task_type": "Base",
        "ref_audio": "https://example.com/reference.wav",
        "ref_text": "Original transcript of the reference audio"
    }' --output cloned.wav
```

## Supported Models

| Model | Task Type | Description |
|-------|-----------|-------------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | CustomVoice | Predefined speaker voices with optional style control |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | VoiceDesign | Natural language voice style description |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base | Voice cloning from reference audio |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | CustomVoice | Smaller/faster variant |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Base | Smaller/faster variant for voice cloning |

## Error Responses

### 400 Bad Request

Invalid parameters:

```json
{
    "error": {
        "message": "Input text cannot be empty",
        "type": "BadRequestError",
        "param": null,
        "code": 400
    }
}
```

### 404 Not Found

Model not found:

```json
{
    "error": {
        "message": "The model `xxx` does not exist.",
        "type": "NotFoundError",
        "param": "model",
        "code": 404
    }
}
```

## Troubleshooting

### "TTS model did not produce audio output"

Ensure you're using the correct model variant for your task type:
- CustomVoice task → CustomVoice model
- VoiceDesign task → VoiceDesign model
- Base task → Base model

### Server Not Running

```bash
# Check if server is responding
curl http://localhost:8091/v1/audio/voices
```

### Out of Memory

If you encounter OOM errors:
1. Use smaller model variant: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
2. Reduce `--gpu-memory-utilization`

### Unsupported Speaker

Use `/v1/audio/voices` to list available voices for the loaded model.

## Development

Enable debug logging:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager \
    --uvicorn-log-level debug
```
