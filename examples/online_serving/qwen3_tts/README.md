# Qwen3-TTS

This directory contains examples for running Qwen3-TTS models with vLLM-Omni's online serving API.

## Supported Models

| Model | Task Type | Description |
|-------|-----------|-------------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | CustomVoice | Predefined speaker voices with optional style control |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | VoiceDesign | Natural language voice style description |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base | Voice cloning from reference audio |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | CustomVoice | Smaller/faster variant |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Base | Smaller/faster variant for voice cloning |

## Quick Start

### 1. Start the Server

```bash
# CustomVoice model (default)
./run_server.sh

# Or specify task type
./run_server.sh CustomVoice
./run_server.sh VoiceDesign
./run_server.sh Base
```

Or launch directly with vllm serve:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni --port 8091 --trust-remote-code --enforce-eager
```

### 2. Run the Client

```bash
# CustomVoice: Use predefined speaker
python openai_speech_client.py \
    --text "你好，我是通义千问" \
    --voice vivian \
    --language Chinese

# CustomVoice with style instruction
python openai_speech_client.py \
    --text "今天天气真好" \
    --voice ryan \
    --instructions "用开心的语气说"

# VoiceDesign: Describe the voice style
python openai_speech_client.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --task-type VoiceDesign \
    --text "哥哥，你回来啦" \
    --instructions "体现撒娇稚嫩的萝莉女声，音调偏高"

# Base: Voice cloning
python openai_speech_client.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --task-type Base \
    --text "Hello, this is a cloned voice" \
    --ref-audio /path/to/reference.wav \
    --ref-text "Original transcript of the reference audio"
```

### 3. Using curl

```bash
# Simple TTS request
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English"
    }' --output output.wav

# With style instruction
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "I am so excited!",
        "voice": "vivian",
        "instructions": "Speak with great enthusiasm"
    }' --output excited.wav

# List available voices in CustomVoice models
curl http://localhost:8091/v1/audio/voices
```

## API Reference

### Endpoint

```
POST /v1/audio/speech
```

This endpoint follows the [OpenAI Audio Speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech) format with additional Qwen3-TTS parameters.

### Voices Endpoint

```
GET /v1/audio/voices
```

Lists available voices for the loaded model:

```json
{
    "voices": ["aiden", "dylan", "eric", "one_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
}
```

### Request Body

```json
{
    "input": "Text to synthesize",
    "voice": "vivian",
    "response_format": "wav",
    "task_type": "CustomVoice",
    "language": "Auto",
    "instructions": "Optional style instructions",
    "ref_audio": "URL or base64 for voice cloning",
    "ref_text": "Reference audio transcript",
    "x_vector_only_mode": false,
    "max_new_tokens": 2048
}
```

> **Note:** The `model` field is optional when serving a single model, as the server already knows which model is loaded.

### Response

Returns audio data in the requested format (default: WAV).

## Parameters

### OpenAI Standard Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | Text to synthesize |
| `model` | string | server's model | Model to use (optional, should match server if specified) |
| `voice` | string | "vivian" | Speaker name (e.g., vivian, ryan, aiden) |
| `response_format` | string | "wav" | Audio format: wav, mp3, flac, pcm, aac, opus |
| `speed` | float | 1.0 | Playback speed (0.25-4.0) |

### vLLM-Omni Extension Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | string | "CustomVoice" | Task: CustomVoice, VoiceDesign, or Base |
| `language` | string | "Auto" | Language (see supported languages below) |
| `instructions` | string | "" | Voice style/emotion instructions |
| `max_new_tokens` | int | 2048 | Maximum tokens to generate |

**Supported languages:** Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

### Voice Clone Parameters (Base task)

| Parameter | Type | Default | Description |
|-----------|------|----------|-------------|
| `ref_audio` | string | null | Reference audio (URL or base64 data URL) |
| `ref_text` | string | null | Transcript of reference audio |
| `x_vector_only_mode` | bool | null | Use speaker embedding only (no ICL) |

## Python Usage

### Using OpenAI SDK

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

### Using httpx

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

## Streaming Text Input

The `/v1/audio/speech/stream` WebSocket endpoint accepts text incrementally (e.g., from a real-time STT pipeline), buffers and splits at sentence boundaries, and generates audio per sentence.

> **Note:** This is streaming *text input* only. Each sentence produces a complete audio response. For streaming audio output, see PR #1189.

### Quick Start

```bash
# Send full text (sentences are auto-detected)
python streaming_speech_client.py \
    --text "Hello world. How are you? I am fine."

# Simulate STT: send text word-by-word
python streaming_speech_client.py \
    --text "Hello world. How are you? I am fine." \
    --simulate-stt --stt-delay 0.1

# VoiceDesign task
python streaming_speech_client.py \
    --text "Today is a great day. The weather is nice." \
    --task-type VoiceDesign \
    --instructions "A cheerful young female voice"
```

### WebSocket Protocol

**Client -> Server:**

```jsonc
// 1. Session config (sent once first)
{"type": "session.config", "voice": "Vivian", "task_type": "CustomVoice", "language": "Auto", "split_granularity": "sentence"}

// 2. Text chunks (sent incrementally)
{"type": "input.text", "text": "Hello, how are you? "}

// 3. End of input (flushes remaining buffer)
{"type": "input.done"}
```

**Server -> Client:**

```jsonc
// Audio metadata (before binary frame)
{"type": "audio.start", "sentence_index": 0, "sentence_text": "Hello, how are you?", "format": "wav"}

// Binary WebSocket frame: raw audio bytes

// Per-sentence completion
{"type": "audio.done", "sentence_index": 0}

// Session complete
{"type": "session.done", "total_sentences": 3}

// Error (non-fatal, session continues)
{"type": "error", "message": "..."}
```

### Session Config Parameters

All parameters from the REST API are supported:

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
| `split_granularity` | string | "sentence" | Text splitting granularity (see below) |

### Sentence Detection

Text is automatically split at boundaries controlled by `split_granularity`:

| Granularity | Boundaries | Use case |
|-------------|-----------|----------|
| `"sentence"` (default) | English `.!?` + whitespace, CJK `。！？` | Best prosody, recommended for most use cases |
| `"clause"` | All of the above + CJK commas `，` and semicolons `；` | Lower latency, more frequent but shorter audio chunks |

If text never forms a complete sentence, it is flushed when `input.done` is sent.

## Limitations

- **Single request**: Batch processing is not yet optimized for online serving.

## Troubleshooting

1. **"TTS model did not produce audio output"**: Ensure you're using the correct model variant for your task type (CustomVoice task → CustomVoice model, etc.)
2. **Connection refused**: Make sure the server is running on the correct port
3. **Out of memory**: Use smaller model variant (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`) or reduce `--gpu-memory-utilization`
4. **Unsupported speaker**: Use `/v1/audio/voices` to list available voices for the loaded model
5. **Voice clone fails**: Ensure you're using the Base model variant for voice cloning
