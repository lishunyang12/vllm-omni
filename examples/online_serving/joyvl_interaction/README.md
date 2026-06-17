# JoyVL Streaming Interaction

A proactive, vision-first streaming assistant: at every tick the model decides on
its own whether to **speak**, stay **silent**, or **delegate** a hard task to a
background brain. The model ([JoyAI-VL-Interaction](https://github.com/jd-opensource/JoyAI-VL-Interaction),
a Qwen3-VL derivative) makes the decision; this server only orchestrates around
it — session state, three-tier memory, and delegation.

## Architecture

```
client (video @ ~1fps) ─▶ interaction server (:8070, OpenAI /v1/chat/completions)
                              │  per-session state · stable-head/append-only prompt
                              │  3-tier memory · delegation bridge
                              └─▶ model server (:8061)  ← plain `vllm serve`, Qwen3-VL
                                  summarizer (optional)  ← Qwen3-VL-4B, or reuse main
```

Each `/v1/chat/completions` call is one tick: it carries the current frame(s) as
`image_url` content plus an optional query; the response is the model's spoken
text (empty on silence) and an `interaction` field with the raw action, memory
snapshot, and any delegation.

## Run

```bash
# 1. model server (plain VLM — NOT --omni)
bash start_model.sh                      # serves on :8061

# 2. interaction orchestrator
bash start_interaction.sh                # serves on :8070

# 3a. headless timeline
python run_cli_demo.py path/to/video.mp4 --query "Alert me if a fire breaks out"

# 3b. Gradio UI (video + live decisions)
python gradio_demo.py --server http://127.0.0.1:8070
```

## Notes

- **`force_silence_before_query`** (default on): the model stays silent until the
  first query arrives, so proactive monitoring is armed by an instruction such as
  "alert me if…". Pass `--no-force-silence` to run from the first frame.
- **Memory**: by default the orchestrator reuses the main model as its own
  summarizer. Point `SUMMARIZER_BACKEND_URL`/`SUMMARIZER_MODEL` at a dedicated
  Qwen3-VL-4B server for production, or set `NO_MEMORY=1` for the lightest setup.
  Mid-term summaries are built per evicted chunk; long-term compression runs every
  N chunks.
- **Delegation** is stubbed (`StubDelegationBridge`): the spoken note surfaces
  immediately and a placeholder digest is folded back into the model's context a
  couple of ticks later. Implement `DelegationBridge` to wire a real agent/API.
- **Speech** (ASR/TTS) is intentionally external and pluggable.

## Voice (TTS) with the JD webui

To give the live webui spoken output, serve Qwen3-TTS and run the bridge that
translates the webui's TTS WebSocket protocol to vLLM-Omni's `/v1/audio/speech/
stream`:

```bash
# serves Qwen3-TTS (:8091) + the bridge (:8092)
GPU=1 bash start_tts.sh

# then launch the webui pointed at the bridge
TTS_URL=ws://127.0.0.1:8092/v1/tts bash scripts/start_server.sh   # (in the webui repo)
```

`tts_bridge.py` is standalone (`aiohttp`), so it can also front any other
`/v1/audio/speech/stream` backend.

## Voice input (ASR) with the JD webui

Symmetric to TTS. `asr_bridge.py` translates the webui's `ASR_URL` protocol
(binary `>iii`+PCM16 frames in, `IS_PARTIAL`/`IS_FINAL` JSON out) to an
OpenAI-compatible `/v1/audio/transcriptions` backend:

```bash
python asr_bridge.py --backend-url http://<asr-host>:<port> --model qwen3-asr   # :8093
ASR_URL=ws://127.0.0.1:8093/v1/asr bash scripts/start_server.sh                 # (in the webui repo)
```

Note: vLLM-Omni does not yet ship a standalone Qwen3-ASR server (ASR lives inside
Qwen3-Omni / the realtime transcription path), so the bridge needs a transcription
endpoint pointed at it. Voice input is optional — typed queries work without ASR.
