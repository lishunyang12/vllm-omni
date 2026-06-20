# JoyAI-VL-Interaction

> Real-time streaming video-language interaction (proactive speak / silence / delegate)

## Summary

- Vendor: JD (Joy Future Academy)
- Model: 8B vanilla Qwen3-VL interaction weights (checkpoint pending public release)
- Task: Per-tick proactive interaction over a live video stream — the model decides
  on its own each second to speak, stay silent, or delegate a hard question
- Mode: Online serving — an OpenAI-compatible interaction orchestrator in front of
  a plain `vllm serve` backend
- Maintainer: Community

## When to use this recipe

Use this to stand up the streaming-interaction serving layer (`vllm_omni/experimental/fullduplex/`).
The model is served unchanged by `vllm serve`; this layer adds session state, 3-tier
summary memory, the per-tick decision, and pluggable ASR / TTS / delegation. For the
framework internals and how to add another full-duplex model, see
[`vllm_omni/experimental/fullduplex/README.md`](../../vllm_omni/experimental/fullduplex/README.md).

## Environment

- OS: Linux
- Python: 3.10+
- Hardware: 1x GPU. The default `T_s=100` frame window + `--max-model-len 131072` wants
  ≈48GB+; for ~24GB, lower `--chunk-frames`, `--max-model-len`, and the image limit together
- vLLM / vLLM-Omni: versions from your current checkout

## Start server

From repository root:

```bash
# 1. Serve the model (plain `vllm serve`, NOT --omni; it is vanilla Qwen3-VL).
#    The image limit must cover the short-term frame window (chunk_frames, default 100 = T_s);
#    prefix caching keeps the accumulating window cheap. Lower both for smaller GPUs.
vllm serve <model-path> \
  --served-model-name <model-name> --port 8061 \
  --max-model-len 131072 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":256,"video":1}'

# 2. Interaction orchestrator (OpenAI-compatible, :8070)
python -m vllm_omni.experimental.fullduplex.joyvl.serving.server --port 8070 \
  --main-backend-url http://127.0.0.1:8061/v1 --main-model <model-name>
```

Optional one-shot launch (model + orchestrator + JD webui + ASR/TTS/background,
all env-configurable). The JD webui frontend is external — set `WEBUI_DIR` to point at it:

```bash
bash examples/online_serving/joyvl_interaction/scripts/start_all.sh
```

## Delegation (background brain)

When the model judges a question too hard, it emits `</delegation> <question>` and the
orchestrator hands it to a **background brain** — any OpenAI-compatible endpoint you
self-host. Enable it by pointing the orchestrator at one:

```bash
python -m vllm_omni.experimental.fullduplex.joyvl.serving.server --port 8070 \
  --main-backend-url http://127.0.0.1:8061/v1 --main-model <model-name> \
  --delegation-backend-url <brain-endpoint>/v1 \
  --delegation-model <brain-model> --delegation-kind chat
```

`--delegation-kind` picks the bridge:

- `chat` — a stronger text/VL model answers (`/chat/completions`)
- `image` — a text-to-image model generates a picture (`/images/generations`, e.g. Qwen-Image)
- `edit` — an image-edit model restyles the current frame (e.g. Qwen-Image-Edit)
- `router` — dispatch each request to chat / image / edit by inspecting it (set
  `--delegation-image-url` / `--delegation-edit-url` for the latter two)

The brain is **bring-your-own**: a larger vLLM you serve, or any OpenAI-compatible API
(e.g. `--delegation-backend-url https://api.anthropic.com/v1/ --delegation-model claude-...
--delegation-api-key …`). The reference deployment instead drives the `codex` CLI as the
brain via a separate background-agent service; that agent runs with its own credentials
and bypasses its sandbox, so it is **not bundled here** — self-host a plain
OpenAI-compatible endpoint instead. Omit `--delegation-backend-url` to keep delegation off.

## Host a demo (Gradio)

A self-contained browser demo ships in-repo — no external webui needed. It talks to the
orchestrator over its HTTP API only:

```bash
uv pip install vllm-omni[demo]       # gradio + opencv-python + requests
python examples/online_serving/joyvl_interaction/app.py --server http://127.0.0.1:8070
```

Open the printed URL, upload a clip (or record from webcam), optionally give a standing
instruction (e.g. "Alert me if a fire breaks out"), and the per-tick speak / silence /
delegate decisions stream into a timeline. Add `--share` for a temporary public link.

## Verification

```bash
# headless: stream a clip and print the per-tick decision timeline
# (the CLI reads video frames via OpenCV: `uv pip install opencv-python`)
python examples/online_serving/joyvl_interaction/cli/run_cli_demo.py \
  path/to/video.mp4 --query "Alert me if a fire breaks out"

pytest tests/fullduplex   # framework + JoyVL unit tests
```

## Testing with an RTSP stream (optional)

RTSP is a **webui-side input** — the browser pulls the stream and feeds frames to the
orchestrator over the normal API; no serving-layer code is involved. To simulate an RTSP
camera from a local video file (no physical IP camera needed), use the helper scripts in
[`examples/online_serving/joyvl_interaction/rtsp/`](../../examples/online_serving/joyvl_interaction/rtsp/),
which wrap [MediaMTX](https://github.com/bluenviron/mediamtx/releases) + `ffmpeg`:

```bash
cd examples/online_serving/joyvl_interaction/rtsp

# 1. Local RTSP server (MediaMTX, listens on :8554)
bash ./mediamtx.sh

# 2. Push a local video file as an RTSP stream (another terminal)
bash ./rtsp.sh ./videos/example.mp4 rtsp://127.0.0.1:8554/fire1

# 3. In the WebUI RTSP box, enter:  rtsp://127.0.0.1:8554/fire1
#    (replace 127.0.0.1 with the MediaMTX host IP if the webui runs on another machine)
```

See the directory's `README.md` for streaming a whole video folder (`rtsp_all.sh`) and
the audio-track caveat.

## Notes

- `--omni` is **not** used: the model is standard Qwen3-VL, so stock `vllm serve`
  runs the forward pass; this recipe only adds the interaction/serving layer.
- On a host without `nvcc` / `ninja`, `vllm serve` of the 8B can crash engine-core in the
  FlashInfer sampler JIT (`FileNotFoundError: 'ninja'`) during `profile_run`. Set
  `VLLM_USE_FLASHINFER_SAMPLER=0` (or install `ninja`) to work around it.
- `force_silence_before_query` is on by default — the model stays silent until an
  instruction arrives; give a standing task (e.g. "translate the on-screen text")
  to arm proactive output.
- Speech is external and pluggable: point `ASR_URL` / `TTS_URL` at the bridges in
  `examples/online_serving/joyvl_interaction/bridges/` or any compatible service.
- The decision prompts, sampling, and 3-tier summary memory (`T_s=100`, mid→long every
  5 chunks, `key_frames=0` = summarize all chunk frames) are aligned to the JoyVL
  reference adapter so per-tick behavior matches the released model; the framework only
  supplies the serving structure.
