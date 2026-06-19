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

## Host a demo (Gradio)

A self-contained browser demo ships in-repo — no external webui needed. It talks to the
orchestrator over its HTTP API only:

```bash
pip install vllm-omni[demo]          # gradio + opencv-python + requests
python examples/online_serving/joyvl_interaction/app.py --server http://127.0.0.1:8070
```

Open the printed URL, upload a clip (or record from webcam), optionally give a standing
instruction (e.g. "Alert me if a fire breaks out"), and the per-tick speak / silence /
delegate decisions stream into a timeline. Add `--share` for a temporary public link.

## Verification

```bash
# headless: stream a clip and print the per-tick decision timeline
# (the CLI reads video frames via OpenCV: `pip install opencv-python`)
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
- `force_silence_before_query` is on by default — the model stays silent until an
  instruction arrives; give a standing task (e.g. "translate the on-screen text")
  to arm proactive output.
- Speech is external and pluggable: point `ASR_URL` / `TTS_URL` at the bridges in
  `examples/online_serving/joyvl_interaction/bridges/` or any compatible service.
- The decision prompts, sampling, and 3-tier summary memory (`T_s=100`, mid→long every
  5 chunks, `key_frames=0` = summarize all chunk frames) are aligned to the JoyVL
  reference adapter so per-tick behavior matches the released model; the framework only
  supplies the serving structure.
