# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Self-contained Gradio demo for the JoyVL streaming interaction orchestrator.

Host it with:

    pip install vllm-omni[demo]            # gradio; plus opencv-python + requests
    # start the model + orchestrator first (see recipes/JD/JoyAI-VL-Interaction.md)
    python examples/online_serving/joyvl_interaction/app.py --server http://127.0.0.1:8070

Then open the printed URL, upload a clip (or record from webcam), optionally give a
standing instruction, and watch the per-tick speak / silence / delegate decisions stream
in. This talks to the orchestrator over its normal HTTP API only — no model code here.
"""

from __future__ import annotations

import argparse
import os
import sys

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cli"))
from stream_client import stream  # noqa: E402

ICON = {"response": "🗣 speak", "delegate": "🤝 delegate", "silence": "… silence"}


def run(video: str, query: str, server: str, fps: float, session_id: str):
    if not video:
        yield "Upload or record a video first."
        return
    header = "| t (s) | action | text |\n|---|---|---|\n"
    rows: list[str] = []
    for tick in stream(video, server, session_id or "gradio", query or None, float(fps)):
        if tick.action == "silence":
            continue
        text = (tick.text or "").replace("\n", " ").strip()
        if tick.delegation and tick.delegation.get("question"):
            text += f" ↪ delegated: {tick.delegation['question']}"
        rows.append(f"| {tick.t:.1f} | {ICON.get(tick.action, tick.action)} | {text} |")
        yield header + "\n".join(rows)
    if not rows:
        yield "_(the model stayed silent for the whole clip)_"


def build(server_default: str) -> gr.Blocks:
    with gr.Blocks(title="JoyAI-VL-Interaction demo") as demo:
        gr.Markdown(
            "# JoyAI-VL-Interaction — streaming demo\n"
            "Upload a clip and (optionally) give a standing instruction. The model decides "
            "every second whether to **speak**, stay **silent**, or **delegate** a hard question."
        )
        with gr.Row():
            video = gr.Video(label="Video", sources=["upload", "webcam"])
            with gr.Column():
                query = gr.Textbox(label="Instruction (optional)", placeholder="Alert me if a fire breaks out")
                server = gr.Textbox(label="Orchestrator URL", value=server_default)
                fps = gr.Slider(0.5, 4.0, value=1.0, step=0.5, label="Sample FPS")
                session_id = gr.Textbox(label="Session ID", value="gradio")
                go = gr.Button("Run", variant="primary")
        out = gr.Markdown(label="Decision timeline")
        go.click(run, [video, query, server, fps, session_id], out)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8070", help="orchestrator base URL")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="expose a temporary public Gradio link")
    args = ap.parse_args()
    build(args.server).queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
