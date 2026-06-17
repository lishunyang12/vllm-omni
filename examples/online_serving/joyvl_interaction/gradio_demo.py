# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gradio demo for the JoyVL interaction server.

Pick a video and an arming query (e.g. "Alert me if a fire breaks out"), hit
Run, and watch the model decide speak/silence/delegate frame by frame next to
the video. Point ``--server`` at a running interaction server (see README).

    python gradio_demo.py --server http://127.0.0.1:8070
"""

from __future__ import annotations

import argparse

import gradio as gr
from stream_client import stream

ACTION_ICON = {"response": "🗣️", "silence": "·", "delegate": "📡"}


def _row(tick) -> list[str]:
    icon = ACTION_ICON.get(tick.action, "?")
    text = tick.text
    if tick.delegation:
        text = f"{text}  ⟶ delegated: {tick.delegation.get('question', '')}"
    return [f"{tick.t:6.1f}s", icon, text or ("(silent)" if tick.action == "silence" else "")]


def run(video, query, server, fps, realtime):
    if not video:
        yield "Please choose a video.", []
        return
    rows: list[list[str]] = []
    spoke = 0
    for tick in stream(video, server, session_id="gradio", query=query or None, fps=fps, realtime=realtime):
        if tick.action != "silence":
            spoke += 1
        rows.append(_row(tick))
        status = f"tick {tick.index} · {tick.t:.1f}s · spoke {spoke} · last {tick.latency_ms:.0f} ms"
        yield status, rows[-200:]


def build(server_default: str) -> gr.Blocks:
    with gr.Blocks(title="JoyVL Interaction") as demo:
        gr.Markdown("## JoyVL-Interaction — proactive streaming demo")
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Video")
                query = gr.Textbox(label="Arming query", placeholder="Alert me if a fire breaks out")
                with gr.Row():
                    fps = gr.Slider(0.5, 4, value=1.0, step=0.5, label="fps")
                    realtime = gr.Checkbox(value=False, label="real-time pacing")
                server = gr.Textbox(value=server_default, label="Server")
                go = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                status = gr.Textbox(label="Status", interactive=False)
                table = gr.Dataframe(headers=["time", "act", "model output"], wrap=True, label="Live decisions")
        go.click(run, [video, query, server, fps, realtime], [status, table])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8070")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="expose a public gradio.live link")
    args = parser.parse_args()
    build(args.server).queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
