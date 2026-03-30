"""Real-time video streaming client with on-screen text overlay.

Continuously streams webcam frames to vLLM-OMNI and displays the
model's response as a text overlay on the live webcam feed.

Usage:
    python realtime_video_client.py --host localhost --port 8091
    python realtime_video_client.py --host 154.54.102.35 --port 14695

Requires:
    pip install websockets pillow opencv-python numpy
"""

import argparse
import asyncio
import base64
import io
import json
import textwrap
import threading
import time

import cv2
import numpy as np


DOWNSCALE_WIDTH = 480


def downscale(frame_rgb):
    """Downscale frame to DOWNSCALE_WIDTH, preserving aspect ratio."""
    h, w = frame_rgb.shape[:2]
    if w <= DOWNSCALE_WIDTH:
        return frame_rgb
    scale = DOWNSCALE_WIDTH / w
    new_h = int(h * scale)
    return cv2.resize(frame_rgb, (DOWNSCALE_WIDTH, new_h),
                      interpolation=cv2.INTER_AREA)


def encode_frame(frame_rgb, quality: int = 70) -> str:
    from PIL import Image
    frame_rgb = downscale(frame_rgb)
    img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def frame_similarity(a, b) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm else 0.0


def draw_overlay(frame, text, max_width=70):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    lines = []
    for p in text.split("\n"):
        lines.extend(textwrap.wrap(p, width=max_width) or [""])
    line_h = 20
    max_lines = min(len(lines), 8)
    lines = lines[-max_lines:]
    if not lines:
        return frame
    box_h = len(lines) * line_h + 16
    cv2.rectangle(overlay, (0, h - box_h), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    for i, line in enumerate(lines):
        y = h - box_h + 14 + i * line_h
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# Shared state between threads
state = {
    "text": "Waiting for first response...",
    "streaming": "",
    "thinking": False,
    "frames_sent": 0,
    "frames_skipped": 0,
    "frames_since_query": 0,  # new frames since last query
    "scene_changed": False,
    "running": True,
    "frame_queue": [],
    "lock": threading.Lock(),
}


def webcam_loop(args):
    """Main thread: capture webcam, display overlay, queue frames to send."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        state["running"] = False
        return

    send_interval = 1.0 / args.fps
    last_send = 0
    last_frame_rgb = None

    while state["running"]:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        now = time.monotonic()
        if now - last_send >= send_interval:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            send = True
            if (last_frame_rgb is not None
                    and args.similarity_threshold < 1.0
                    and frame_similarity(frame_rgb, last_frame_rgb)
                    > args.similarity_threshold):
                send = False
                state["frames_skipped"] += 1

            if send:
                b64 = encode_frame(frame_rgb)
                with state["lock"]:
                    state["frame_queue"].append(b64)
                    state["frames_since_query"] += 1
                    state["scene_changed"] = True
                last_frame_rgb = frame_rgb
                state["frames_sent"] += 1
            last_send = now

        # Draw overlay
        with state["lock"]:
            display_text = state["streaming"] or state["text"]
        display = draw_overlay(frame_bgr, display_text)

        status = (f"Sent: {state['frames_sent']} | "
                  f"Skip: {state['frames_skipped']} | "
                  f"{'[Thinking...]' if state['thinking'] else '[Ready]'} | "
                  f"q=quit")
        cv2.putText(display, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("vLLM-OMNI Realtime", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            state["running"] = False

    cap.release()
    cv2.destroyAllWindows()


async def run_session(ws, args):
    """Run one session: send frames, query, collect responses.
    Returns when max_session_frames is reached (for session rotation)."""
    import websockets

    await ws.send(json.dumps({
        "type": "session.config",
        "model": args.model,
        "modalities": ["text"],
        "num_frames": args.num_frames,
        "max_tokens": args.max_tokens,
    }))

    last_query = 0
    session_frames = 0

    while state["running"]:
        # Send queued frames
        with state["lock"]:
            frames = state["frame_queue"][:]
            state["frame_queue"].clear()
        for b64 in frames:
            await ws.send(json.dumps({
                "type": "video.frame", "data": b64
            }))
            session_frames += 1

        # Rotate session if too many frames (prevents KV cache overflow)
        if session_frames >= args.max_session_frames:
            print(f"[Session rotation] {session_frames} frames, "
                  f"reconnecting...")
            await ws.send(json.dumps({"type": "video.done"}))
            return "rotate"

        # Auto-query at interval, only if scene changed
        now = time.monotonic()
        with state["lock"]:
            scene_changed = state["scene_changed"]
        first_query = (last_query == 0)
        if (now - last_query >= args.query_interval
                and not state["thinking"]
                and state["frames_sent"] > 0
                and (scene_changed or first_query)):
            state["thinking"] = True
            with state["lock"]:
                state["streaming"] = ""
                state["scene_changed"] = False
                state["frames_since_query"] = 0
            if first_query or not scene_changed:
                query = args.query
            else:
                query = (f"Previously: {state['text'][:80]}. "
                         f"What changed? Reply in one sentence.")
            await ws.send(json.dumps({
                "type": "video.query", "text": query,
            }))
            last_query = now

            # Collect response
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                if isinstance(msg, bytes):
                    continue
                event = json.loads(msg)
                t = event.get("type")
                if t == "response.text.delta":
                    with state["lock"]:
                        state["streaming"] += event["delta"]
                elif t == "response.text.done":
                    with state["lock"]:
                        state["text"] = event["text"]
                        state["streaming"] = ""
                    state["thinking"] = False
                    print(f"[Response] {event['text'][:100]}...")
                    break
                elif t in ("error",):
                    state["thinking"] = False
                    print(f"ERROR: {event.get('message', '')}")
                    break
                elif t in ("response.audio.start",
                           "response.audio.done",
                           "response.start"):
                    pass

        await asyncio.sleep(0.05)

    await ws.send(json.dumps({"type": "video.done"}))
    return "done"


async def ws_loop(args):
    """Background thread: manage WebSocket sessions with auto-rotation."""
    import websockets

    uri = f"ws://{args.host}:{args.port}/v1/video/chat/stream"

    while state["running"]:
        print(f"Connecting to {uri}...")
        try:
            async with websockets.connect(uri) as ws:
                result = await run_session(ws, args)
                if result == "done":
                    break
                # "rotate" — loop back and reconnect
        except Exception as e:
            print(f"[WebSocket error] {e}, reconnecting in 2s...")
            await asyncio.sleep(2)


def run_ws(args):
    asyncio.run(ws_loop(args))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Real-time video with text overlay")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--query",
                   default="In one sentence, describe what you see.")
    p.add_argument("--query-interval", type=float, default=3.0,
                   help="Seconds between queries (default: 3)")
    p.add_argument("--max-tokens", type=int, default=60,
                   help="Max response tokens (default: 60)")
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--similarity-threshold", type=float, default=0.95)
    p.add_argument("--max-session-frames", type=int, default=200,
                   help="Reconnect after N frames to prevent KV cache overflow "
                        "(default: 200)")
    args = p.parse_args()

    print("Press 'q' in the webcam window to quit")

    # WebSocket in background thread, webcam in main thread
    ws_thread = threading.Thread(target=run_ws, args=(args,), daemon=True)
    ws_thread.start()

    webcam_loop(args)
    state["running"] = False
    ws_thread.join(timeout=3)
