"""Example client for streaming video frames to vLLM-OMNI.

Sends video frames via WebSocket to the Qwen3-Omni pipeline and
receives text + audio responses.

Usage:
    # Stream from webcam:
    python streaming_video_client.py --source webcam

    # Stream from a video file:
    python streaming_video_client.py --source video.mp4

    # Stream from a directory of images:
    python streaming_video_client.py --source ./frames/

Requires:
    pip install websockets pillow opencv-python
"""

import argparse
import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path

import websockets


def encode_frame(frame_rgb, quality: int = 85) -> str:
    """Encode a numpy RGB frame to base64 JPEG."""
    from PIL import Image

    img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def stream_webcam(ws, fps: float, duration: float):
    """Stream frames from webcam."""
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return 0

    interval = 1.0 / fps
    start = time.monotonic()
    count = 0

    try:
        while time.monotonic() - start < duration:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            b64 = encode_frame(frame_rgb)
            await ws.send(json.dumps({"type": "video.frame", "data": b64}))
            count += 1
            if count % max(1, int(fps)) == 0:
                print(f"  Sent {count} frames ({time.monotonic() - start:.1f}s)")
            await asyncio.sleep(interval)
    finally:
        cap.release()
    return count


async def stream_video_file(ws, path: str, fps: float):
    """Stream frames from a video file."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = max(1, int(src_fps / fps))
    idx = 0
    count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            b64 = encode_frame(frame_rgb)
            await ws.send(json.dumps({"type": "video.frame", "data": b64}))
            count += 1
            if count % 10 == 0:
                print(f"  Sent {count} frames")
        idx += 1
    cap.release()
    return count


async def stream_image_dir(ws, dir_path: str, fps: float):
    """Stream frames from a directory of images."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in Path(dir_path).iterdir() if p.suffix.lower() in exts)
    if not files:
        print(f"ERROR: No images in {dir_path}")
        return 0

    for i, f in enumerate(files):
        with open(f, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("utf-8")
        await ws.send(json.dumps({"type": "video.frame", "data": b64}))
        if (i + 1) % 10 == 0:
            print(f"  Sent {i + 1}/{len(files)} frames")
        await asyncio.sleep(1.0 / fps)
    return len(files)


async def main(args):
    uri = f"ws://{args.host}:{args.port}/v1/video/chat/stream"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        # 1. Send config
        config = {
            "type": "session.config",
            "model": args.model,
            "modalities": args.modalities.split(","),
            "num_frames": args.num_frames,
        }
        await ws.send(json.dumps(config))

        # 2. Stream frames
        source = args.source
        if source == "webcam":
            count = await stream_webcam(ws, args.fps, args.duration)
        elif Path(source).is_dir():
            count = await stream_image_dir(ws, source, args.fps)
        elif Path(source).is_file():
            count = await stream_video_file(ws, source, args.fps)
        else:
            print(f"ERROR: Unknown source: {source}", file=sys.stderr)
            return

        print(f"\nBuffered {count} frames. Submitting query: {args.query!r}\n")

        # 3. Submit query
        query_start = time.monotonic()
        await ws.send(json.dumps({"type": "video.query", "text": args.query}))

        # 4. Receive response
        token_count = 0
        first_token_time = None
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                print(f"  [audio: {len(msg)} bytes]")
                continue
            event = json.loads(msg)
            t = event.get("type")
            if t == "response.start":
                print("Response: ", end="", flush=True)
            elif t == "response.text.delta":
                if first_token_time is None:
                    first_token_time = time.monotonic()
                token_count += 1
                print(event["delta"], end="", flush=True)
            elif t == "response.text.done":
                elapsed = time.monotonic() - query_start
                ttft = (first_token_time - query_start) if first_token_time else 0
                print(f"\n\n[Done] Full text: {event['text'][:100]}...")
                print(f"  Frames sent: {count}")
                print(f"  Time to first token: {ttft:.2f}s")
                print(f"  Total inference time: {elapsed:.2f}s")
                print(f"  Tokens generated: {token_count}")
                if elapsed > 0:
                    print(f"  Throughput: {count / elapsed:.2f} frames/s, "
                          f"{token_count / elapsed:.1f} tokens/s")
                break
            elif t == "response.audio.start":
                print(f"\n  [Audio stream: {event.get('format', 'wav')}]")
            elif t == "response.audio.done":
                print("  [Audio complete]")
            elif t == "error":
                print(f"\nERROR: {event['message']}", file=sys.stderr)
                break

        # 5. Close session — send video.done first, then wait for
        #    server acknowledgement (session.done)
        await ws.send(json.dumps({"type": "video.done"}))
        msg = await ws.recv()
        event = json.loads(msg)
        if event.get("type") == "session.done":
            print("Session closed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stream video to vLLM-OMNI")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--source", default="webcam", help="'webcam', video file, or image directory")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--query", default="Describe what you see in the video.")
    p.add_argument("--duration", type=float, default=10.0, help="Webcam capture duration (seconds)")
    p.add_argument("--modalities", default="text", help="Comma-separated output modalities: text,audio")
    p.add_argument("--num-frames", type=int, default=32)
    args = p.parse_args()
    asyncio.run(main(args))
