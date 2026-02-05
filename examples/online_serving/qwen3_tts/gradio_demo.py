"""Gradio demo for Qwen3-TTS online serving via /v1/audio/speech API.

Supports all 3 task types:
  - CustomVoice: Predefined speaker with optional style instructions
  - VoiceDesign: Natural language voice description
  - Base: Voice cloning from reference audio

Usage:
    # Start the server first (see run_server.sh), then:
    python gradio_demo.py --api-base http://localhost:8000

    # Or use run_gradio_demo.sh to start both server and demo together.
"""

import argparse
import base64
import io

import gradio as gr
import httpx
import numpy as np
import soundfile as sf

SUPPORTED_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

TASK_TYPES = ["CustomVoice", "VoiceDesign", "Base"]


def fetch_voices(api_base: str) -> list[str]:
    """Fetch available voices from the server."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{api_base}/v1/audio/voices",
                headers={"Authorization": "Bearer EMPTY"},
            )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("voices", ["Vivian", "Ryan"])
    except Exception:
        pass
    return ["Vivian", "Ryan"]


def encode_audio_to_base64(audio_data: tuple) -> str:
    """Encode Gradio audio input (sample_rate, numpy_array) to base64 data URL."""
    sample_rate, audio_np = audio_data

    if audio_np.dtype != np.int16:
        if audio_np.dtype in (np.float32, np.float64):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_np.astype(np.int16)

    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV")
    wav_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{wav_b64}"


def generate_speech(
    api_base: str,
    text: str,
    task_type: str,
    voice: str,
    language: str,
    instructions: str,
    ref_audio: tuple | None,
    ref_text: str,
    response_format: str,
    speed: float,
):
    """Call /v1/audio/speech and return audio for Gradio."""
    if not text or not text.strip():
        raise gr.Error("Please enter text to synthesize.")

    # Build request payload
    payload = {
        "input": text.strip(),
        "response_format": response_format,
        "speed": speed,
    }

    if task_type:
        payload["task_type"] = task_type
    if language:
        payload["language"] = language

    # Task-specific parameters
    if task_type == "CustomVoice":
        if voice:
            payload["voice"] = voice
        if instructions and instructions.strip():
            payload["instructions"] = instructions.strip()

    elif task_type == "VoiceDesign":
        if not instructions or not instructions.strip():
            raise gr.Error(
                "VoiceDesign task requires voice style instructions."
            )
        payload["instructions"] = instructions.strip()

    elif task_type == "Base":
        if ref_audio is None:
            raise gr.Error(
                "Base (voice clone) task requires reference audio."
            )
        payload["ref_audio"] = encode_audio_to_base64(ref_audio)
        if ref_text and ref_text.strip():
            payload["ref_text"] = ref_text.strip()

    # Call the API
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{api_base}/v1/audio/speech",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer EMPTY",
                },
            )
    except httpx.TimeoutException:
        raise gr.Error("Request timed out. The server may be busy.")
    except httpx.ConnectError:
        raise gr.Error(
            f"Cannot connect to server at {api_base}. "
            "Make sure the vLLM server is running."
        )

    if resp.status_code != 200:
        raise gr.Error(f"Server error ({resp.status_code}): {resp.text}")

    # Check for JSON error response
    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            error_data = resp.json()
            raise gr.Error(f"Server error: {error_data}")
        except ValueError:
            pass

    # Decode audio response
    try:
        audio_np, sample_rate = sf.read(io.BytesIO(resp.content))
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]
        return (sample_rate, audio_np.astype(np.float32))
    except Exception as e:
        raise gr.Error(f"Failed to decode audio response: {e}")


def on_task_type_change(task_type: str):
    """Update UI visibility based on selected task type."""
    if task_type == "CustomVoice":
        return (
            gr.update(visible=True),   # voice dropdown
            gr.update(visible=True, info="Optional style/emotion instructions"),
            gr.update(visible=False),  # ref_audio
            gr.update(visible=False),  # ref_text
        )
    elif task_type == "VoiceDesign":
        return (
            gr.update(visible=False),  # voice dropdown
            gr.update(visible=True, info="Required: describe the voice style"),
            gr.update(visible=False),  # ref_audio
            gr.update(visible=False),  # ref_text
        )
    elif task_type == "Base":
        return (
            gr.update(visible=False),  # voice dropdown
            gr.update(visible=False),  # instructions
            gr.update(visible=True),   # ref_audio
            gr.update(visible=True),   # ref_text
        )
    return (
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def build_interface(api_base: str):
    """Build the Gradio interface."""
    voices = fetch_voices(api_base)

    css = """
    #generate-btn button { width: 100%; }
    .task-info { padding: 8px 12px; border-radius: 6px;
                 background: #f0f4ff; margin-bottom: 8px; }
    """

    with gr.Blocks(css=css, title="Qwen3-TTS Demo") as demo:
        gr.Markdown("# Qwen3-TTS Online Serving Demo")
        gr.Markdown(f"**Server:** `{api_base}`")

        with gr.Row():
            # Left column: inputs
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter text here, e.g., Hello, how are you?",
                    lines=4,
                )

                with gr.Row():
                    task_type = gr.Radio(
                        choices=TASK_TYPES,
                        value="CustomVoice",
                        label="Task Type",
                        scale=2,
                    )
                    language = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGES,
                        value="Auto",
                        label="Language",
                        scale=1,
                    )

                # CustomVoice controls
                voice = gr.Dropdown(
                    choices=voices,
                    value=voices[0] if voices else None,
                    label="Speaker",
                    visible=True,
                )

                # Instructions (CustomVoice optional, VoiceDesign required)
                instructions = gr.Textbox(
                    label="Instructions",
                    placeholder=(
                        "e.g., Speak with excitement / "
                        "A warm, friendly female voice"
                    ),
                    lines=2,
                    visible=True,
                    info="Optional style/emotion instructions",
                )

                # Base (voice clone) controls
                ref_audio = gr.Audio(
                    label="Reference Audio (for voice cloning)",
                    type="numpy",
                    sources=["upload", "microphone"],
                    visible=False,
                )
                ref_text = gr.Textbox(
                    label="Reference Audio Transcript",
                    placeholder="Transcript of the reference audio (optional, improves quality)",
                    lines=2,
                    visible=False,
                )

                with gr.Row():
                    response_format = gr.Dropdown(
                        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
                        value="wav",
                        label="Audio Format",
                        scale=1,
                    )
                    speed = gr.Slider(
                        minimum=0.25,
                        maximum=4.0,
                        value=1.0,
                        step=0.05,
                        label="Speed",
                        scale=1,
                    )

                generate_btn = gr.Button(
                    "Generate Speech",
                    variant="primary",
                    size="lg",
                    elem_id="generate-btn",
                )

            # Right column: output
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Generated Audio",
                    interactive=False,
                )
                gr.Markdown(
                    "### Task Types\n"
                    "- **CustomVoice**: Use a predefined speaker "
                    "(Vivian, Ryan, etc.) with optional style instructions\n"
                    "- **VoiceDesign**: Describe the desired voice in natural "
                    "language (instructions required)\n"
                    "- **Base**: Clone a voice from reference audio"
                )

        # Dynamic UI updates
        task_type.change(
            fn=on_task_type_change,
            inputs=[task_type],
            outputs=[voice, instructions, ref_audio, ref_text],
        )

        # Generate button
        generate_btn.click(
            fn=lambda *args: generate_speech(api_base, *args),
            inputs=[
                text_input,
                task_type,
                voice,
                language,
                instructions,
                ref_audio,
                ref_text,
                response_format,
                speed,
            ],
            outputs=[audio_output],
        )

        demo.queue()
    return demo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio demo for Qwen3-TTS online serving."
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000",
        help="Base URL for the vLLM API server (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="Host/IP for Gradio server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio server (default: 7860).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the Gradio demo publicly.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Connecting to vLLM server at: {args.api_base}")
    demo = build_interface(args.api_base)

    try:
        demo.launch(
            server_name=args.ip,
            server_port=args.port,
            share=args.share,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
