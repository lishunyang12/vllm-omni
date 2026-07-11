import importlib.util
import sys
import wave
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "examples/online_serving/minicpmo/realtime_web/static/app.js"
DEMO_PATH = Path(__file__).resolve().parents[2] / "examples/online_serving/minicpmo/realtime_duplex_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("minicpmo_realtime_duplex_demo_test", DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_realtime_duplex_demo_resolves_distinct_turn_inputs():
    demo = _load_demo_module()
    primary = Path("first.wav")

    assert demo._turn_input_paths(primary, [], turns=3) == [primary, primary, primary]
    assert demo._turn_input_paths(primary, ["second.wav", "third.wav"], turns=3) == [
        primary,
        Path("second.wav"),
        Path("third.wav"),
    ]
    with pytest.raises(ValueError, match="one --turn-input-wav"):
        demo._turn_input_paths(primary, ["second.wav"], turns=3)


def test_realtime_duplex_demo_resolves_explicit_turn_durations():
    demo = _load_demo_module()

    assert demo._turn_durations([], turns=3, first_turn_ms=3000) == [3000, 1200, 1200]
    assert demo._turn_durations([0, 900, 1500], turns=3, first_turn_ms=3000) == [None, 900, 1500]
    with pytest.raises(ValueError, match="one --turn-duration-ms"):
        demo._turn_durations([0, 900], turns=3, first_turn_ms=3000)
    with pytest.raises(ValueError, match="non-negative"):
        demo._turn_durations([0, -1, 900], turns=3, first_turn_ms=3000)


def test_realtime_duplex_demo_resolves_transcript_labels_for_requested_turns():
    demo = _load_demo_module()

    assert demo._turn_transcripts("first", turns=1) == ["first"]
    assert demo._turn_transcripts("first", turns=4) == ["first", "继续", "再说一次", "turn-4"]


def test_realtime_duplex_demo_reads_response_playback_cursor():
    demo = _load_demo_module()
    state = demo.DemoState()
    state.add(
        {
            "type": "response.done",
            "response_id": "resp-1",
            "response": {
                "id": "resp-1",
                "metadata": {
                    "playback": {
                        "sent_ms": 27920,
                        "played_ms": 0,
                    }
                },
            },
        }
    )

    assert state.response_playback_sent_ms("resp-1") == 27920


def test_realtime_duplex_demo_full_turn_duration_does_not_slice_audio():
    demo = _load_demo_module()
    pcm16 = b"\x01\x00" * (demo.PCM16_SAMPLE_RATE * 2)

    assert demo._select_turn_audio(pcm16, None) == pcm16
    assert len(demo._select_turn_audio(pcm16, 1000)) == demo.PCM16_SAMPLE_RATE * demo.PCM16_BYTES_PER_SAMPLE


def test_realtime_duplex_demo_distinct_inputs_compare_audio_content():
    demo = _load_demo_module()
    paths = [Path("first.wav"), Path("second.wav"), Path("third.wav")]

    assert demo._turn_inputs_are_distinct(paths, [b"first", b"second", b"third"]) is True
    assert demo._turn_inputs_are_distinct(paths, [b"same", b"same", b"third"]) is False
    assert demo._turn_inputs_are_distinct([paths[0], paths[0]], [b"first", b"second"]) is False


def _add_response_transcript(state, response_id, *, transcript, audio=True):
    state.add(
        {
            "type": "response.audio.delta",
            "response_id": response_id,
            "delta": "YQ==" if audio else "",
        }
    )
    if transcript:
        state.add(
            {
                "type": "response.audio_transcript.delta",
                "response_id": response_id,
                "delta": transcript,
            }
        )
        state.add(
            {
                "type": "response.audio_transcript.done",
                "response_id": response_id,
                "transcript": transcript,
            }
        )
    state.add(
        {
            "type": "response.done",
            "response_id": response_id,
            "response": {"id": response_id, "status": "completed"},
        }
    )


def test_realtime_duplex_demo_gate_checks_delta_done_and_turn_independence():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(
        state,
        "resp-1",
        transcript="第一轮回答",
    )
    _add_response_transcript(
        state,
        "resp-2",
        transcript="第二轮回答",
    )

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-1", "resp-2"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=True,
    )

    assert result["transcript_delta_done_ok"] is True
    assert result["cross_turn_independent_ok"] is True


def test_realtime_duplex_demo_gate_rejects_cross_turn_tail_reuse():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(state, "resp-1", transcript="第一轮回答")
    _add_response_transcript(state, "resp-2", transcript="第二轮仍然带着第一轮回答")

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-1", "resp-2"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=True,
    )

    assert result["cross_turn_independent_ok"] is False


def test_realtime_duplex_demo_gate_rejects_terminal_only_previous_tail():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(state, "resp-1", transcript="上一轮回答结尾是的吗？")
    _add_response_transcript(state, "resp-2", transcript="的吗？")

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-1", "resp-2"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=True,
    )

    assert result["cross_turn_independent_ok"] is False


def test_realtime_duplex_demo_gate_rejects_delta_done_mismatch():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(state, "resp-1", transcript="delta文本")
    done = next(event for event in state.events if event.get("type") == "response.audio_transcript.done")
    done["transcript"] = "另一个done文本"

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-1"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=False,
    )

    assert result["transcript_delta_done_ok"] is False


def test_realtime_duplex_demo_gate_rejects_terminal_only_stale_tail():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(
        state,
        "resp-empty",
        transcript="的吗？",
        audio=False,
    )

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-empty"],
        expected_empty_response_ids={"resp-empty"},
        require_cross_turn_independence=False,
    )

    assert result["empty_turns_ok"] is False


def test_realtime_duplex_demo_gate_rejects_audio_without_transcript():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(
        state,
        "resp-audio-no-text",
        transcript="",
        audio=True,
    )

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-audio-no-text"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=False,
    )

    assert result["nonempty_audio_has_transcript_ok"] is False


def test_realtime_duplex_demo_gate_rejects_incomplete_model_turn_sentence():
    demo = _load_demo_module()
    state = demo.DemoState()
    _add_response_transcript(state, "resp-1", transcript="哎，不是说好不")

    result = demo._evaluate_transcript_integrity(
        state,
        ["resp-1"],
        expected_empty_response_ids=set(),
        require_cross_turn_independence=False,
        require_terminal_punctuation=True,
    )

    assert result["terminal_punctuation_ok"] is False


def test_realtime_duplex_demo_writes_audio_per_response(tmp_path):
    demo = _load_demo_module()
    state = demo.DemoState()
    for response_id, payload in (("resp-1", b"\x01\x00"), ("resp-2", b"\x02\x00")):
        state.add({"type": "response.created", "response": {"id": response_id}})
        state.add(
            {
                "type": "response.audio.delta",
                "response_id": response_id,
                "delta": demo.base64.b64encode(payload).decode(),
                "sample_rate_hz": 24000,
            }
        )

    demo._write_demo_artifacts(state, tmp_path, output_audio_format="pcm16")

    for index, expected in enumerate((b"\x01\x00", b"\x02\x00"), start=1):
        with wave.open(str(tmp_path / f"response_{index:02d}.wav"), "rb") as wf:
            assert wf.getframerate() == 24000
            assert wf.readframes(wf.getnframes()) == expected


def test_realtime_web_defaults_to_streaming_playback():
    source = APP_JS.read_text(encoding="utf-8")

    assert "PLAYBACK_MODE === 'buffered'" in source
    assert "QUERY.get('buffered') === '1'" in source
    assert "const BUFFER_OUTPUT_AUDIO = true" not in source
    assert "streaming-default" in source


def test_realtime_web_streams_audio_delta_before_audio_done():
    source = APP_JS.read_text(encoding="utf-8")

    delta_case = source[source.index("case 'response.audio.delta'") : source.index("case 'response.audio.done'")]
    assert "if (BUFFER_OUTPUT_AUDIO) bufferPlayback" in delta_case
    assert "else {" in delta_case
    assert "feedPlayback(pcm, sr)" in delta_case
    assert "decodeOutputAudioDelta(e)" in delta_case
    assert "pcm_f32le" in source
    assert "decodeAudioData" in source

    start_call = source[source.index("async function startCall()") : source.index("function stopCall()")]
    assert "if (!BUFFER_OUTPUT_AUDIO)" in start_call
    assert "audioWorklet.addModule('static/ttsPlaybackProcessor.js')" in start_call


def test_realtime_web_waits_for_server_boundary_before_reopening_full_mode_mic():
    source = APP_JS.read_text(encoding="utf-8")

    assert "assistantServerBoundarySeen" in source

    stopped_case = source[
        source.index("if (data.type === 'ttsPlaybackStopped')") : source.index(
            "if (data.type === 'ttsPlaybackUnderrun')"
        )
    ]
    assert "assistantServerBoundarySeen" in stopped_case
    assert "endAssistantOutput(ECHO_GUARD_MS)" in stopped_case

    idle_timer = source[
        source.index("assistantAudioIdleTimer = setTimeout") : source.index("function endAssistantOutput")
    ]
    assert "assistantShouldWaitForServerBoundary()" in idle_timer


def test_realtime_web_drains_streaming_playback_on_response_done():
    source = APP_JS.read_text(encoding="utf-8")

    boundary_fn = source[
        source.index("function markAssistantServerBoundary(") : source.index("function endAssistantOutput")
    ]
    assert "ttsNode.port.postMessage({ type: 'drain' })" in boundary_fn

    done_case = source[source.index("case 'response.done'") : source.index("case 'error'")]
    assert "markAssistantServerBoundary(e)" in done_case
    assert "gapFillMs" in source
