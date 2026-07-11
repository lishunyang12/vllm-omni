from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexCapabilities,
    DuplexSession,
    DuplexSessionConfig,
    DuplexSessionRegistry,
    DuplexTurnController,
    DuplexTurnEventType,
    DuplexTurnState,
)


def test_duplex_session_commits_text_and_audio_as_one_turn():
    registry = DuplexSessionRegistry()
    session = registry.create(DuplexSessionConfig(model="test-model"))

    session.append_text("hello")
    session.append_audio("YWJj", fmt="wav", sample_rate_hz=16000)
    committed = session.commit_user_input()

    assert committed is not None
    assert committed.turn_id == 0
    assert committed.input_commit_seq == 1
    assert committed.epoch == 0
    assert len(session.history) == 1
    content = session.history[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "hello"}
    assert content[1]["type"] == "audio_url"
    assert content[1]["audio_url"]["url"] == "data:audio/wav;base64,YWJj"
    assert session.turn_state == DuplexTurnState.USER_COMMITTED


def test_native_input_commit_does_not_advance_model_turn_identity():
    session = DuplexSession(
        session_id="sid-model-owned-turn",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )

    first = session.commit_native_audio_input(transcript="first chunk")
    second = session.commit_native_audio_input(transcript="second chunk")

    assert session.input_commit_seq == 2
    assert first.input_commit_seq == 1
    assert second.input_commit_seq == 2
    assert first.turn_id == second.turn_id == session.turn_id == 0


def test_duplex_barge_in_advances_epoch_and_drops_uncommitted_assistant_text():
    registry = DuplexSessionRegistry()
    session = registry.create()
    response_id = session.begin_response()
    session.active_request_id = "chatcmpl-duplex-test"
    session.append_assistant_text("unplayed answer")

    new_epoch = session.barge_in()

    assert response_id is not None
    assert new_epoch == 1
    assert session.epoch == 1
    assert session.active_request_id is None
    assert session.active_response_id is None
    assert session.assistant_text_buffer == []
    assert session.history == []
    assert session.turn_state == DuplexTurnState.BARGE_IN


def test_duplex_playback_ack_tracks_committed_cursor_separately():
    registry = DuplexSessionRegistry()
    session = registry.create()
    session.mark_audio_sent(duration_ms=10_000)

    session.acknowledge_playback(played_ms=2_000)

    assert session.playback.generated_ms == 10_000
    assert session.playback.sent_ms == 10_000
    assert session.playback.played_ms == 2_000
    assert session.playback.committed_ms == 2_000


def test_duplex_history_commit_uses_audio_text_alignment_marks():
    registry = DuplexSessionRegistry()
    session = registry.create()
    session.begin_response()

    session.append_assistant_text("hello ")
    session.mark_audio_sent(duration_ms=1_000, text_chars=6)
    session.append_assistant_text("world")
    session.mark_audio_sent(duration_ms=2_000, text_chars=11)
    session.acknowledge_playback(played_ms=1_200, committed_ms=1_200)

    committed = session.end_response(commit_text=True)

    assert committed == {"role": "assistant", "content": "hello w"}


def test_turn_controller_accepts_external_signal_source():
    registry = DuplexSessionRegistry()
    session = registry.create()
    controller = DuplexTurnController()

    event = controller.signal(session, DuplexTurnEventType.USER_STARTED.value)

    assert event["type"] == "turn.event"
    assert event["event"] == "user_started"
    assert event["turn_state"] == "user_speaking"
    assert session.turn_state == DuplexTurnState.USER_SPEAKING


def test_duplex_capabilities_do_not_claim_core_kv_or_input_append():
    registry = DuplexSessionRegistry()
    session = registry.create()
    caps = session.capabilities.as_dict()

    assert caps["implementation_level"] == "serving_session_adapter"
    assert caps["supports_kv_lease"] is False
    assert caps["supports_input_append"] is False
    assert caps["supports_reencode_context"] is True
    assert caps["adapter_patterns"] == ["chunk_group_append"]
    assert "turn_commit_only" in caps["input_modes"]
    assert "client_event" in caps["signal_sources"]


def test_minicpmo_native_capabilities_separate_model_state_from_core_kv_lease():
    caps = DuplexCapabilities.minicpmo45_native().as_dict()

    assert caps["implementation_level"] == "model_native_duplex"
    assert caps["supports_input_append"] is True
    assert caps["input_modes"] == ["append_audio_chunk"]
    assert caps["adapter_patterns"] == ["scheduler_data_plane"]
    assert caps["supports_model_internal_state"] is True
    assert caps["requires_model_runner_kv"] is True
    assert caps["requires_native_stage_role"] is True
    assert caps["supports_kv_lease"] is False
    assert caps["supports_core_kv_lease"] is False
    assert caps["supports_stage_resumption"] is True
    assert caps["supports_scheduler_native_append"] is True
    assert caps["supports_core_resumable_request"] is True
    assert caps["supports_stage_connector_handoff"] is True
    assert caps["supports_audio_truncate"] is True
    assert caps["stage_handoff_transport"] == "scheduler_data_plane"
