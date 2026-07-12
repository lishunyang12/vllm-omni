# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.experimental.fullduplex.core.continuous import (
    ContinuousDuplexState,
    ContinuousTransitionError,
    reduce_continuous_event,
)
from vllm_omni.experimental.fullduplex.core.events import (
    AppendToEngine,
    CancelFence,
    CloseSessionResources,
    EmitProtocolEvent,
    InputChunk,
    InterruptRequested,
    ModelAudioDelta,
    ModelListening,
    ModelSegmentEnded,
    ModelSpeaking,
    ModelTextDelta,
    ModelTurnEnded,
    PlaybackAcknowledged,
    ProtocolEventKind,
    RebuildStage0Context,
    ResetStage1,
    SessionCloseRequested,
)
from vllm_omni.experimental.fullduplex.core.identity import DuplexFence


def _open() -> ContinuousDuplexState:
    return ContinuousDuplexState.open("s")


def _epoch_fence(state: ContinuousDuplexState) -> DuplexFence:
    return state.fence


def _drive(state, *events):
    """Reduce a sequence, returning (final_state, [effects_per_event])."""
    effects = []
    for event in events:
        state, produced = reduce_continuous_event(state, event)
        effects.append(produced)
    return state, effects


def test_input_chunk_is_accepted_while_model_is_speaking():
    state = _open()
    ef = _epoch_fence(state)
    # Open a response, then append input mid-response — the full-duplex case the
    # turn-based reducer forbids.
    state, _ = reduce_continuous_event(state, ModelSpeaking(fence=ef))
    assert state.speaking is True

    state, effects = reduce_continuous_event(state, InputChunk(data=b"\x00\x00\x00\x00", modality="audio"))

    assert effects == (AppendToEngine(ef, chunk=InputChunk(data=b"\x00\x00\x00\x00", modality="audio")),)
    assert state.speaking is True  # still speaking; input did not interrupt


def test_model_owned_response_without_client_commit():
    state = _open()
    ef = _epoch_fence(state)
    rf = DuplexFence("s", epoch=0, turn_id=1, response_seq=1)

    state, effects = _drive(
        state,
        ModelSpeaking(fence=ef),
        ModelTextDelta(text="hi", fence=ef),
        ModelSegmentEnded(fence=ef),
        ModelTurnEnded(fence=ef),
    )

    assert effects[0] == (EmitProtocolEvent(rf, ProtocolEventKind.RESPONSE_STARTED, payload=ModelSpeaking(fence=ef)),)
    assert effects[1] == (
        EmitProtocolEvent(rf, ProtocolEventKind.TEXT_DELTA, payload=ModelTextDelta(text="hi", fence=ef)),
    )
    assert effects[2] == (EmitProtocolEvent(rf, ProtocolEventKind.SEGMENT_ENDED, payload=ModelSegmentEnded(fence=ef)),)
    assert effects[3] == (
        EmitProtocolEvent(rf, ProtocolEventKind.RESPONSE_COMPLETED, payload=None),
        ResetStage1(ef),
    )
    assert state.speaking is False
    assert state.response_fence is None


def test_second_response_in_epoch_gets_distinct_response_fence():
    state = _open()
    ef = _epoch_fence(state)
    state, _ = _drive(state, ModelSpeaking(fence=ef), ModelTurnEnded(fence=ef))
    state, effects = _drive(state, ModelSpeaking(fence=ef))

    rf2 = DuplexFence("s", epoch=0, turn_id=2, response_seq=2)
    assert effects[0] == (EmitProtocolEvent(rf2, ProtocolEventKind.RESPONSE_STARTED, payload=ModelSpeaking(fence=ef)),)
    assert state.response_counter == 2


def test_repeated_speaking_is_idempotent_within_a_response():
    state = _open()
    ef = _epoch_fence(state)
    state, first = reduce_continuous_event(state, ModelSpeaking(fence=ef))
    state, second = reduce_continuous_event(state, ModelSpeaking(fence=ef))

    assert first  # RESPONSE_STARTED emitted once
    assert second == ()  # continuation, no duplicate start
    assert state.response_counter == 1


def test_audio_cursor_accumulates_from_byte_length():
    state = _open()
    ef = _epoch_fence(state)
    state, _ = reduce_continuous_event(state, ModelSpeaking(fence=ef))

    # 100 float32 samples then 50 more -> generated cursor 100 then 150.
    state, e1 = reduce_continuous_event(
        state, ModelAudioDelta(data=b"\x00\x00\x00\x00" * 100, generated_cursor=999, sent_cursor=999, fence=ef)
    )
    assert state.playback.generated == 100 and state.playback.sent == 100
    state, e2 = reduce_continuous_event(
        state, ModelAudioDelta(data=b"\x00\x00\x00\x00" * 50, generated_cursor=1, sent_cursor=1, fence=ef)
    )
    assert state.playback.generated == 150 and state.playback.sent == 150
    assert e1[0].kind is ProtocolEventKind.AUDIO_DELTA and e2[0].kind is ProtocolEventKind.AUDIO_DELTA


def test_model_listening_while_speaking_ends_the_response():
    state = _open()
    ef = _epoch_fence(state)
    rf = DuplexFence("s", epoch=0, turn_id=1, response_seq=1)
    state, _ = reduce_continuous_event(state, ModelSpeaking(fence=ef))

    state, effects = reduce_continuous_event(state, ModelListening(fence=ef))

    assert effects == (
        EmitProtocolEvent(rf, ProtocolEventKind.MODEL_LISTENING, payload=ModelListening(fence=ef)),
        EmitProtocolEvent(rf, ProtocolEventKind.RESPONSE_COMPLETED, payload=None),
        ResetStage1(ef),
    )
    assert state.speaking is False


def test_idle_model_listening_changes_nothing_but_emits():
    state = _open()
    ef = _epoch_fence(state)
    state, effects = reduce_continuous_event(state, ModelListening(fence=ef))
    assert effects == (EmitProtocolEvent(ef, ProtocolEventKind.MODEL_LISTENING, payload=ModelListening(fence=ef)),)
    assert state.speaking is False


def test_barge_in_bumps_epoch_and_rebuilds_without_killing_session():
    state = _open()
    ef = _epoch_fence(state)
    state, _ = _drive(
        state,
        ModelSpeaking(fence=ef),
        ModelAudioDelta(data=b"\x00\x00\x00\x00" * 10, generated_cursor=0, sent_cursor=0, fence=ef),
    )

    state, effects = reduce_continuous_event(state, InterruptRequested(reason="user", fence=ef))

    new_fence = DuplexFence("s", epoch=1)
    assert effects == (
        CancelFence(ef),
        ResetStage1(ef),
        RebuildStage0Context(new_fence, committed_history=(), committed_playback_position=0),
    )
    assert state.fence == new_fence
    assert state.speaking is False
    assert state.playback.generated == 0  # cursor reset for the new epoch


def test_stale_model_output_from_old_epoch_is_dropped():
    state = _open()
    ef = _epoch_fence(state)
    state, _ = reduce_continuous_event(state, InterruptRequested(reason="user", fence=ef))
    assert state.fence.epoch == 1

    # Audio tagged with the pre-barge-in epoch is stale.
    state, effects = reduce_continuous_event(
        state, ModelAudioDelta(data=b"\x00\x00\x00\x00", generated_cursor=1, sent_cursor=1, fence=ef)
    )
    assert effects == ()
    assert state.stale_event_count == 1


def test_playback_ack_advances_cursor():
    state = _open()
    ef = _epoch_fence(state)
    state, _ = _drive(
        state,
        ModelSpeaking(fence=ef),
        ModelAudioDelta(data=b"\x00\x00\x00\x00" * 100, generated_cursor=0, sent_cursor=0, fence=ef),
    )

    state, effects = reduce_continuous_event(state, PlaybackAcknowledged(cursor=60, committed_cursor=40, fence=ef))
    assert effects == ()
    assert state.playback.played == 60 and state.playback.committed == 40


def test_text_before_speaking_is_illegal():
    state = _open()
    ef = _epoch_fence(state)
    with pytest.raises(ContinuousTransitionError):
        reduce_continuous_event(state, ModelTextDelta(text="x", fence=ef))


def test_session_close_cancels_active_response_and_closes():
    state = _open()
    ef = _epoch_fence(state)
    state, _ = reduce_continuous_event(state, ModelSpeaking(fence=ef))

    state, effects = reduce_continuous_event(state, SessionCloseRequested(reason="bye"))
    assert effects == (CancelFence(ef), ResetStage1(ef), CloseSessionResources(ef))
    assert state.session_phase.value == "closed"
