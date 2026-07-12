# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Continuous full-duplex reducer for model-owned listen/speak streaming.

The turn-based reducer in :mod:`.state` models a request/response protocol
(client streams input, *commits*, then the model responds). MiniCPM-o native
duplex is different: audio streams continuously, and the model itself decides —
per ~1 s chunk — whether to listen or speak, with no client commit and with a
spoken utterance spanning many input chunks *while new input keeps arriving*.

This reducer captures that continuous model:

* ``InputChunk`` is legal at all times (including while the model speaks), which
  is the essence of full duplex and what the turn-based reducer forbids.
* A response is **model-owned**: it opens when the model starts speaking and
  closes on the model's turn end (or when the model returns to listening).
* Two fence notions coexist. The *epoch fence* ``(session, epoch, 0, 0)`` is
  stable per epoch and scopes input appends and model-output validation; a
  barge-in bumps the epoch. The *response fence* ``(session, epoch, k, k)`` is
  minted here when the model opens its ``k``-th response of the epoch, and is
  used for protocol emission so each response gets a distinct id.

Model outputs are validated by **epoch** (not the full fence): the engine tags
them with the epoch fence, and anything from an older epoch is stale (dropped).
Playback cursors are computed here from the audio byte length and are
epoch-cumulative, so they do not depend on any model-adapter cursor scoping.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .events import (
    AppendToEngine,
    CancelFence,
    CloseSessionResources,
    CommittedHistoryItem,
    DomainEvent,
    DuplexEffect,
    EmitProtocolEvent,
    EngineAppendAccepted,
    EngineFailed,
    HistoryCommitted,
    InputChunk,
    InputCommitted,
    InputStarted,
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
from .identity import DuplexFence
from .playback import PlaybackCursor, PlaybackCursorError
from .state import (
    DuplexFenceMismatchError,
    DuplexMissingFenceError,
    DuplexReducerError,
    DuplexSessionPhase,
)


class ContinuousTransitionError(DuplexReducerError):
    """An event that is not legal for the current continuous-duplex state."""

    def __init__(self, state: ContinuousDuplexState, event: DomainEvent, detail: str) -> None:
        phase = "speaking" if state.speaking else "listening"
        super().__init__(
            f"illegal {type(event).__name__} transition from "
            f"session={state.session_phase.value}, model={phase}: {detail}"
        )
        self.state = state
        self.event = event


_FENCED_MODEL_EVENT = (
    EngineAppendAccepted
    | HistoryCommitted
    | InterruptRequested
    | ModelListening
    | ModelSpeaking
    | ModelTextDelta
    | ModelAudioDelta
    | ModelSegmentEnded
    | ModelTurnEnded
    | PlaybackAcknowledged
    | EngineFailed
)


@dataclass(frozen=True, slots=True)
class ContinuousDuplexState:
    """State for one continuous full-duplex session.

    ``fence`` is the epoch fence; ``response_fence`` is the live response's fence
    (``None`` when the model is not speaking). ``response_counter`` names the next
    response within the current epoch.
    """

    fence: DuplexFence
    session_phase: DuplexSessionPhase = DuplexSessionPhase.OPEN
    speaking: bool = False
    response_fence: DuplexFence | None = None
    response_counter: int = 0
    playback: PlaybackCursor = PlaybackCursor()
    last_committed_playback_position: int = 0
    committed_history: tuple[CommittedHistoryItem, ...] = ()
    terminal_reason: str | None = None
    capabilities: frozenset[str] = frozenset()
    stale_event_count: int = 0
    duplicate_terminal_count: int = 0

    @classmethod
    def open(cls, session_id: str, *, capabilities: frozenset[str] = frozenset()) -> ContinuousDuplexState:
        return cls(fence=DuplexFence(session_id), capabilities=capabilities)


def _transition_error(state: ContinuousDuplexState, event: DomainEvent, detail: str) -> ContinuousTransitionError:
    return ContinuousTransitionError(state, event, detail)


def _validate_epoch(state: ContinuousDuplexState, event: _FENCED_MODEL_EVENT) -> bool:
    """Return True if the event is current, False if stale; raise on mismatch."""
    fence = event.fence
    if fence is None:
        raise DuplexMissingFenceError(event)  # type: ignore[arg-type]
    if fence.session_id != state.fence.session_id:
        raise DuplexFenceMismatchError(state.fence, fence)
    if fence.epoch == state.fence.epoch:
        return True
    if fence.epoch < state.fence.epoch:
        return False
    raise DuplexFenceMismatchError(state.fence, fence)


def _close_state(state: ContinuousDuplexState, *, reason: str) -> ContinuousDuplexState:
    return replace(
        state,
        session_phase=DuplexSessionPhase.CLOSED,
        speaking=False,
        response_fence=None,
        terminal_reason=reason,
    )


def _close_effects(state: ContinuousDuplexState) -> tuple[DuplexEffect, ...]:
    effects: list[DuplexEffect] = []
    if state.speaking:
        effects.extend((CancelFence(state.fence), ResetStage1(state.fence)))
    effects.append(CloseSessionResources(state.fence))
    return tuple(effects)


def _open_response(state: ContinuousDuplexState) -> tuple[ContinuousDuplexState, DuplexFence]:
    counter = state.response_counter + 1
    response_fence = replace(state.fence, turn_id=counter, response_seq=counter)
    return (
        replace(
            state,
            speaking=True,
            response_fence=response_fence,
            response_counter=counter,
            playback=PlaybackCursor(),
        ),
        response_fence,
    )


def _end_response(state: ContinuousDuplexState) -> tuple[ContinuousDuplexState, tuple[DuplexEffect, ...]]:
    """Close the live response and emit completion + Stage1 reset."""
    response_fence = state.response_fence
    if response_fence is None:
        return state, ()
    effects = (
        EmitProtocolEvent(response_fence, ProtocolEventKind.RESPONSE_COMPLETED, payload=None),
        ResetStage1(state.fence),
    )
    return replace(state, speaking=False, response_fence=None), effects


def reduce_continuous_event(
    state: ContinuousDuplexState,
    event: DomainEvent,
) -> tuple[ContinuousDuplexState, tuple[DuplexEffect, ...]]:
    if isinstance(event, SessionCloseRequested):
        if state.session_phase is DuplexSessionPhase.CLOSED:
            return replace(state, duplicate_terminal_count=state.duplicate_terminal_count + 1), ()
        return _close_state(state, reason=event.reason), _close_effects(state)

    if isinstance(event, EngineFailed):
        if not _validate_epoch(state, event):
            return replace(state, stale_event_count=state.stale_event_count + 1), ()
        if state.session_phase is DuplexSessionPhase.CLOSED:
            return replace(state, duplicate_terminal_count=state.duplicate_terminal_count + 1), ()
        effects = (
            EmitProtocolEvent(
                state.response_fence or state.fence, ProtocolEventKind.ENGINE_FAILED, payload=event.message
            ),
            *_close_effects(state),
        )
        return _close_state(state, reason=event.message), effects

    if state.session_phase is not DuplexSessionPhase.OPEN:
        raise _transition_error(state, event, "session is not open")

    # Client input: always accepted, streamed straight to the engine. Overlap
    # with an active response is expected (full duplex) and intentional.
    if isinstance(event, InputStarted):
        return state, ()
    if isinstance(event, InputChunk):
        return state, (AppendToEngine(state.fence, chunk=event),)
    if isinstance(event, InputCommitted):
        # Continuous duplex has no client-driven turn commit; accept and ignore
        # so a client that still sends it does not break the stream.
        return state, ()

    if isinstance(event, InterruptRequested):
        if not _validate_epoch(state, event):
            return replace(state, stale_event_count=state.stale_event_count + 1), ()
        old_fence = state.fence
        new_fence = old_fence.next_epoch()
        committed_position = max(state.last_committed_playback_position, state.playback.committed)
        effects: tuple[DuplexEffect, ...] = (
            CancelFence(old_fence),
            ResetStage1(old_fence),
            RebuildStage0Context(
                new_fence,
                committed_history=state.committed_history,
                committed_playback_position=committed_position,
            ),
        )
        return (
            replace(
                state,
                fence=new_fence,
                speaking=False,
                response_fence=None,
                playback=PlaybackCursor(),
                last_committed_playback_position=committed_position,
            ),
            effects,
        )

    if isinstance(
        event,
        (
            EngineAppendAccepted,
            HistoryCommitted,
            ModelListening,
            ModelSpeaking,
            ModelTextDelta,
            ModelAudioDelta,
            ModelSegmentEnded,
            ModelTurnEnded,
            PlaybackAcknowledged,
        ),
    ):
        if not _validate_epoch(state, event):
            return replace(state, stale_event_count=state.stale_event_count + 1), ()

    if isinstance(event, EngineAppendAccepted):
        # No commit gate in continuous mode; append acceptance carries no phase.
        return state, ()

    if isinstance(event, HistoryCommitted):
        return replace(state, committed_history=(*state.committed_history, event.item)), ()

    if isinstance(event, ModelSpeaking):
        if state.speaking:
            return state, ()
        next_state, response_fence = _open_response(state)
        return next_state, (EmitProtocolEvent(response_fence, ProtocolEventKind.RESPONSE_STARTED, payload=event),)

    if isinstance(event, ModelTextDelta):
        if not state.speaking or state.response_fence is None:
            raise _transition_error(state, event, "text arrived while the model was not speaking")
        return state, (EmitProtocolEvent(state.response_fence, ProtocolEventKind.TEXT_DELTA, payload=event),)

    if isinstance(event, ModelAudioDelta):
        if not state.speaking or state.response_fence is None:
            raise _transition_error(state, event, "audio arrived while the model was not speaking")
        # Cursor advances by the audio length itself (float32 = 4 bytes/sample),
        # epoch-cumulative and independent of any adapter cursor scoping.
        samples = len(event.data) // 4
        cursor = state.playback.generated + samples
        try:
            playback = state.playback.mark_generated(cursor).mark_sent(cursor)
        except PlaybackCursorError as exc:
            raise _transition_error(state, event, str(exc)) from exc
        return replace(state, playback=playback), (
            EmitProtocolEvent(state.response_fence, ProtocolEventKind.AUDIO_DELTA, payload=event),
        )

    if isinstance(event, ModelSegmentEnded):
        if not state.speaking or state.response_fence is None:
            raise _transition_error(state, event, "segment ended while the model was not speaking")
        return state, (EmitProtocolEvent(state.response_fence, ProtocolEventKind.SEGMENT_ENDED, payload=event),)

    if isinstance(event, ModelListening):
        # The model chose to listen. If a response was live, this ends it (the
        # reference resets the active response on a listen decision). An idle
        # listen (no active response) is surfaced without changing phase.
        if state.speaking:
            listening_fence = state.response_fence or state.fence
            ended_state, end_effects = _end_response(state)
            return ended_state, (
                EmitProtocolEvent(listening_fence, ProtocolEventKind.MODEL_LISTENING, payload=event),
                *end_effects,
            )
        return state, (EmitProtocolEvent(state.fence, ProtocolEventKind.MODEL_LISTENING, payload=event),)

    if isinstance(event, ModelTurnEnded):
        if not state.speaking or state.response_fence is None:
            # A terminal for a response that already ended is a benign duplicate.
            return replace(state, duplicate_terminal_count=state.duplicate_terminal_count + 1), ()
        return _end_response(state)

    if isinstance(event, PlaybackAcknowledged):
        try:
            playback = state.playback.acknowledge(played=event.cursor, committed=event.committed_cursor)
        except PlaybackCursorError as exc:
            raise _transition_error(state, event, str(exc)) from exc
        return replace(
            state,
            playback=playback,
            last_committed_playback_position=max(state.last_committed_playback_position, playback.committed),
        ), ()

    raise TypeError(f"unsupported continuous duplex event: {type(event).__name__}")


__all__ = [
    "ContinuousDuplexState",
    "ContinuousTransitionError",
    "DuplexReducerError",
    "reduce_continuous_event",
]
