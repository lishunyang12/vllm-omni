"""End-to-end MiniCPM-o 4.5 Realtime duplex demo client.

This script is intentionally scenario-based instead of a generic chat client.
It validates the full-duplex semantics implemented by vLLM-Omni:

1. normal audio input -> automatic audio response -> response.done,
2. the next user turn starts only after the previous response.done,
3. repeated clean turns keep producing independent automatic responses.

Explicit serving-side barge-in is intentionally not part of this smoke test.
MiniCPM-o native duplex currently exposes model-owned listen/speak switching,
not a separate model-level barge-in contract.

Run only after a MiniCPM-o 4.5 vLLM-Omni server is up:

  python examples/online_serving/minicpmo/realtime_duplex_demo.py \
      --url ws://localhost:8099/v1/realtime?duplex=1 \
      --model openbmb/MiniCPM-o-4_5 \
      --input-wav input_16k_mono_pcm16.wav \
      --output-dir /tmp/minicpmo_duplex_demo
"""

from __future__ import annotations

import argparse
import array
import asyncio
import base64
import hashlib
import json
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError as exc:  # pragma: no cover - demo dependency.
    raise SystemExit("Install websockets first: pip install websockets") from exc


PCM16_SAMPLE_RATE = 16000
PCM16_BYTES_PER_SAMPLE = 2


@dataclass
class DemoState:
    events: list[dict[str, object]] = field(default_factory=list)
    audio_deltas: list[bytes] = field(default_factory=list)
    response_audio_deltas: dict[str, list[bytes]] = field(default_factory=dict)
    response_ids: list[str] = field(default_factory=list)
    assistant_item_ids: list[str] = field(default_factory=list)
    done_count: int = 0
    cancelled_count: int = 0
    listen_count: int = 0
    model_listen_count: int = 0
    buffering_listen_count: int = 0
    model_speak_event_count: int = 0
    model_speak_delta_count: int = 0
    playback_ack_count: int = 0
    playback_history_committed_count: int = 0
    truncate_count: int = 0
    input_transcription_count: int = 0
    audio_marks_seen: bool = False
    overlap_decisions: list[dict[str, object]] = field(default_factory=list)
    output_sample_rate_hz: int = 24000

    def add(self, event: dict[str, object]) -> None:
        self.events.append(event)
        event_type = event.get("type")
        if event_type == "response.created":
            response = event.get("response")
            response_id = response.get("id") if isinstance(response, dict) else event.get("response_id")
            if isinstance(response_id, str) and response_id not in self.response_ids:
                self.response_ids.append(response_id)
        elif event_type == "conversation.item.added":
            item = event.get("item")
            if isinstance(item, dict) and item.get("role") == "assistant":
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id not in self.assistant_item_ids:
                    self.assistant_item_ids.append(item_id)
        elif event_type == "response.audio.delta":
            delta = event.get("delta") or event.get("audio")
            if isinstance(delta, str) and delta:
                try:
                    decoded = base64.b64decode(delta)
                    self.audio_deltas.append(decoded)
                    response_id = self._event_response_id(event)
                    if isinstance(response_id, str):
                        self.response_audio_deltas.setdefault(response_id, []).append(decoded)
                except Exception:
                    pass
            metadata = event.get("metadata")
            if isinstance(metadata, dict):
                if metadata.get("model_speak") is True:
                    self.model_speak_delta_count += 1
                if isinstance(metadata.get("audio_text_marks"), list):
                    self.audio_marks_seen = True
            sample_rate_hz = event.get("sample_rate_hz")
            if isinstance(sample_rate_hz, int) and sample_rate_hz > 0:
                self.output_sample_rate_hz = sample_rate_hz
        elif event_type == "response.done":
            self.done_count += 1
            response = event.get("response")
            if isinstance(response, dict) and response.get("status") == "cancelled":
                self.cancelled_count += 1
        elif event_type in {"audio.cancelled", "input.cancelled"}:
            self.cancelled_count += 1
        elif event_type == "response.listen":
            self.listen_count += 1
            response = event.get("response")
            metadata = response.get("metadata") if isinstance(response, dict) else None
            if isinstance(metadata, dict) and metadata.get("model_listen") is True:
                self.model_listen_count += 1
            if isinstance(metadata, dict) and metadata.get("buffering") is True:
                self.buffering_listen_count += 1
        elif event_type == "response.speak":
            self.model_speak_event_count += 1
        elif event_type == "overlap.decision":
            self.overlap_decisions.append(event)
        elif event_type == "playback.acknowledged":
            self.playback_ack_count += 1
            payload = event.get("event")
            if isinstance(payload, dict) and payload.get("history_committed") is True:
                self.playback_history_committed_count += 1
        elif event_type == "conversation.item.truncated":
            self.truncate_count += 1
        elif event_type == "conversation.item.input_audio_transcription.completed":
            self.input_transcription_count += 1

    def count(self, event_type: str) -> int:
        return sum(1 for event in self.events if event.get("type") == event_type)

    def first_index(self, event_type: str, predicate=None) -> int | None:
        for index, event in enumerate(self.events):
            if event.get("type") != event_type:
                continue
            if predicate is not None and not predicate(event):
                continue
            return index
        return None

    @staticmethod
    def _event_response_id(event: dict[str, object]) -> str | None:
        response_id = event.get("response_id")
        if isinstance(response_id, str) and response_id:
            return response_id
        response = event.get("response")
        if isinstance(response, dict):
            response_id = response.get("id")
            if isinstance(response_id, str) and response_id:
                return response_id
        return None

    @staticmethod
    def _event_item_id(event: dict[str, object]) -> str | None:
        item_id = event.get("item_id")
        if isinstance(item_id, str) and item_id:
            return item_id
        item = event.get("item")
        if isinstance(item, dict):
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id:
                return item_id
        return None

    def first_response_lifecycle_indices(self) -> dict[str, int]:
        response_created_index = self.first_index("response.created")
        if response_created_index is None:
            return {}
        response_id = self._event_response_id(self.events[response_created_index])
        if not response_id:
            return {}
        item_id = f"item_{response_id}"
        indices: dict[str, int] = {"response.created": response_created_index}
        for event_type in (
            "conversation.item.added",
            "response.output_item.added",
            "response.content_part.added",
            "response.speak",
            "response.audio.delta",
            "response.audio.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.done",
        ):
            index = self.first_index(
                event_type,
                lambda event, event_type=event_type: (
                    self._event_item_id(event) == item_id
                    if event_type == "conversation.item.added"
                    else self._event_response_id(event) == response_id
                ),
            )
            if index is None and event_type not in {
                "response.speak",
                "response.audio.delta",
                "response.audio.done",
            }:
                return {}
            if index is not None:
                indices[event_type] = index
        return indices

    def event_order_ok(self) -> bool:
        if not self.events or self.events[0].get("type") != "session.created":
            return False
        first_commit_index = self.first_index("input_audio_buffer.committed")
        first_response_index = self.first_index("response.created")
        if first_commit_index is None or first_response_index is None or first_commit_index > first_response_index:
            return False
        indices_by_type = self.first_response_lifecycle_indices()
        if not indices_by_type:
            return False
        required_types = [
            "response.created",
            "conversation.item.added",
            "response.output_item.added",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_item.done",
            "response.done",
        ]
        if any(event_type not in indices_by_type for event_type in required_types):
            return False
        ordered_types = [
            "response.created",
            "conversation.item.added",
            "response.output_item.added",
            "response.content_part.added",
            "response.speak",
            "response.audio.delta",
            "response.audio.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.done",
        ]
        if "response.audio.delta" not in indices_by_type:
            listen_index = self.first_index(
                "response.listen",
                lambda event: (
                    isinstance(event.get("response"), dict)
                    and isinstance(event["response"].get("metadata"), dict)
                    and event["response"]["metadata"].get("model_listen") is True
                ),
            )
            return (
                listen_index is not None
                and indices_by_type["response.created"] < listen_index < indices_by_type["response.done"]
            )
        if any(event_type not in indices_by_type for event_type in ordered_types):
            return False
        indices = [indices_by_type[event_type] for event_type in ordered_types]
        return indices == sorted(indices)

    def model_speak_before_audio_ok(self) -> bool:
        speak_index = self.first_index("response.speak")
        audio_index = self.first_index("response.audio.delta")
        return speak_index is not None and audio_index is not None and speak_index < audio_index

    def response_done(self, response_id: str | None) -> bool:
        if not response_id:
            return False
        return any(
            event.get("type") == "response.done" and self._event_response_id(event) == response_id
            for event in self.events
        )

    def response_audio_delta_count(self, response_id: str | None) -> int:
        if not response_id:
            return 0
        return sum(
            1
            for event in self.events
            if event.get("type") == "response.audio.delta" and self._event_response_id(event) == response_id
        )

    def response_playback_sent_ms(self, response_id: str | None) -> int:
        if not response_id:
            return 0
        for event in reversed(self.events):
            if event.get("type") != "response.done" or self._event_response_id(event) != response_id:
                continue
            response = event.get("response")
            metadata = response.get("metadata") if isinstance(response, dict) else None
            playback = metadata.get("playback") if isinstance(metadata, dict) else event.get("playback")
            sent_ms = playback.get("sent_ms") if isinstance(playback, dict) else None
            if isinstance(sent_ms, int | float):
                return max(0, int(sent_ms))
        return 0

    def response_transcript_delta(self, response_id: str) -> str:
        return "".join(
            str(event.get("delta", ""))
            for event in self.events
            if event.get("type") == "response.audio_transcript.delta" and self._event_response_id(event) == response_id
        )

    def response_transcript_done(self, response_id: str) -> list[str]:
        return [
            str(event.get("transcript", ""))
            for event in self.events
            if event.get("type") == "response.audio_transcript.done" and self._event_response_id(event) == response_id
        ]

    def completed_response_ids(self) -> list[str]:
        response_ids: list[str] = []
        for event in self.events:
            if event.get("type") != "response.done":
                continue
            response_id = self._event_response_id(event)
            if isinstance(response_id, str):
                response_ids.append(response_id)
        return response_ids

    def stale_audio_delta_count(self) -> int:
        cancelled_epochs_by_index: list[tuple[int, int]] = []
        for index, event in enumerate(self.events):
            if event.get("type") != "response.done":
                continue
            response = event.get("response")
            if not isinstance(response, dict) or response.get("status") != "cancelled":
                continue
            metadata = response.get("metadata")
            if not isinstance(metadata, dict):
                continue
            cancelled_epoch = metadata.get("cancelled_epoch")
            if isinstance(cancelled_epoch, int):
                cancelled_epochs_by_index.append((index, cancelled_epoch))
        if not cancelled_epochs_by_index:
            return 0
        stale = 0
        for index, event in enumerate(self.events):
            if event.get("type") != "response.audio.delta":
                continue
            metadata = event.get("metadata")
            if not isinstance(metadata, dict):
                continue
            event_epoch = metadata.get("epoch")
            for cancel_index, cancelled_epoch in cancelled_epochs_by_index:
                if index > cancel_index and event_epoch == cancelled_epoch:
                    stale += 1
                    break
        return stale


def _url_with_model(url: str, model: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("duplex", "1")
    query.setdefault("model", model)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _read_wav_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("input WAV must be mono")
        if wf.getsampwidth() != PCM16_BYTES_PER_SAMPLE:
            raise ValueError("input WAV must be 16-bit PCM")
        if wf.getframerate() != PCM16_SAMPLE_RATE:
            raise ValueError("input WAV must be 16 kHz")
        if wf.getcomptype() != "NONE":
            raise ValueError("input WAV must be uncompressed PCM")
        return wf.readframes(wf.getnframes())


def _turn_input_paths(primary: Path, additional: list[str], *, turns: int) -> list[Path]:
    turn_count = max(1, turns)
    if not additional:
        return [primary] * turn_count
    if len(additional) != turn_count - 1:
        raise ValueError(
            f"provide exactly one --turn-input-wav for each turn after the first "
            f"(expected {turn_count - 1}, got {len(additional)})"
        )
    return [primary, *(Path(path) for path in additional)]


def _turn_inputs_are_distinct(paths: list[Path], pcm16_inputs: list[bytes]) -> bool:
    if len(paths) != len(pcm16_inputs):
        return False
    distinct_paths = len({str(path.resolve()) for path in paths}) == len(paths)
    distinct_audio = len({hashlib.sha256(pcm16).digest() for pcm16 in pcm16_inputs}) == len(pcm16_inputs)
    return distinct_paths and distinct_audio


def _turn_durations(
    explicit: list[int],
    *,
    turns: int,
    first_turn_ms: int,
) -> list[int | None]:
    turn_count = max(1, turns)
    if not explicit:
        return [first_turn_ms, *([min(first_turn_ms, 1200)] * (turn_count - 1))]
    if len(explicit) != turn_count:
        raise ValueError(
            f"provide exactly one --turn-duration-ms for each turn (expected {turn_count}, got {len(explicit)})"
        )
    if any(duration_ms < 0 for duration_ms in explicit):
        raise ValueError("--turn-duration-ms values must be non-negative")
    return [None if duration_ms == 0 else duration_ms for duration_ms in explicit]


def _turn_transcripts(first: str, *, turns: int) -> list[str]:
    turn_count = max(1, turns)
    transcripts = [first, "继续", "再说一次"][:turn_count]
    transcripts.extend(f"turn-{turn_index + 1}" for turn_index in range(len(transcripts), turn_count))
    return transcripts


def _write_wav(path: Path, pcm_bytes: bytes, *, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(PCM16_BYTES_PER_SAMPLE)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm_bytes)


def _write_demo_artifacts(state: DemoState, output_dir: Path, *, output_audio_format: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if state.audio_deltas and output_audio_format == "pcm16":
        _write_wav(
            output_dir / "joined_audio_deltas.wav",
            b"".join(state.audio_deltas),
            sample_rate_hz=state.output_sample_rate_hz,
        )
    elif state.audio_deltas:
        (output_dir / "joined_audio_deltas.bin").write_bytes(b"".join(state.audio_deltas))
    for index, response_id in enumerate(state.response_ids, start=1):
        response_audio = state.response_audio_deltas.get(response_id, [])
        if not response_audio:
            continue
        payload = b"".join(response_audio)
        if output_audio_format == "pcm16":
            _write_wav(
                output_dir / f"response_{index:02d}.wav",
                payload,
                sample_rate_hz=state.output_sample_rate_hz,
            )
        else:
            (output_dir / f"response_{index:02d}.bin").write_bytes(payload)
    (output_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in state.events) + "\n",
        encoding="utf-8",
    )


def _pcm16_silence(duration_ms: int) -> bytes:
    samples = PCM16_SAMPLE_RATE * max(0, duration_ms) // 1000
    return b"\x00\x00" * samples


def _pcm16_slice(pcm16: bytes, duration_ms: int) -> bytes:
    byte_count = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * max(1, duration_ms) // 1000
    return pcm16[: min(len(pcm16), byte_count)]


def _pcm16_active_slice(pcm16: bytes, duration_ms: int) -> bytes:
    byte_count = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * max(1, duration_ms) // 1000
    byte_count = min(len(pcm16), max(PCM16_BYTES_PER_SAMPLE, byte_count))
    byte_count -= byte_count % PCM16_BYTES_PER_SAMPLE
    if byte_count <= 0:
        return _pcm16_slice(pcm16, duration_ms)
    step = max(PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * 20 // 1000, PCM16_BYTES_PER_SAMPLE)
    step -= step % PCM16_BYTES_PER_SAMPLE
    best_offset = 0
    best_energy = -1.0
    for offset in range(0, max(1, len(pcm16) - byte_count + 1), max(PCM16_BYTES_PER_SAMPLE, step)):
        chunk = pcm16[offset : offset + byte_count]
        samples = array.array("h")
        samples.frombytes(chunk)
        if not samples:
            continue
        energy = sum(abs(sample) for sample in samples) / len(samples)
        if energy > best_energy:
            best_energy = energy
            best_offset = offset
    return pcm16[best_offset : best_offset + byte_count]


def _select_turn_audio(pcm16: bytes, duration_ms: int | None) -> bytes:
    if duration_ms is None:
        return pcm16
    return _pcm16_active_slice(pcm16, duration_ms)


def _canonical_transcript(text: str) -> str:
    return "".join(text.split())


def _reuses_previous_turn_tail(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if len(previous) >= 4 and previous in current:
        return True
    max_overlap = min(len(previous), len(current))
    return any(previous.endswith(current[:overlap]) for overlap in range(max_overlap, 2, -1))


def _has_terminal_punctuation(text: str) -> bool:
    stripped = text.rstrip("\"'”’）)]} ")
    return bool(stripped) and stripped[-1] in "。！？!?…"


def _evaluate_transcript_integrity(
    state: DemoState,
    response_ids: list[str],
    *,
    expected_empty_response_ids: set[str],
    require_cross_turn_independence: bool,
    require_terminal_punctuation: bool = False,
) -> dict[str, object]:
    details: list[dict[str, object]] = []
    transcripts: list[str] = []
    transcript_delta_done_ok = True
    empty_turns_ok = True
    nonempty_audio_has_transcript_ok = True
    terminal_punctuation_ok = True

    for response_id in response_ids:
        transcript = state.response_transcript_delta(response_id)
        done_transcripts = state.response_transcript_done(response_id)
        canonical_transcript = _canonical_transcript(transcript)
        response_delta_done = (
            len(done_transcripts) == 1 and _canonical_transcript(done_transcripts[0]) == canonical_transcript
        ) or (not done_transcripts and not canonical_transcript)
        expected_empty = response_id in expected_empty_response_ids
        audio_delta_count = state.response_audio_delta_count(response_id)
        response_empty_ok = not expected_empty or (not canonical_transcript and audio_delta_count == 0)
        response_audio_has_transcript = expected_empty or audio_delta_count == 0 or bool(canonical_transcript)
        response_terminal_punctuation_ok = (
            expected_empty
            or not canonical_transcript
            or not require_terminal_punctuation
            or _has_terminal_punctuation(canonical_transcript)
        )
        transcript_delta_done_ok = transcript_delta_done_ok and response_delta_done
        empty_turns_ok = empty_turns_ok and response_empty_ok
        nonempty_audio_has_transcript_ok = nonempty_audio_has_transcript_ok and response_audio_has_transcript
        terminal_punctuation_ok = terminal_punctuation_ok and response_terminal_punctuation_ok
        transcripts.append(canonical_transcript)
        details.append(
            {
                "response_id": response_id,
                "transcript": transcript,
                "delta_done_ok": response_delta_done,
                "expected_empty": expected_empty,
                "empty_ok": response_empty_ok,
                "audio_delta_count": audio_delta_count,
                "audio_has_transcript": response_audio_has_transcript,
                "terminal_punctuation_ok": response_terminal_punctuation_ok,
            }
        )

    cross_turn_independent_ok = True
    if require_cross_turn_independence:
        for index, current in enumerate(transcripts):
            for previous in transcripts[:index]:
                if _reuses_previous_turn_tail(previous, current):
                    cross_turn_independent_ok = False
                    break
            if not cross_turn_independent_ok:
                break

    return {
        "transcript_delta_done_ok": transcript_delta_done_ok,
        "cross_turn_independent_ok": cross_turn_independent_ok,
        "empty_turns_ok": empty_turns_ok,
        "nonempty_audio_has_transcript_ok": nonempty_audio_has_transcript_ok,
        "terminal_punctuation_ok": terminal_punctuation_ok,
        "transcript_integrity": details,
    }


async def _reader(ws, state: DemoState, stop: asyncio.Event) -> None:
    try:
        while not stop.is_set():
            raw = await ws.recv()
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            if isinstance(event, dict):
                state.add(event)
    except ConnectionClosed:
        return


async def _send_pcm16(
    ws,
    pcm16: bytes,
    *,
    chunk_ms: int,
    realtime_delay: bool,
    hints: dict[str, object] | None = None,
    first_chunk_hints: dict[str, object] | None = None,
) -> None:
    hints = hints or {}
    first_chunk_hints = first_chunk_hints or {}
    chunk_bytes = max(PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * chunk_ms // 1000, PCM16_BYTES_PER_SAMPLE)
    audio_ms = 0
    for offset in range(0, len(pcm16), chunk_bytes):
        chunk = pcm16[offset : offset + chunk_bytes]
        duration_ms = int(len(chunk) / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE) * 1000)
        audio_ms += duration_ms
        chunk_hints = dict(hints)
        if offset == 0:
            chunk_hints.update(first_chunk_hints)
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                    "input_audio_format": "pcm16",
                    "sample_rate_hz": PCM16_SAMPLE_RATE,
                    "duration_ms": duration_ms,
                    "audio_end_ms": audio_ms,
                    **chunk_hints,
                }
            )
        )
        if realtime_delay:
            await asyncio.sleep(duration_ms / 1000)


async def _wait_for(
    state: DemoState,
    predicate,
    *,
    timeout_s: float,
    label: str,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for {label}")


async def _send_clean_turn(
    ws,
    state: DemoState,
    pcm16: bytes,
    *,
    transcript: str,
    duration_ms: int | None,
    chunk_ms: int,
    timeout_s: float,
    require_audio: bool,
) -> str:
    before_created = state.count("response.created")
    await _send_pcm16(
        ws,
        _select_turn_audio(pcm16, duration_ms),
        chunk_ms=chunk_ms,
        realtime_delay=False,
        hints={"transcript": transcript},
    )
    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
    await _wait_for(
        state,
        lambda: state.count("response.created") > before_created,
        timeout_s=timeout_s,
        label=f"{transcript} response.created",
    )
    response_id = state.response_ids[-1] if state.response_ids else None
    if require_audio:
        await _wait_for(
            state,
            lambda: state.response_audio_delta_count(response_id) > 0,
            timeout_s=timeout_s,
            label=f"{transcript} response.audio.delta",
        )
    await _wait_for(
        state,
        lambda: state.response_done(response_id),
        timeout_s=timeout_s,
        label=f"{transcript} response.done",
    )
    if not isinstance(response_id, str):
        raise TimeoutError(f"Missing response id for {transcript}")
    played_ms = state.response_playback_sent_ms(response_id)
    if played_ms > 0:
        before_ack = state.playback_ack_count
        await ws.send(
            json.dumps(
                {
                    "type": "playback.ack",
                    "played_ms": played_ms,
                    "committed_ms": played_ms,
                }
            )
        )
        await _wait_for(
            state,
            lambda: state.playback_ack_count > before_ack,
            timeout_s=timeout_s,
            label=f"{transcript} playback.acknowledged",
        )
    return response_id


async def run_demo(args: argparse.Namespace) -> dict[str, object]:
    turn_input_paths = _turn_input_paths(
        Path(args.input_wav),
        list(getattr(args, "turn_input_wav", []) or []),
        turns=args.turns,
    )
    turn_pcm16 = [_read_wav_pcm16(path) for path in turn_input_paths]
    if any(not pcm16 for pcm16 in turn_pcm16):
        raise ValueError("input WAV has no audio")
    turn_durations = _turn_durations(
        list(getattr(args, "turn_duration_ms", []) or []),
        turns=args.turns,
        first_turn_ms=args.first_turn_ms,
    )
    expected_empty_turns = set(getattr(args, "expect_empty_turn", []) or [])
    invalid_empty_turns = sorted(
        turn_number for turn_number in expected_empty_turns if turn_number < 1 or turn_number > max(1, args.turns)
    )
    if invalid_empty_turns:
        raise ValueError(
            f"--expect-empty-turn values are 1-based and must refer to an existing turn: {invalid_empty_turns}"
        )
    distinct_turn_inputs = _turn_inputs_are_distinct(turn_input_paths, turn_pcm16)
    if getattr(args, "require_distinct_inputs", False) and not distinct_turn_inputs:
        raise ValueError("--require-distinct-inputs requires a different WAV path and audio payload for every turn")
    url = _url_with_model(args.url, args.model)
    state = DemoState()
    stop = asyncio.Event()
    output_dir = Path(args.output_dir)
    turn_response_ids: list[str] = []

    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        reader = asyncio.create_task(_reader(ws, state, stop))
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "model": args.model,
                            "modalities": ["audio", "text"],
                            "input_audio_format": "pcm16",
                            "output_audio_format": args.output_audio_format,
                            "turn_detection": {
                                "type": "server_vad",
                                "interrupt_response": False,
                                "silence_duration_ms": args.short_ack_ms,
                                "threshold": 0.35,
                            },
                            "overlap_policy": "listen_only",
                            "overlap_short_ack_ms": args.short_ack_ms,
                            "playback_commit_policy": "ack_only",
                            "extra_body": {
                                "auto_response": True,
                                "force_listen_count": 0,
                            },
                        },
                    }
                )
            )
            await _wait_for(state, lambda: state.count("session.created") > 0, timeout_s=20, label="session.created")

            turn_transcripts = _turn_transcripts(args.first_turn_transcript, turns=args.turns)
            turn_specs = list(zip(turn_transcripts, turn_durations, strict=True))
            for turn_index, (transcript, duration_ms) in enumerate(turn_specs):
                response_id = await _send_clean_turn(
                    ws,
                    state,
                    turn_pcm16[turn_index],
                    transcript=transcript,
                    duration_ms=duration_ms,
                    chunk_ms=args.chunk_ms,
                    timeout_s=args.timeout_s,
                    require_audio=args.require_audio and (turn_index + 1) not in expected_empty_turns,
                )
                turn_response_ids.append(response_id)

            await ws.send(json.dumps({"type": "session.close"}))
            await _wait_for(state, lambda: state.count("session.closed") > 0, timeout_s=20, label="session.closed")
        finally:
            stop.set()
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass
            _write_demo_artifacts(state, output_dir, output_audio_format=args.output_audio_format)

    overlap_barge_in = any(decision.get("action") == "barge_in" for decision in state.overlap_decisions)
    event_order_ok = state.event_order_ok()
    input_transcription_ok = state.input_transcription_count > 0
    model_speak_event_ok = state.model_speak_before_audio_ok()
    realtime_audio_lifecycle_ok = state.count("response.audio.delta") > 0 and state.count("response.audio.done") > 0
    completed_response_ids = state.completed_response_ids()
    expected_turns = max(1, args.turns)
    expected_empty_response_ids = {
        turn_response_ids[turn_number - 1]
        for turn_number in expected_empty_turns
        if turn_number <= len(turn_response_ids)
    }
    transcript_integrity = _evaluate_transcript_integrity(
        state,
        turn_response_ids,
        expected_empty_response_ids=expected_empty_response_ids,
        require_cross_turn_independence=getattr(args, "require_distinct_inputs", False),
        require_terminal_punctuation=getattr(args, "require_distinct_inputs", False),
    )
    expected_audio_turns = expected_turns - len(expected_empty_turns)
    lifecycle_counts_ok = (
        state.count("response.created") == expected_turns
        and state.count("response.done") == expected_turns
        and state.count("response.audio.done") == expected_audio_turns
        and len(state.response_ids) == expected_turns
        and len(completed_response_ids) == expected_turns
        and len(set(completed_response_ids)) == expected_turns
    )
    clean_turn_audio_ok = all(
        state.response_audio_delta_count(response_id) > 0
        for response_id in completed_response_ids
        if response_id not in expected_empty_response_ids
    )
    full_audio_response_ok = (
        lifecycle_counts_ok
        and clean_turn_audio_ok
        and model_speak_event_ok
        and state.model_speak_delta_count > 0
        and realtime_audio_lifecycle_ok
        and state.audio_marks_seen
    )
    stale_audio_delta_count = state.stale_audio_delta_count()
    result = {
        "ok": state.count("response.done") > 0
        and state.count("session.closed") > 0
        and state.cancelled_count == 0
        and not overlap_barge_in
        and state.truncate_count == 0
        and event_order_ok
        and input_transcription_ok
        and stale_audio_delta_count == 0
        and lifecycle_counts_ok
        and full_audio_response_ok,
        "event_counts": {
            event_type: state.count(event_type)
            for event_type in sorted({str(event.get("type")) for event in state.events})
        },
        "audio_delta_count": len(state.audio_deltas),
        "done_count": state.done_count,
        "cancelled_count": state.cancelled_count,
        "listen_count": state.listen_count,
        "model_listen_count": state.model_listen_count,
        "buffering_listen_count": state.buffering_listen_count,
        "model_speak_event_count": state.model_speak_event_count,
        "model_speak_delta_count": state.model_speak_delta_count,
        "playback_ack_count": state.playback_ack_count,
        "playback_history_committed_count": state.playback_history_committed_count,
        "truncate_count": state.truncate_count,
        "input_transcription_count": state.input_transcription_count,
        "audio_marks_seen": state.audio_marks_seen,
        "overlap_decisions": state.overlap_decisions,
        "overlap_barge_in": overlap_barge_in,
        "event_order_ok": event_order_ok,
        "input_transcription_ok": input_transcription_ok,
        "completed_response_ids": completed_response_ids,
        "lifecycle_counts_ok": lifecycle_counts_ok,
        "clean_turn_audio_ok": clean_turn_audio_ok,
        "full_audio_response_ok": full_audio_response_ok,
        "model_speak_event_ok": model_speak_event_ok,
        "realtime_audio_lifecycle_ok": realtime_audio_lifecycle_ok,
        "stale_audio_delta_count": stale_audio_delta_count,
        "distinct_turn_inputs": distinct_turn_inputs,
        **transcript_integrity,
        "turn_inputs": [
            {
                "path": str(path),
                "source_duration_ms": len(pcm16) * 1000 // (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE),
                "requested_duration_ms": duration_ms,
                "sent_duration_ms": len(_select_turn_audio(pcm16, duration_ms))
                * 1000
                // (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE),
            }
            for path, pcm16, duration_ms in zip(turn_input_paths, turn_pcm16, turn_durations, strict=True)
        ],
        "output_dir": str(output_dir),
    }
    result["ok"] = bool(
        result["ok"]
        and transcript_integrity["transcript_delta_done_ok"]
        and transcript_integrity["cross_turn_independent_ok"]
        and transcript_integrity["empty_turns_ok"]
        and transcript_integrity["nonempty_audio_has_transcript_ok"]
        and transcript_integrity["terminal_punctuation_ok"]
        and (distinct_turn_inputs or not getattr(args, "require_distinct_inputs", False))
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument(
        "--turn-input-wav",
        action="append",
        default=[],
        help="WAV for each turn after the first; repeat exactly turns-1 times.",
    )
    parser.add_argument("--output-dir", default="/tmp/minicpmo_realtime_duplex_demo")
    parser.add_argument(
        "--output-audio-format",
        default="pcm16",
        choices=["pcm16", "wav", "g711_ulaw", "g711_alaw"],
    )
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--first-turn-ms", type=int, default=1400)
    parser.add_argument(
        "--turn-duration-ms",
        action="append",
        type=int,
        default=[],
        help="Audio duration for each turn; repeat turns times. Use 0 to send the complete WAV.",
    )
    parser.add_argument("--first-turn-transcript", default="demo input speech")
    parser.add_argument(
        "--require-distinct-inputs",
        action="store_true",
        help="Require distinct WAV paths and audio payloads, and reject cross-turn transcript reuse.",
    )
    parser.add_argument(
        "--expect-empty-turn",
        action="append",
        type=int,
        default=[],
        help="1-based turn expected to end without text or audio; repeat for multiple turns.",
    )
    parser.add_argument("--short-ack-ms", type=int, default=350)
    parser.add_argument("--silence-ms", type=int, default=500)
    parser.add_argument("--playback-ack-ms", type=int, default=500)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--require-audio",
        action="store_true",
        help="Fail if the first native response listens instead of producing audio.",
    )
    return parser.parse_args()


def main() -> None:
    result = asyncio.run(run_demo(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
