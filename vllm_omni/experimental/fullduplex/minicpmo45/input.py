from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MiniCPMO45CommittedInput:
    payload: dict[str, object] | None
    had_input: bool
    had_speech: bool


class MiniCPMO45PcmAppendBuffer:
    """Accumulates short native-duplex PCM chunks into model-sized appends."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._sample_rate_hz: int | None = None
        self._force_listen = False
        self._is_speech = False
        self._new_user_turn = False
        self._turn_had_input = False
        self._turn_had_speech = False

    def clear(self) -> None:
        self._buffer.clear()
        self._sample_rate_hz = None
        self._force_listen = False
        self._is_speech = False
        self._new_user_turn = False
        self._turn_had_input = False
        self._turn_had_speech = False

    def clear_force_listen(self) -> None:
        self._force_listen = False

    def has_pending(self) -> bool:
        return bool(self._buffer)

    def append(
        self,
        payload: dict[str, object],
        *,
        chunk_period_ms: int,
        flush: bool = False,
        allow_emit: bool = True,
    ) -> dict[str, object] | None:
        fmt = payload.get("format")
        sample_rate_hz = payload.get("sample_rate_hz")
        audio = payload.get("audio")
        if fmt != "pcm_f32le" or not isinstance(sample_rate_hz, int) or not isinstance(audio, str):
            return payload
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            return payload
        if len(raw) % 4 != 0:
            return payload

        if self._sample_rate_hz is not None and self._sample_rate_hz != sample_rate_hz:
            raise ValueError("MiniCPM-o native duplex audio append sample_rate_hz changed within a session")
        self._sample_rate_hz = sample_rate_hz
        self._buffer.extend(raw)
        self._turn_had_input = self._turn_had_input or bool(raw)
        self._turn_had_speech = self._turn_had_speech or bool(payload.get("is_speech", False))
        self._force_listen = self._force_listen or bool(payload.get("force_listen", False))
        self._is_speech = self._is_speech or bool(payload.get("is_speech", False))
        self._new_user_turn = self._new_user_turn or bool(payload.get("new_user_turn", False))
        if not allow_emit:
            return None

        min_samples = max(1, int(sample_rate_hz * max(1, int(chunk_period_ms)) / 1000))
        buffered_samples = len(self._buffer) // 4
        if not flush and buffered_samples < min_samples:
            return None

        # Emit whole model chunks only: the engine reserves scheduler slots
        # from the payload size and the worker consumes whole chunks per
        # append, so partial chunks would turn into pad embeddings inside the
        # model KV. On flush, the tail is zero-padded (silence) up to the
        # chunk boundary instead.
        if flush:
            emit_samples = buffered_samples
            remainder = emit_samples % min_samples
            pad_samples = (min_samples - remainder) if remainder else 0
        else:
            emit_samples = buffered_samples - (buffered_samples % min_samples)
            pad_samples = 0
        emit_bytes = emit_samples * 4
        emit_raw = bytes(self._buffer[:emit_bytes]) + b"\x00" * (pad_samples * 4)
        del self._buffer[:emit_bytes]

        out = dict(payload)
        out.pop("force_speak", None)
        out["audio"] = base64.b64encode(emit_raw).decode("ascii")
        out["sample_rate_hz"] = sample_rate_hz
        out["force_listen"] = self._force_listen
        out["is_speech"] = self._is_speech
        if self._new_user_turn:
            out["new_user_turn"] = True
        if not self._buffer:
            self._force_listen = False
            self._is_speech = False
            self._new_user_turn = False
        return out

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        if not self._buffer:
            return None
        payload: dict[str, object] = {
            "type": "audio",
            "audio": "",
            "format": "pcm_f32le",
            "sample_rate_hz": self._sample_rate_hz or 16000,
            "force_listen": self._force_listen,
            "is_speech": self._is_speech,
        }
        if self._new_user_turn:
            payload["new_user_turn"] = True
        return self.append(payload, chunk_period_ms=chunk_period_ms, flush=True)

    def commit(self, *, chunk_period_ms: int) -> MiniCPMO45CommittedInput:
        """Commit the real residual PCM and reset per-client-input accounting.

        Complete model units are emitted by :meth:`append` as soon as they are
        available.  A commit at that exact boundary therefore has no payload;
        synthesizing another unit would add a model decision that does not
        exist in the official continuous streaming loop.
        """
        had_input = self._turn_had_input
        had_speech = self._turn_had_speech
        payload = self.flush(chunk_period_ms=chunk_period_ms) if had_speech else None
        if payload is not None:
            payload["final"] = True
        self.clear()
        return MiniCPMO45CommittedInput(
            payload=payload,
            had_input=had_input,
            had_speech=had_speech,
        )


__all__ = ["MiniCPMO45CommittedInput", "MiniCPMO45PcmAppendBuffer"]
