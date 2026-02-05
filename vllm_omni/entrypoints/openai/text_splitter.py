"""Multi-language sentence boundary detector for streaming TTS input.

Buffers incoming text and splits at sentence boundaries (English and CJK),
yielding complete sentences for audio generation.
"""

import re

# Sentence boundary pattern:
# - English: .!? followed by whitespace or end of string
# - CJK fullwidth: 。！？，；
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])\s+"  # English punctuation followed by whitespace
    r"|(?<=[。！？，；])"  # CJK fullwidth punctuation
)


class SentenceSplitter:
    """Incremental sentence splitter for streaming text input.

    Buffers text and yields complete sentences when boundaries are detected.
    Designed for TTS pipelines where text arrives incrementally (e.g., from STT).

    Args:
        min_sentence_length: Minimum character length for a sentence.
            Sentences shorter than this are kept in the buffer to avoid
            splitting on abbreviations like "Dr." or "U.S.".
    """

    def __init__(self, min_sentence_length: int = 2) -> None:
        self._buffer: str = ""
        self._min_sentence_length = min_sentence_length

    @property
    def buffer(self) -> str:
        """Current buffered text."""
        return self._buffer

    def add_text(self, text: str) -> list[str]:
        """Add text to the buffer and return any complete sentences.

        Args:
            text: Incoming text chunk.

        Returns:
            List of complete sentences extracted from the buffer.
            May be empty if no sentence boundary was found.
        """
        if not text:
            return []

        self._buffer += text
        return self._extract_sentences()

    def flush(self) -> str | None:
        """Flush remaining buffered text as a final sentence.

        Returns:
            The remaining buffered text (stripped), or None if buffer is empty.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None

    def _extract_sentences(self) -> list[str]:
        """Split buffer at sentence boundaries, keeping incomplete text buffered."""
        parts = _SENTENCE_BOUNDARY_RE.split(self._buffer)

        if len(parts) <= 1:
            # No boundary found — keep everything in buffer
            return []

        sentences: list[str] = []
        carry = ""
        # All parts except the last are complete sentences
        for i in range(len(parts) - 1):
            text = carry + parts[i]
            carry = ""
            stripped = text.strip()
            if len(stripped) >= self._min_sentence_length:
                sentences.append(stripped)
            elif stripped:
                # Too short (e.g. "Dr.") — carry forward to next part
                carry = text
            # else: empty, skip

        # Last part stays in buffer (may be incomplete)
        self._buffer = carry + parts[-1]

        return sentences
