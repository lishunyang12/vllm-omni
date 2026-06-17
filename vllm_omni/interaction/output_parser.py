# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse the interaction model's per-tick control tokens into an action.

The model emits one of ``</silence>``, ``</response> <text>``, or
``</response> <note> <delegation> <question>``. It must be served with
``skip_special_tokens=False`` so the tokens survive into the completion;
if they are stripped, any non-empty text is treated as a response.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

SILENCE_TOKEN = "</silence>"
RESPONSE_TOKEN = "</response>"
DELEGATION_TAG = "<delegation>"


class Action(enum.Enum):
    SILENCE = "silence"
    RESPONSE = "response"
    DELEGATE = "delegate"


@dataclass
class ParsedAction:
    action: Action
    text: str = ""
    delegated_question: str | None = None
    raw: str = ""

    @property
    def spoke(self) -> bool:
        return self.action is not Action.SILENCE


def parse_action(raw: str) -> ParsedAction:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ParsedAction(Action.SILENCE, raw=raw or "")

    marker = _first_marker(text)
    if marker == -1:
        return ParsedAction(Action.RESPONSE, text=_collapse(text), raw=raw)

    if text[marker:].startswith(SILENCE_TOKEN):
        return ParsedAction(Action.SILENCE, raw=raw)

    payload = text[marker + len(RESPONSE_TOKEN) :].strip()
    if DELEGATION_TAG in payload:
        note, question = payload.split(DELEGATION_TAG, 1)
        return ParsedAction(
            Action.DELEGATE,
            text=_collapse(note),
            delegated_question=_collapse(question) or None,
            raw=raw,
        )
    return ParsedAction(Action.RESPONSE, text=_collapse(payload), raw=raw)


def to_token_form(action: ParsedAction) -> str:
    """Serialize back to the model's control-token form (what UIs expect)."""
    if action.action is Action.SILENCE:
        return SILENCE_TOKEN
    if action.action is Action.DELEGATE:
        return f"{RESPONSE_TOKEN} {action.text} {DELEGATION_TAG} {action.delegated_question or ''}".strip()
    return f"{RESPONSE_TOKEN} {action.text}".strip()


def _first_marker(text: str) -> int:
    present = [i for i in (text.find(SILENCE_TOKEN), text.find(RESPONSE_TOKEN)) if i != -1]
    return min(present) if present else -1


def _collapse(text: str) -> str:
    return " ".join(text.split())
