# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the shared SessionBrain interaction state."""

from vllm_omni.interaction.state import SessionBrain


def test_update_query_freshness():
    brain = SessionBrain()
    assert brain.update_query("count the bottles") is True
    assert brain.update_query("count the bottles") is False  # unchanged
    assert brain.update_query("") is False
    assert brain.update_query("now describe the scene") is True


def test_fresh_query_not_in_head_then_carried():
    brain = SessionBrain(frame_seconds=1.0)
    for _ in range(3):
        brain.tick()
    brain.update_query("alert me if a fire breaks out")
    # fresh query rides in the turn message, not the stable head
    assert "alert me if a fire breaks out" not in brain.build_prefix()
    brain.record_response("a fire is breaking out")
    brain.end_turn()
    # once the turn ends, the query + Q&A move into the stable head
    prefix = brain.build_prefix()
    assert "alert me if a fire breaks out" in prefix
    assert "Q&A History" in prefix
    assert "a fire is breaking out" in prefix


def test_responses_accumulate_under_one_query():
    brain = SessionBrain(frame_seconds=1.0)
    brain.update_query("count the bottles")
    brain.tick()
    brain.record_response("1 bottle")
    brain.tick()
    brain.record_response("2 bottles")
    assert len(brain.memory.qa_history) == 1
    assert brain.memory.qa_history[-1].responses[-1] == ("2.0s", "2 bottles")


def test_empty_response_is_ignored():
    brain = SessionBrain()
    brain.update_query("anything")
    brain.record_response("")
    assert brain.memory.qa_history == []
