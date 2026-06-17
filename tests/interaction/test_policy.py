# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.interaction.delegation import DelegationResult
from vllm_omni.interaction.policy import JoyVLPolicy, sample_frames


def test_sample_frames_keeps_recent():
    assert sample_frames(["a", "b"], 4) == ["a", "b"]
    out = sample_frames([str(i) for i in range(10)], 4)
    assert out[-1] == "9" and len(out) == 4


def test_build_messages_stable_head_and_query():
    p = JoyVLPolicy(num_frames=4)
    p.tick(2)
    p.set_query("count the bottles")
    msgs, user = p.build_messages([{"type": "image_url", "image_url": {"url": "x"}}])
    assert msgs[0]["role"] == "system" and "</silence>" in msgs[0]["content"]

    assert any("count the bottles" in part.get("text", "") for part in user["content"])


def test_commit_records_only_spoken():
    p = JoyVLPolicy()
    p.set_query("q")
    assert p.commit("</silence>").action.value == "silence"
    assert p.brain.response_records == []
    assert p.commit("</response> hello").action.value == "response"
    assert p.brain.response_records == [("0.0s", "hello")]


class _Delegation:
    async def submit(self, question, note, frames):
        return "t1"

    async def poll(self, task_id):
        return DelegationResult(task_id, "ready", digest="background answer")


@pytest.mark.asyncio
async def test_delegation_submit_and_fold():
    p = JoyVLPolicy(delegation=_Delegation())
    action = p.commit("</response> hold on <delegation> hard question")
    assert action.action.value == "delegate"
    info = await p.submit_if_delegate(action)
    assert info["status"] == "submitted"
    folded = await p.fold_delegations()
    assert folded["status"] == "ready"
    assert p.brain.memory.qa_history[-1].responses[0][1] == "background answer"
