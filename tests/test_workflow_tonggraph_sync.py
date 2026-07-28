from __future__ import annotations

import asyncio
import sys
import types


def _install_rllm_workflow_stubs() -> None:
    if "rllm.workflows.multi_turn_workflow" in sys.modules:
        return
    rllm = types.ModuleType("rllm")
    rllm.__path__ = []
    modules = {
        "rllm": rllm,
        "rllm.engine": types.ModuleType("rllm.engine"),
        "rllm.engine.rollout": types.ModuleType("rllm.engine.rollout"),
        "rllm.engine.rollout.rollout_engine": types.ModuleType("rllm.engine.rollout.rollout_engine"),
        "rllm.agents": types.ModuleType("rllm.agents"),
        "rllm.agents.agent": types.ModuleType("rllm.agents.agent"),
        "rllm.workflows": types.ModuleType("rllm.workflows"),
        "rllm.workflows.multi_turn_workflow": types.ModuleType("rllm.workflows.multi_turn_workflow"),
        "rllm.workflows.workflow": types.ModuleType("rllm.workflows.workflow"),
    }
    for name, module in modules.items():
        sys.modules[name] = module

    class _Base:
        pass

    class _TerminationEvent(Exception):
        pass

    class _TerminationReason:
        ENV_DONE = "env_done"
        MAX_PROMPT_LENGTH_EXCEEDED = "max_prompt_length_exceeded"
        MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
        MAX_TURNS_EXCEEDED = "max_turns_exceeded"

    modules["rllm.engine.rollout.rollout_engine"].ModelOutput = _Base
    modules["rllm.agents.agent"].Episode = _Base
    modules["rllm.workflows.multi_turn_workflow"].MultiTurnWorkflow = _Base
    modules["rllm.workflows.workflow"].TerminationEvent = _TerminationEvent
    modules["rllm.workflows.workflow"].TerminationReason = _TerminationReason


def test_record_graph_snapshot_syncs_each_snapshot(monkeypatch) -> None:
    _install_rllm_workflow_stubs()
    from bcg.agent.workflow import BeliefTracerWorkflow

    calls: list[dict] = []

    def fake_sync_graph_payload(payload, **kwargs):
        calls.append({"payload": payload, "kwargs": kwargs})

        class Result:
            graph = kwargs["graph"]
            logical_graph_id = kwargs["logical_graph_id"]
            nodes_created = 1
            nodes_reused = 0
            nodes_deleted = 0
            edges_created = 0
            edges_reused = 0
            edges_deleted = 0
            embeddings_upserted = 0
            readback_verified = kwargs["verify_readback"]

        return Result()

    monkeypatch.setattr("bcg.agent.tonggraph_sync.sync_graph_payload", fake_sync_graph_payload)

    async def run_inline(func, /, *args, **kwargs):
        """Keep this unit test independent of executor shutdown behavior."""
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    workflow = BeliefTracerWorkflow.__new__(BeliefTracerWorkflow)
    workflow._graph_snapshots = []
    workflow._problem_id = "claim:roundtrip"
    workflow._tonggraph_sync_config = {
        "enabled": True,
        "base_url": "http://tonggraph",
        "token": "token",
        "graph": "agent_workspace",
        "logical_graph_id": "",
        "timeout": 1.0,
        "text_index": None,
        "embedding_url": "",
        "embedding_model": "",
        "embedding_index": None,
        "embedding_batch_size": 16,
    }

    first = {"beliefs": [{"id": "n1", "belief": "first"}], "n_beliefs": 1}
    second = {"beliefs": [{"id": "n1", "belief": "second"}], "n_beliefs": 1}

    asyncio.run(workflow._record_graph_snapshot(first, phase="turn"))
    asyncio.run(workflow._record_graph_snapshot(second, phase="turn"))

    assert workflow._graph_snapshots == [first, second]
    assert [call["payload"] for call in calls] == [first, second]
    assert all(call["kwargs"]["verify_readback"] is True for call in calls)
    assert all(call["kwargs"]["logical_graph_id"] == "claim_roundtrip" for call in calls)


def test_graph_payloads_move_only_when_raw_turn_is_evicted() -> None:
    _install_rllm_workflow_stubs()
    from bcg.agent.workflow import _evict_graph_turn_payloads

    window: list[list[dict]] = []
    turn0 = [{"role": "assistant", "content": "a0"}, {"role": "tool", "content": "t0"}]
    turn1 = [{"role": "assistant", "content": "a1"}, {"role": "tool", "content": "t1"}]
    turn2 = [{"role": "assistant", "content": "a2"}, {"role": "tool", "content": "t2"}]

    assert _evict_graph_turn_payloads(window, turn0, 2) == []
    assert _evict_graph_turn_payloads(window, turn1, 2) == []
    assert _evict_graph_turn_payloads(window, turn2, 2) == turn0
    assert window == [turn1, turn2]


def test_graph_payload_eviction_handles_unbounded_and_graph_only_modes() -> None:
    _install_rllm_workflow_stubs()
    from bcg.agent.workflow import _evict_graph_turn_payloads

    turn = [{"role": "assistant", "content": "answer"}]
    unbounded: list[list[dict]] = []
    graph_only: list[list[dict]] = []

    assert _evict_graph_turn_payloads(unbounded, turn, None) == []
    assert unbounded == []
    assert _evict_graph_turn_payloads(graph_only, turn, 0) == turn
    assert graph_only == []
