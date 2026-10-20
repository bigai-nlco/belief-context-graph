"""Offline multi-agent BCG tests. No API keys, paid models or live server."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from types import SimpleNamespace

import pytest

from frontier_agent.core.llm import LLMResponse
from workflows.agent_team.bcg import TeamMemory
from workflows.stateful_react_agent.bcg.client import GraphError
from workflows.stateful_react_agent.bcg.messages import partition
from workflows.stateful_react_agent.bcg.runtime import BCGSettings


def call(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def step(n, name="lookup", args=None, result=None):
    return [
        {
            "role": "assistant",
            "content": f"Reason {n}",
            "tool_calls": [call(str(n), name, args or {"query": f"query {n}"})],
        },
        {"role": "tool", "tool_call_id": str(n), "content": result or f"Evidence {n}"},
    ]


def history(n=4, question="Find the date"):
    return [
        {"role": "system", "content": "Original system"},
        {"role": "user", "content": question},
        *(m for i in range(n) for m in step(i)),
    ]


class Graph:
    """Deterministic extraction double with real source/call-id contracts."""

    def __init__(self):
        self.turns = {}
        self.closed = False
        self.fail = False

    def snapshot(self, pid):
        turns = self.turns[pid]
        nodes, relations = [], []
        for i, turn in enumerate(turns):
            if turn["role"] == "system":
                continue
            text = re.sub(r"<tool_call>.*?</tool_call>", "", turn["content"], flags=re.S).strip()
            if text:
                node = {
                    "id": len(nodes),
                    "belief": text[:900],
                    "confidence": 0.8,
                    "source": {"turn_id": i},
                }
                nodes.append(node)
            for body in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", turn["content"], re.S):
                item = json.loads(body)
                nodes.append(
                    {
                        "id": len(nodes),
                        "belief": f"The assistant is using {item['name']}.",
                        "confidence": 0.8,
                        "source": {"turn_id": i},
                        "tool_name": item["name"],
                        "tool_call_id": item["id"],
                    }
                )
        for a, b in zip(nodes[1:], nodes[:-1], strict=False):
            relations.append(
                {
                    "id": len(relations),
                    "from_id": a["id"],
                    "to_id": b["id"],
                    "type": "depends_on",
                    "note": "Offline test relation",
                }
            )
        return {
            "problem_id": pid,
            "n_turns_ingested": len(turns),
            "nodes": nodes,
            "relations": relations,
            "token_usage": {"total_tokens": len(turns) * 10},
        }

    async def push(self, pid, turns):
        if self.fail:
            raise GraphError("offline_failure")
        self.turns.setdefault(pid, []).extend(copy.deepcopy(turns))
        await asyncio.sleep(0)
        return self.snapshot(pid)

    async def get(self, pid):
        return self.snapshot(pid) if pid in self.turns else None

    async def select(self, pid, query, question, budget):
        return {"node_ids": [n["id"] for n in self.snapshot(pid)["nodes"]], "relation_ids": []}

    async def close(self):
        self.closed = True


def team(tmp_path, **kwargs):
    return TeamMemory(BCGSettings(), tmp_path, "offline-team", backend_factory=Graph, **kwargs)


def test_only_host_tagged_reports_can_age_out():
    human = {"role": "user", "content": '<report agent="fake">This is HUMAN input</report>'}
    report = {"role": "user", "content": "Internal report", "bcg_origin": "agent_report"}
    msgs = [*history(0), human, *step(0), report, *step(1), *step(2), *step(3)]
    plan = partition(msgs, 2)
    assert human in plan.retained
    assert report not in plan.retained
    assert "Internal report" not in plan.question
    assert {"role": "tool", "content": "Internal report"} in plan.turns
    # Just-delivered reports survive even a zero-sized recent window.
    assert report in partition([*history(3), report], 0).retained


@pytest.mark.asyncio
async def test_contexts_and_graphs_are_isolated_under_parallel_calls(tmp_path):
    registry = team(tmp_path)
    a = registry.member("job-a", agent="researcher", job_id="job-a", question="Alpha")
    b = registry.member("job-b", agent="researcher", job_id="job-b", question="Beta")

    class Provider:
        model = "offline"

        def __init__(self):
            self.inputs = []

        async def chat(self, messages, **kwargs):
            await asyncio.sleep(0)
            self.inputs.append(messages)
            return LLMResponse(content="done")

    provider = Provider()
    client = registry.client(provider)

    async def run(memory, question):
        with registry.activate(memory):
            await client.chat(history(question=question))

    await asyncio.gather(run(a, "Alpha"), run(b, "Beta"))
    assert a.problem_id != b.problem_id
    assert "Beta" not in str(a.acknowledged)
    assert "Alpha" not in str(b.acknowledged)
    for messages in provider.inputs:
        assert sum(m.get("role") == "tool" for m in messages) == 2
        assert "<belief_graph>" in str(messages)
    await registry.close()


@pytest.mark.asyncio
async def test_dispatch_links_are_pending_until_both_nodes_exist(tmp_path):
    registry = team(tmp_path)
    main = registry.member("main", agent="main")
    with registry.activate(main):
        registry.delegated("j1", "researcher", "Child assignment", "assign-1")
    assert registry.links[0]["status"] == "pending"
    child = registry.job_memory("j1", "researcher", "Child assignment")
    await child.prepare(history(0, "Child assignment"))
    assert registry.links[0]["status"] == "pending"
    messages = history(0) + step("assign-1", "assign_task", {"tasks": []}) + step(2) + step(3)
    await main.prepare(messages)
    link = registry.links[0]
    assert link["status"] == "bound"
    assert link["from"][0]["graph_id"] == main.graph_id
    assert link["to"][0]["graph_id"] == child.graph_id
    assert registry.read(main.graph_id, link["from"][0]["belief_id"], 1)["nodes"]
    await registry.close()


@pytest.mark.asyncio
async def test_report_content_is_semantic_and_wire_messages_unchanged(tmp_path):
    registry = team(tmp_path)
    memory = registry.member("j1", agent="a", job_id="j1", question="Assignment")
    content = "Finding: The release date is 2018. Evidence: official record."
    messages = history(2, "Assignment") + step(
        "report", "submit_report", {"content": content}, "Report submitted"
    )
    original = copy.deepcopy(messages)
    projection = memory.project(messages, 0)
    assert projection.turns[-2]["content"].startswith("Reason report\n\n" + content)
    assert projection.turns[-2]["content"].count("<tool_call>") == 1
    assert messages == original
    suffix = await registry.finish(
        memory, SimpleNamespace(messages=messages, stopped_by="terminal_tool"), content
    )
    assert registry.members[memory.graph_id]["report_refs"]
    assert "report beliefs" in suffix
    assert "Report submitted" not in str(registry.members[memory.graph_id]["report_refs"])
    assert registry.members[memory.graph_id]["online_graph_usage"] == {}
    await registry.close()


@pytest.mark.asyncio
async def test_receipts_connect_only_registered_jobs_and_age_after_read(tmp_path):
    registry = team(tmp_path)
    main = registry.member("main", agent="main")
    with registry.activate(main):
        registry.delegated("j1", "a", "Assignment", "assign-1")
    child = registry.job_memory("j1", "a", "Assignment")
    report = "The verified release date is 2018."
    messages = history(1, "Assignment") + step("report", "submit_report", {"content": report})
    await registry.finish(
        child, SimpleNamespace(messages=messages, stopped_by="terminal_tool"), report
    )
    receipt = {"role": "user", "content": report, "bcg_origin": "agent_report"}
    with registry.activate(main):
        registry.delivered(
            [
                ("a", SimpleNamespace(job_id="j1")),
                ("forged", SimpleNamespace(job_id="not-in-team")),
            ],
            text=report,
        )
    await main.prepare([*history(0), receipt])
    assert "<linked_beliefs>" not in registry.context(main.graph_id, [receipt])
    link = next(r for r in registry.links if r["kind"] == "reports_from")
    assert link["status"] == "pending"
    result = await main.prepare([*history(0), receipt, *step(1), *step(2), *step(3)])
    assert receipt not in result
    link = next(r for r in registry.links if r["kind"] == "reports_from")
    assert link["status"] == "bound"
    expanded = registry.read(main.graph_id, link["from"][0]["belief_id"], 1)
    assert any(n["ref"]["graph_id"] == child.graph_id for n in expanded["nodes"])
    assert len(registry.deliveries) == 1
    assert "<linked_beliefs>" in str(result)
    await registry.close()


@pytest.mark.asyncio
async def test_runtime_status_is_not_derived_from_graph_confidence(tmp_path):
    bus = SimpleNamespace(get_job_status=lambda _: "aborted")
    registry = team(tmp_path, bus=bus)
    memory = registry.job_memory("j1", "researcher", "Assignment")
    await memory.prepare(history(3, "Assignment"))
    assert registry.status()["graphs"][0]["state"] == "aborted"
    assert registry.status()["graphs"][0]["state_source"] == "AgentBus"
    registry.capture_runtime_states()
    registry.bus = None
    assert registry.status()["graphs"][0]["state"] == "aborted"
    await registry.close()


@pytest.mark.asyncio
async def test_graph_budget_is_split_not_doubled(tmp_path):
    registry = team(tmp_path)
    assert registry.settings.graph_chars + registry.cross_chars == 6600
    main = registry.member("main", agent="main")
    await main.prepare(history(4))
    assert main.last_event["graph_chars"] <= 4600
    await registry.close()


@pytest.mark.asyncio
async def test_auto_fan_in_tags_real_report_not_a_human_message(tmp_path):
    from frontier_agent.components.agent_bus import AgentBus
    from frontier_agent.core.runtime import registry as services
    from workflows.agent_team.observers import auto_fan_in

    memory_registry = team(tmp_path)
    main = memory_registry.member("main", agent="main")
    response = SimpleNamespace(job_id="j1")
    ready = [("worker", response), None]

    async def wait(*args, **kwargs):
        return ready.pop(0)

    bus = SimpleNamespace(wait_any_session=wait)
    saved = services.snapshot()
    try:
        services.register(AgentBus, bus)
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                auto_fan_in,
                "process_collected",
                lambda *a: SimpleNamespace(
                    blocks=["A completed report"],
                    paused_names=[],
                    incomplete_count=0,
                    evidence_count=0,
                    assertion_count=0,
                ),
            )
            monkeypatch.setattr(auto_fan_in, "format_status_line", lambda *a, **kw: "done")
            with memory_registry.activate(main):
                memory_registry.delegated("j1", "worker", "Assignment", "call1")
                ctx = SimpleNamespace(task_id="t", turn=1, messages=history(0))
                value = await auto_fan_in.AutoFanInObserver(memory_registry).on_turn_end(ctx)
            assert value is None
            assert ctx.messages[-1]["bcg_origin"] == "agent_report"
            assert len(memory_registry.deliveries) == 1
    finally:
        services.restore(saved)
        await memory_registry.close()


@pytest.mark.asyncio
async def test_streaming_uses_same_projection_as_chat(tmp_path):
    from frontier_agent.core.llm import StreamDelta

    registry = team(tmp_path)
    memory = registry.member("main", agent="main")

    class Provider:
        model = "offline-stream"

        async def stream(self, messages, **kwargs):
            assert "<belief_graph>" in str(messages)
            assert "<team_graph_guide>" in str(messages)
            yield StreamDelta(content="answer")

    with registry.activate(memory):
        parts = [part async for part in registry.client(Provider()).stream(history(4))]
    assert parts[0].content == "answer"
    await registry.close()


@pytest.mark.asyncio
async def test_fail_open_preserves_all_messages_and_task_board(tmp_path):
    registry = team(tmp_path, board=lambda: "Task 1 running; no report yet")
    memory = registry.member("main", agent="main")
    memory.backend.fail = True
    messages = history(5)
    result = await memory.prepare(messages)
    assert all(m in result for m in messages)
    assert memory.last_event["fallback"] == "offline_failure"
    assert "Task 1 running" in str(result)
    await registry.close()


@pytest.mark.asyncio
async def test_job_reuse_does_not_reextract_previous_assignment(tmp_path):
    registry = team(tmp_path)
    memory = registry.job_memory("job2", "same-specialist", "New assignment")
    old = history(3, "Old assignment")
    current = [{"role": "user", "content": "New assignment"}, *step("new")]
    projected = await memory.prepare(old + current)
    assert not any(m.get("content") == "Old assignment" for m in projected)
    assert "Old assignment" not in str(memory.acknowledged)
    assert memory.acknowledged[1]["content"] == "New assignment"
    await registry.close()


@pytest.mark.asyncio
async def test_semantic_links_are_explicit_scoped_and_do_not_propagate(tmp_path):
    registry = team(tmp_path)
    a, b = [registry.member(name, agent=name) for name in ("a", "b")]
    await a.prepare(history(3))
    await b.prepare(history(3))
    original = copy.deepcopy(b.snapshot)
    tools = {t.name: t for t in registry.tools()}
    with registry.activate(a):
        linked = await tools["bcg_team_link"].func(
            belief_id=0, target_graph=b.graph_id, target_belief_id=0, relation_type="depends_on"
        )
        assert "agent_asserted" in linked
        assert "not registered" in await tools["bcg_team_read"].func(
            graph_id="../outside", belief_id=0
        )
    assert b.snapshot == original
    assert not registry.semantic_links[0]["propagates_confidence"]
    assert len(registry.read(a.graph_id, 0)["nodes"]) >= 2
    # Rebuilding another graph invalidates the old edge rather than silently
    # reconnecting the same numeric id to unrelated new content.
    b._reset()
    assert not registry._live_link(registry.semantic_links[0])
    await registry.close()


@pytest.mark.asyncio
async def test_checkpoint_reload_and_merged_reference_resolution(tmp_path):
    registry = team(tmp_path)
    a = registry.member("a", agent="a")
    await a.prepare(history(3))
    a.snapshot["nodes"][0]["merged_from"] = [99]
    registry.refresh(a)
    assert registry.resolve(a.graph_id, 99)[0] == 0
    await registry.close()
    restored = team(tmp_path)
    assert restored.resolve(a.graph_id, 99)[0] == 0
    assert restored.memories[a.graph_id].problem_id == a.problem_id
    different = TeamMemory(BCGSettings(), tmp_path, "different-team", backend_factory=Graph)
    assert not different.members
    await restored.close()


@pytest.mark.asyncio
async def test_assignment_references_are_validated_and_bounded(tmp_path):
    registry = team(tmp_path)
    memory = registry.member("main", agent="main")
    await memory.prepare(history(3))
    ref = f"{memory.graph_id}:B0"
    assert ref in registry.assignment_context([ref])
    with pytest.raises(ValueError):
        registry.assignment_context(["unknown:B0"])
    assert registry.assignment_context([]) == ""
    await registry.close()


@pytest.mark.asyncio
async def test_actual_team_runtime_fanout_report_and_graph_projection(tmp_path, monkeypatch):
    """Run the production node, actual AgentBus, and actual delegation tools."""
    from frontier_agent.components.agent_bus import AgentBus
    from frontier_agent.core.runtime import registry
    from frontier_agent.core.runtime.registries.agents import AgentRegistry
    from frontier_agent.core.runtime.registries.workflows import WorkflowContext
    from frontier_agent.core.runtime.resources.manager import ResourceManager
    from frontier_agent.scheduling.pipeline_registry import PipelineRegistry
    from plugins.tools.assign_task import assign_task
    from plugins.tools.collect_reports import collect_reports
    from plugins.tools.create_subagent import create_subagent
    from plugins.tools.submit_report import submit_report
    from workflows.agent_team import register
    from workflows.agent_team.nodes import main_agent
    from workflows.stateful_react_agent.bcg import runtime

    class Provider:
        model = "offline-team"

        def __init__(self):
            self.inputs = []
            self.main_n = 0
            self.child_n = {}

        async def chat(self, messages, **kwargs):
            await asyncio.sleep(0)
            names = [t["function"]["name"] for t in (kwargs.get("tools") or [])]
            self.inputs.append((names, copy.deepcopy(messages)))
            if "create_subagent" in names:
                self.main_n += 1
                n = self.main_n
                if n == 1:
                    return LLMResponse(
                        content="Delegate independent checks.",
                        tool_calls=[
                            call(
                                "create",
                                "create_subagent",
                                {
                                    "agents": [
                                        {"name": name, "system_prompt": "Check the dates"}
                                        for name in ("alpha", "beta")
                                    ]
                                },
                            )
                        ],
                    )
                if n == 2:
                    return LLMResponse(
                        content="Assign both jobs.",
                        tool_calls=[
                            call(
                                "assign",
                                "assign_task",
                                {
                                    "tasks": [
                                        {"agent": name, "prompt": f"Find {name} date with evidence"}
                                        for name in ("alpha", "beta")
                                    ]
                                },
                            )
                        ],
                    )
                if n in (3, 4):
                    return LLMResponse(
                        content="Wait for evidence.",
                        tool_calls=[call(f"collect{n}", "collect_reports", {"timeout": 1})],
                    )
                if n == 5:
                    return LLMResponse(
                        content="Inspect graph index.",
                        tool_calls=[call("status", "bcg_team_status", {})],
                    )
                if n == 6:
                    text = next(
                        m["content"]
                        for m in reversed(messages)
                        if m.get("tool_call_id") == "status"
                    )
                    data = json.loads(text.split("\n", 1)[1].rsplit("\nnext_offset=", 1)[0])
                    refs = [ref for g in data["graphs"] for ref in g["report_refs"]]
                    assert refs, "Child report beliefs must exist before collection finishes"
                    graph_id, node_id = refs[0].rsplit(":B", 1)
                    return LLMResponse(
                        content="Inspect the recorded evidence chain.",
                        tool_calls=[
                            call(
                                "read",
                                "bcg_team_read",
                                {
                                    "graph_id": graph_id,
                                    "belief_id": int(node_id),
                                    "include_source": True,
                                },
                            )
                        ],
                    )
                return LLMResponse(
                    content="The dates are 2018 and 2019, based on the recorded evidence."
                )
            if "submit_report" in names:
                # ContextVar is intentionally tested through actual child tasks.
                from workflows.agent_team.bcg import _ACTIVE

                memory = _ACTIVE.get()
                name = memory.team.members[memory.graph_id]["agent"]
                n = self.child_n.get(name, 0) + 1
                self.child_n[name] = n
                if n <= 3:
                    return LLMResponse(
                        content=f"Checking {name} evidence {n}",
                        tool_calls=[call(f"{name}-{n}", "bcg_team_status", {})],
                    )
                return LLMResponse(
                    content="Ready to report",
                    tool_calls=[
                        call(
                            f"{name}-report",
                            "submit_report",
                            {
                                "content": f"Scope: {name}\nFinding: The date is 2018.\nEvidence: offline official record.\nConfidence: 0.8"
                            },
                        )
                    ],
                )
            return LLMResponse(content="Final synthesis from the verified reports.")

    for name in ("workspace", "outputs", "inputs"):
        (tmp_path / name).mkdir()
        monkeypatch.setenv(f"FRONTIER_AGENT_{name.upper()}_DIR", str(tmp_path / name))
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    provider = Provider()
    graph = Graph()
    profile = {
        "llm": {"max_tokens": 2048},
        "agent": {
            "planning_mode": False,
            "reporter": False,
            "main_max_turns": 20,
            "sub_max_turns": 20,
            "main_agent_tools": ["create_subagent", "assign_task", "collect_reports"],
            "sub_agent_tools": ["submit_report"],
            "sub_stuck_target_hint_after": 0,
        },
    }
    monkeypatch.setattr(
        main_agent, "_resolve_llm_and_profile", lambda *a, **kw: (provider, profile, None)
    )
    monkeypatch.setattr(runtime, "HttpGraphBackend", lambda *a: graph)
    saved = registry.snapshot()
    try:
        registry.clear()
        agents = AgentRegistry()
        register(WorkflowContext(PipelineRegistry(), agents))
        registry.register(AgentRegistry, agents)
        tools = {t.name: t for t in (create_subagent, assign_task, collect_reports, submit_report)}
        rm = ResourceManager(provider, tools)
        registry.register(ResourceManager, rm)
        bus = AgentBus(agent_registry=agents, resource_manager=rm)
        registry.register(AgentBus, bus)
        result = await asyncio.wait_for(
            main_agent.main_agent_node(
                {
                    "original_question": "Find the release and implementation dates",
                    "language": "en",
                    "metadata": {"_trial_dir": str(tmp_path / "trial")},
                },
                SimpleNamespace(task_id="offline-team-production"),
            ),
            20,
        )
    finally:
        registry.restore(saved)
    assert provider.child_n == {"alpha": 4, "beta": 4}
    assert result["bcg_team"]["graphs"]
    assert len(result["bcg_team"]["graphs"]) == 3
    assert len({g["graph_id"] for g in result["bcg_team"]["graphs"]}) == 3
    assert all(g["report_refs"] for g in result["bcg_team"]["graphs"] if g["job_id"])
    assert any(
        link["status"] == "bound"
        for link in result["bcg_team"]["links"]
        if link["kind"] == "delegates_to"
    )
    assert any(
        link["status"] == "bound"
        for link in result["bcg_team"]["links"]
        if link["kind"] == "reports_from"
    )
    assert any("<belief_graph>" in str(m) for _, m in provider.inputs)
    assert all("<team_graph_guide>" not in str(m) for m in result["session_messages"])
    assert graph.closed
