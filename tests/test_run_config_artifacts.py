from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field

from bcg.agent.config import AgentRolloutConfig


@dataclass
class _RewardConfig:
    toolcall_bonus: float = 0.0
    apply_repetition_penalty: bool = False
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    enable_step_bonus: bool = False


@dataclass
class _RewardInput:
    task_info: dict
    action: str


@dataclass
class _RewardOutput:
    reward: float = 0.0
    is_correct: bool = False
    metadata: dict = field(default_factory=dict)


class _RewardSearchFn:
    def __init__(self, *_args, **_kwargs):
        pass

    def extract_answer_from_response(self, response: str) -> str:
        return str(response)


def _install_rllm_stubs() -> None:
    rllm = types.ModuleType("rllm")
    rllm.__path__ = []
    modules = {
        "rllm": rllm,
        "rllm.engine": types.ModuleType("rllm.engine"),
        "rllm.engine.agent_workflow_engine": types.ModuleType("rllm.engine.agent_workflow_engine"),
        "rllm.engine.rollout": types.ModuleType("rllm.engine.rollout"),
        "rllm.engine.rollout.rollout_engine": types.ModuleType("rllm.engine.rollout.rollout_engine"),
        "rllm.agents": types.ModuleType("rllm.agents"),
        "rllm.agents.agent": types.ModuleType("rllm.agents.agent"),
        "rllm.environments": types.ModuleType("rllm.environments"),
        "rllm.environments.base": types.ModuleType("rllm.environments.base"),
        "rllm.environments.base.base_env": types.ModuleType("rllm.environments.base.base_env"),
        "rllm.tools": types.ModuleType("rllm.tools"),
        "rllm.tools.multi_tool": types.ModuleType("rllm.tools.multi_tool"),
        "rllm.tools.tool_base": types.ModuleType("rllm.tools.tool_base"),
        "rllm.workflows": types.ModuleType("rllm.workflows"),
        "rllm.workflows.multi_turn_workflow": types.ModuleType("rllm.workflows.multi_turn_workflow"),
        "rllm.workflows.workflow": types.ModuleType("rllm.workflows.workflow"),
        "rllm.rewards": types.ModuleType("rllm.rewards"),
        "rllm.rewards.reward_fn": types.ModuleType("rllm.rewards.reward_fn"),
        "rllm.rewards.reward_types": types.ModuleType("rllm.rewards.reward_types"),
        "rllm.rewards.search_reward": types.ModuleType("rllm.rewards.search_reward"),
    }
    for name, module in modules.items():
        sys.modules[name] = module

    class _AgentWorkflowEngine:
        pass

    class _Base:
        pass

    @dataclass
    class _Step:
        chat_completions: list | None = None
        observation: object = None
        action: object = None
        model_response: str = ""
        model_output: object = None
        thought: str = ""
        reward: float = 0.0
        done: bool = False
        info: dict = field(default_factory=dict)

    @dataclass
    class _Trajectory:
        steps: list = field(default_factory=list)
        reward: float = 0.0

    class _TerminationReason:
        MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
        ENV_DONE = "env_done"

    class _TerminationEvent(Exception):
        pass

    class _Tool:
        def __init__(self, name: str, description: str | None = None, **_kwargs):
            self.name = name
            self.description = description or ""

    @dataclass
    class _ToolOutput:
        name: str
        output: object = None
        error: str | None = None
        metadata: dict | None = None

        def __str__(self) -> str:
            if self.error:
                return f"Error: {self.error}"
            return "" if self.output is None else str(self.output)

    modules["rllm.engine.agent_workflow_engine"].colorful_print = lambda *_args, **_kwargs: None
    modules["rllm.engine.agent_workflow_engine"].AgentWorkflowEngine = _AgentWorkflowEngine
    modules["rllm.engine.rollout.rollout_engine"].ModelOutput = _Base
    modules["rllm.agents.agent"].BaseAgent = _Base
    modules["rllm.agents.agent"].Action = object
    modules["rllm.agents.agent"].Episode = _Base
    modules["rllm.agents.agent"].Step = _Step
    modules["rllm.agents.agent"].Trajectory = _Trajectory
    modules["rllm.environments.base.base_env"].BaseEnv = _Base
    modules["rllm.tools.multi_tool"].MultiTool = _Base
    modules["rllm.tools.tool_base"].Tool = _Tool
    modules["rllm.tools.tool_base"].ToolCall = _Base
    modules["rllm.tools.tool_base"].ToolOutput = _ToolOutput
    modules["rllm.workflows.multi_turn_workflow"].MultiTurnWorkflow = _Base
    modules["rllm.workflows.workflow"].TerminationEvent = _TerminationEvent
    modules["rllm.workflows.workflow"].TerminationReason = _TerminationReason
    modules["rllm.rewards.reward_fn"].search_reward_fn = lambda *_args, **_kwargs: _RewardOutput()
    modules["rllm.rewards.reward_types"].RewardConfig = _RewardConfig
    modules["rllm.rewards.reward_types"].RewardInput = _RewardInput
    modules["rllm.rewards.reward_types"].RewardOutput = _RewardOutput
    modules["rllm.rewards.search_reward"].RewardSearchFn = _RewardSearchFn


_install_rllm_stubs()

from bcg.agent.runner import _write_run_config
from bcg.agent.ui import _read_run_config_for_state, _run_state_summary


def test_write_run_config_creates_latest_and_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "bcg.agent.runner._machine_info",
        lambda: {
            "hostname": "test-host",
            "gpu": {"available": False, "error": "test"},
        },
    )
    cfg = AgentRolloutConfig(
        model="/models/Qwen3.5-4B",
        output_dir=str(tmp_path),
        backend="sglang_dp",
        enable_thinking=True,
        max_new_tokens=1234,
    )

    _write_run_config(
        cfg,
        run_id="run-123",
        started_at=1000.0,
        status="starting",
        phase="initializing_backend",
    )

    model_dir = tmp_path / "Qwen3.5-4B_thinking"
    latest = model_dir / "run_config.json"
    history = model_dir / "run_configs" / "run-123.json"
    assert latest.exists()
    assert history.exists()

    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-123"
    assert payload["config"]["backend"] == "sglang_dp"
    assert payload["config"]["max_new_tokens"] == 1234
    assert payload["config"]["temperature"] is None
    assert payload["effective_config"]["engine"]["backend"] == "sglang_dp"
    assert payload["effective_config"]["engine"]["sglang_context_length"] == (
        cfg.max_prompt_length + cfg.max_response_length
    )
    assert payload["effective_config"]["sampling"]["temperature"] == 1.0
    assert payload["effective_config"]["sampling"]["top_k"] == 20
    assert (
        payload["effective_config"]["sampling_sources"]["temperature"]
        == "qwen3.5/3.6 thinking default"
    )
    assert payload["sampling_params"]["max_tokens"] == 1234
    assert payload["machine"]["hostname"] == "test-host"
    assert payload == json.loads(history.read_text(encoding="utf-8"))


def test_ui_reads_run_config_for_run_state(tmp_path) -> None:
    run_dir = tmp_path / "Qwen3.5-4B_thinking"
    config_dir = run_dir / "run_configs"
    config_dir.mkdir(parents=True)
    state_path = run_dir / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "run_id": "run-abc",
                "status": "running",
                "updated_at": 1000.0,
                "config": {
                    "model": "/models/Qwen3.5-4B",
                    "enable_thinking": True,
                },
            }
        ),
        encoding="utf-8",
    )
    history_path = config_dir / "run-abc.json"
    history_path.write_text(
        json.dumps({"schema_version": 1, "run_id": "run-abc", "config": {"x": 1}}),
        encoding="utf-8",
    )

    summary = _run_state_summary(state_path)
    payload = _read_run_config_for_state(state_path)

    assert summary["run_id"] == "run-abc"
    assert summary["run_config_path"] == str(history_path)
    assert payload["run_id"] == "run-abc"
    assert payload["_path"] == str(history_path)


def test_ui_run_config_falls_back_to_run_state(tmp_path) -> None:
    run_dir = tmp_path / "Qwen3.5-4B_thinking"
    run_dir.mkdir(parents=True)
    state_path = run_dir / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "run_id": "run-missing",
                "status": "completed",
                "phase": "completed",
                "config": {"model": "/models/Qwen3.5-4B"},
            }
        ),
        encoding="utf-8",
    )

    payload = _read_run_config_for_state(state_path)

    assert payload["schema_version"] == 0
    assert payload["source"] == "fallback_from_run_state"
    assert payload["config"]["model"] == "/models/Qwen3.5-4B"
