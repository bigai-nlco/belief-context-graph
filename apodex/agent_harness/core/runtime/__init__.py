"""AgentHarness Kernel — agent loop and DAG engine.

Layout:
- ``loop/``        generic ReAct engine + tool-call parsing + execution context
- ``resources/``   tool permissions + LLM routing per agent role
- ``events/``      in-memory bus + SQLite event store
- ``dag/``         MiniDAG engine + DynamicGraphBuilder
- ``registries/``  service DI container, agent definitions, workflow context

``registry`` remains as a thin back-compat module at the top level so
external code can keep using ``from agent_harness.core.runtime import registry``.
"""
