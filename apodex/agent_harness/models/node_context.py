"""NodeContext — minimal facade between a DAG node and the framework.

The ``react_base`` workflow's ``main_agent`` node consumes only one
attribute on the context: ``ctx.task_id``. Earlier revisions exposed a
larger surface (``call_llm`` / ``call_tool`` / ``search`` / ``scrape`` /
``emit_event`` / ``persist_*`` / ``run_agent_loop`` / ``service``) so
research-style nodes could stay decoupled from the runtime, but those
methods had no callers after the OSS prune deleted every workflow
except ``react_base``. The shape kept here is just enough for
``graph_builder._wrap_with_node_context`` to inject an object whose
``.task_id`` reads the live ``state["task_id"]`` on every access.
"""

from __future__ import annotations

import logging
from typing import Callable

from agent_harness.models.pipeline_spec import NodeDefinition

logger = logging.getLogger(__name__)


class NodeContext:
    """Identity-only context object passed to every wrapped node."""

    def __init__(
        self,
        node_def: NodeDefinition,
        task_id_getter: Callable[[], str],
    ) -> None:
        self._node_def = node_def
        self._task_id_getter = task_id_getter

    @property
    def node_id(self) -> str:
        return self._node_def.node_id

    @property
    def role_id(self) -> str:
        return self._node_def.role_id

    @property
    def task_id(self) -> str:
        return self._task_id_getter()


# Alias for callers that import ``DefaultNodeContext``; ``NodeContext``
# is the single concrete implementation.
DefaultNodeContext = NodeContext
