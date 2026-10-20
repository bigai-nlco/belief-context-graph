"""AutoFanInObserver — drain completed sub-agent reports between turns."""

from __future__ import annotations

import logging
from typing import Any, cast

from frontier_agent.components.agent_bus import AgentBus, SubAgentResult
from frontier_agent.components.agent_bus.fan_in import (
    format_status_line,
    process_collected,
)
from frontier_agent.core.loop_types import (
    BaseObserver,
    Intervention,
    TurnContext,
)
from frontier_agent.core.messages import user_msg
from frontier_agent.core.runtime import registry

logger = logging.getLogger(__name__)


class AutoFanInObserver(BaseObserver):
    """Drain completed sub-agent reports on turn_end and inject them."""

    critical: bool = True

    def __init__(self, team_memory: Any = None) -> None:
        self.team_memory = team_memory

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        bus = registry.get_optional(AgentBus)
        if bus is None:
            return None

        collected: list[tuple[str, SubAgentResult]] = []
        while True:
            ready = await bus.wait_any_session(ctx.task_id, timeout=0.0)
            if ready is None:
                break
            collected.append(ready)

        if not collected:
            return None

        batch = process_collected(bus, ctx.task_id, collected)
        status = format_status_line(
            bus, ctx.task_id,
            paused_names=batch.paused_names,
            incomplete_count=batch.incomplete_count,
        )
        if batch.evidence_count or batch.assertion_count:
            status += (
                f" harvested_this_turn={batch.evidence_count} evidence "
                f"+ {batch.assertion_count} assertions"
            )

        logger.info(
            "AutoFanIn: task=%s turn=%d reports=%d ev=%d as=%d incomplete=%d",
            ctx.task_id, ctx.turn, len(collected),
            batch.evidence_count, batch.assertion_count, batch.incomplete_count,
        )
        text = "\n\n".join([*batch.blocks, status])
        if self.team_memory is not None:
            # Direct host metadata, not a marker inferred from report text.
            # The engine otherwise coerces injected strings into ordinary user
            # messages, making every report permanently pinned by BCG.
            message = user_msg(text)
            cast(dict[str, Any], message)["bcg_origin"] = "agent_report"
            ctx.messages.append(message)
            self.team_memory.delivered(collected, text=text)
            return None
        return Intervention(inject_messages=[text])


__all__ = ["AutoFanInObserver"]
