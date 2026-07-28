"""BeliefTracer tool implementations."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "AVeriTeCSearchTool":
        from bcg.agent.tools.averitec_search import AVeriTeCSearchTool

        return AVeriTeCSearchTool
    if name == "FileReadTool":
        from bcg.agent.tools.file_read_tool import FileReadTool

        return FileReadTool
    if name == "BCPSearchTool":
        from bcg.agent.tools.bcp_search import BCPSearchTool

        return BCPSearchTool
    if name == "SerperSearchTool":
        from bcg.agent.tools.serper_search import SerperSearchTool

        return SerperSearchTool
    if name == "SerperScrapeTool":
        from bcg.agent.tools.serper_scrape import SerperScrapeTool

        return SerperScrapeTool
    raise AttributeError(name)


__all__ = [
    "AVeriTeCSearchTool",
    "FileReadTool",
    "BCPSearchTool",
    "SerperSearchTool",
    "SerperScrapeTool",
]
