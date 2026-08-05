"""Belief Context Graph public SDK interface.

Importing ``bcg`` loads only the SDK core: no apps, no concrete construct
backends, and no environment-variable side effects (load ``bcg.env`` /
call ``load_project_env`` explicitly when needed).
"""

from bcg.config import BCGSettings, load_settings
from bcg.core.graph import BCG
from bcg.core.llm import LLMClient
from bcg.core.memory import BCGMemory
from bcg.core.runner import BCGRunner

__all__ = [
    "BCG",
    "BCGMemory",
    "BCGRunner",
    "BCGSettings",
    "LLMClient",
    "load_settings",
]
