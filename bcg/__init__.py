"""Belief Context Graph public SDK interface.

Importing ``bcg`` loads only the SDK core: no apps, no concrete construct
backends, and no environment-variable side effects (load ``bcg.env`` /
call ``load_project_env`` explicitly when needed).
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from bcg.config import BCGSettings, load_settings
from bcg.core.graph import BCG
from bcg.core.llm import LLMClient
from bcg.core.memory import BCGMemory
from bcg.core.runner import BCGRunner


def _package_version() -> str:
    """Package version from installed metadata (pyproject is the single source)."""
    try:
        return _pkg_version("bcg")
    except PackageNotFoundError:
        return "0.1.0"


__version__ = _package_version()

__all__ = [
    "BCG",
    "BCGMemory",
    "BCGRunner",
    "BCGSettings",
    "LLMClient",
    "__version__",
    "load_settings",
]
