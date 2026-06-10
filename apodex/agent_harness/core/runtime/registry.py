"""Back-compat shim. Moved to agent_harness.core.runtime.registries.services.

Uses module-level ``__getattr__`` delegation so that test patches applied to
``agent_harness.core.runtime.registries.services`` (or attributes of this module
itself) propagate to callers that import via ``from agent_harness.core.runtime
import registry``.
"""

from agent_harness.core.runtime.registries import services as _impl


def __getattr__(name: str):  # type: ignore[override]
    return getattr(_impl, name)


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(globals()) | set(dir(_impl)))


# Names resolved via __getattr__ — ruff can't see the dynamic bindings.
__all__ = [  # noqa: F822
    "register",
    "get",
    "get_optional",
    "clear",
    "is_registered",
    "_services",
]
