"""Role normalization shared by both construct backends."""

from __future__ import annotations

# function-role turns are tool outputs.
ROLE_ALIASES = {"function": "tool"}


def normalize_role(role: str) -> str:
    """Map legacy role aliases onto their canonical construct roles."""
    return ROLE_ALIASES.get(role, role)


__all__ = ["ROLE_ALIASES", "normalize_role"]
