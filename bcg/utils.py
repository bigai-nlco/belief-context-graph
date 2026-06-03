from __future__ import annotations

import base64
import inspect
import uuid
from datetime import UTC, datetime
from typing import Any


def get_random_uuid() -> str:
    """Generate a random UUID string."""

    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Get the current UTC time."""

    return datetime.now(UTC)


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def inline_image_url(image_bytes: bytes, *, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
