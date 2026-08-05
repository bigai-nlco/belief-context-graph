from __future__ import annotations

import base64
import inspect
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_random_uuid() -> str:
    """Generate a random UUID string."""

    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Get the current UTC time."""

    return datetime.now(UTC)


def save_json(payload: Any, path: str | Path) -> None:
    """Write a JSON file with stable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def new_run_id() -> str:
    """Create a compact timestamped run id."""

    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{get_random_uuid()[:8]}"


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def inline_image_url(image_bytes: bytes, *, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
