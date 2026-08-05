"""File-write components shared by both construct backends.

``EventRecorder`` owns the append-only JSONL event log; ``ArtifactWriter``
owns atomic JSON artifact writes.  Backend-specific result shapes and CSV
formats stay in each backend.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class EventRecorder:
    """Append-only JSONL event log for one construct run."""

    def __init__(self, events_path: Path) -> None:
        self._path = Path(events_path)
        self._path.write_text("", encoding="utf-8")

    def record(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Write one timestamped event and return the record."""
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": kind}
        rec.update(payload)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


class ArtifactWriter:
    """Atomic JSON artifact writes under one run directory."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        """Write ``payload`` as pretty JSON via temp-file + atomic replace."""
        path = self.out_dir / name
        fd, tmp = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".tmp", dir=self.out_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path


__all__ = ["ArtifactWriter", "EventRecorder"]
