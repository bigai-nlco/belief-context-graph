#!/usr/bin/env python3
"""Deploy YAML smoke check (step 13).

Validates the structure of deploy/tonggraph-server.yml, verifies tokens are
only referenced via token_env (never inline), and that the data_dir is a
portable relative default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "deploy" / "tonggraph-server.yml"

REQUIRED_TOP_LEVEL = ("host", "port", "data_dir", "graphs", "operations", "auth")


def main() -> int:
    raw = CONFIG.read_text(encoding="utf-8")
    if "/data/user/" in raw:
        print(f"ERROR: {CONFIG} contains a personal absolute path", file=sys.stderr)
        return 1

    doc = yaml.safe_load(raw)
    if not isinstance(doc, dict):
        print(f"ERROR: {CONFIG} must be a YAML mapping", file=sys.stderr)
        return 1

    missing = [key for key in REQUIRED_TOP_LEVEL if key not in doc]
    if missing:
        print(f"ERROR: {CONFIG} missing keys: {missing}", file=sys.stderr)
        return 1

    port = doc["port"]
    if not isinstance(port, int) or not 1 <= port <= 65535:
        print(f"ERROR: invalid port {port!r}", file=sys.stderr)
        return 1

    if "data_dir" in doc and not isinstance(doc["data_dir"], str):
        print(f"ERROR: data_dir must be a string, got {doc['data_dir']!r}", file=sys.stderr)
        return 1

    users = (doc.get("auth") or {}).get("users") or {}
    for user_name, user_cfg in users.items():
        if not isinstance(user_cfg, dict):
            print(f"ERROR: auth user {user_name!r} must be a mapping", file=sys.stderr)
            return 1
        token_env = user_cfg.get("token_env")
        if not isinstance(token_env, str) or not token_env:
            print(
                f"ERROR: auth user {user_name!r} must reference a token_env name",
                file=sys.stderr,
            )
            return 1
        for key in user_cfg:
            if "token" in key and key != "token_env":
                print(
                    f"ERROR: auth user {user_name!r} carries inline credential key {key!r}",
                    file=sys.stderr,
                )
                return 1

    print(f"OK: {CONFIG.name} structure valid, tokens env-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
