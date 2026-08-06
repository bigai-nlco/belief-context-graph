#!/usr/bin/env python3
"""Generate the release manifest (step 14 / ADR-0001).

Records the exact component combination for a release: Python version,
Agent version, contract schema version, Dashboard version, and lockfile
state. Run with --check to fail on lockfile drift or version mismatches
(used by `make check-release` and the packaging CI job).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _python_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("bcg")
    except PackageNotFoundError:
        pyproject = json.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        return pyproject["project"]["version"]


def _read_package_version(rel: str) -> str:
    doc = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    return doc["version"]


def _contract_versions() -> dict[str, int]:
    http = json.loads((ROOT / "contracts" / "http.schema.json").read_text(encoding="utf-8"))
    defaults = json.loads((ROOT / "contracts" / "defaults.json").read_text(encoding="utf-8"))
    return {
        "http_schema": int(http["schema_version"]),
        "defaults": int(defaults["schema_version"]),
    }


def _lockfile_drift() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "uv.lock", "agent-cli/package-lock.json", "dashboard/package-lock.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def build_manifest() -> dict:
    return {
        "schema_version": 1,
        "python": {"bcg": _python_version()},
        "agent": {"bcg_agent": _read_package_version("agent-cli/package.json")},
        "dashboard": {"bcg_dashboard": _read_package_version("dashboard/package.json")},
        "contracts": _contract_versions(),
        "lockfiles_clean": not _lockfile_drift(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify consistency without writing")
    parser.add_argument("--output", type=Path, default=ROOT / "release-manifest.json")
    args = parser.parse_args()

    manifest = build_manifest()
    drift = _lockfile_drift()
    if drift:
        print("ERROR: lockfile drift detected (uncommitted changes):", file=sys.stderr)
        for line in drift:
            print(f"  {line}", file=sys.stderr)
        return 1

    if args.check:
        current = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
        if current != manifest:
            print(
                f"ERROR: {args.output} is stale; regenerate with scripts/release-manifest.py",
                file=sys.stderr,
            )
            return 1
        print(
            "OK: release manifest matches "
            f"bcg {manifest['python']['bcg']} / agent {manifest['agent']['bcg_agent']} / "
            f"contract {manifest['contracts']['http_schema']} / dashboard {manifest['dashboard']['bcg_dashboard']}"
        )
        return 0

    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
