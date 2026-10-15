"""Initialize and launch the native Codex context gateway."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import Config, private_write


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="bcg codex",
        description="Native Codex with the BCG context gateway.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Initialize project-local gateway configuration")
    init.add_argument("--project", type=Path, default=Path.cwd())
    init.add_argument("--mode", choices=["gateway"], default="gateway")
    init.add_argument("--env-file", type=Path, default=Path.home() / ".bcg/.env")
    init.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    init.add_argument("--allow-graph-upload", action="store_true")
    uninstall = sub.add_parser(
        "uninstall", help="Disable the gateway; preserve configuration and memories"
    )
    uninstall.add_argument("--project", type=Path, default=Path.cwd())
    for name in ("status", "disable", "enable"):
        parser = sub.add_parser(name)
        parser.add_argument(
            "--config", type=Path, default=Path.cwd() / ".bcg-codex/config.json"
        )
    run = sub.add_parser(
        "run", help="Launch one native Codex TUI through the context gateway"
    )
    run.add_argument(
        "--config", type=Path, default=Path.cwd() / ".bcg-codex/config.json"
    )
    run.add_argument("--mode", choices=["gateway"], default="gateway")
    run.add_argument(
        "--record-model-io",
        action="store_true",
        help="Sensitive opt-in request/response logs; private local directory",
    )
    run.add_argument("codex_args", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    os.umask(0o077)
    try:
        if args.command == "init":
            from .install import install

            config = install(
                args.project,
                mode=args.mode,
                env_file=args.env_file,
                embedding_model=args.embedding_model,
                allow_upload=args.allow_graph_upload,
            )
            print(
                f"Initialized gateway in {config.project}.\n"
                "No hooks/MCP installed; project and global Codex config/auth untouched.\n"
                f"Graph config: {config.graph_config}\n"
                f"Graph credentials file: {config.env_file} (not modified)\n"
                f"Private data: {config.state_dir}\n"
                "Review graph-model.json and configure its api_key_env in the env file.\n"
                "From the project root, start: bcg codex run -- --no-daemon\n"
                "Keep private configuration and session records out of version control."
            )
            return
        if args.command == "uninstall":
            from .install import uninstall

            uninstall(args.project)
            print(
                "BCG gateway disabled; installation manifest removed. "
                "Configuration and memories retained. Exit any running gateway."
            )
            return
        config = Config.load(args.config)
        if args.command == "run":
            from .gateway import run

            arguments = args.codex_args
            if arguments[:1] == ["--"]:
                arguments = arguments[1:]
            raise SystemExit(run(config, arguments, record_io=args.record_model_io))
        if args.command in {"disable", "enable"}:
            raw = json.loads(args.config.read_text())
            raw["enabled"] = args.command == "enable"
            private_write(args.config, raw)
            print(
                f"BCG {args.command}d for future gateway launches. "
                "Exit Codex to stop an already-running gateway."
            )
            return
        state = Path(config.state_dir)
        print(
            json.dumps(
                {
                    "mode": "gateway",
                    "enabled": config.enabled,
                    "project": config.project,
                    "graph_config": config.graph_config,
                    "request_log": str(state / "gateway/requests.jsonl"),
                    "graph_dir": str(state / "graphs"),
                },
                indent=2,
            )
        )
    except Exception as exc:
        print(
            f"bcg codex: {exc if isinstance(exc, ValueError) else type(exc).__name__}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
