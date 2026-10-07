"""Install once, then use the ordinary Codex TUI continuously."""

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
        description="Native Codex BCG memory: project hooks + MCP; optional context gateway.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser(
        "init", help="Install project-local hooks/MCP (review and trust in /hooks)"
    )
    init.add_argument("--project", type=Path, default=Path.cwd())
    init.add_argument("--mode", choices=["assist", "compact"], default="assist")
    init.add_argument("--env-file", type=Path, default=Path.home() / ".bcg/.env")
    init.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    init.add_argument("--allow-graph-upload", action="store_true")
    init.add_argument(
        "--compact-at",
        type=int,
        help="Optional native auto-compaction token threshold; compact mode only",
    )
    uninstall = sub.add_parser(
        "uninstall", help="Remove only BCG wiring; preserve private memories"
    )
    uninstall.add_argument("--project", type=Path, default=Path.cwd())
    for name in ("hook", "mcp", "status", "disable", "enable", "retry"):
        parser = sub.add_parser(name)
        parser.add_argument(
            "--config", type=Path, default=Path.cwd() / ".bcg-codex/config.json"
        )
    run = sub.add_parser(
        "run",
        help="Launch one native Codex TUI through the experimental context gateway",
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
                compact_at=args.compact_at,
            )
            print(
                f"Installed {config.mode} in {config.project}. Start codex, trust this project and review BCG hooks in /hooks.\nGraph config: {config.graph_config}\nPrivate data: {config.state_dir}\nDo not commit .bcg-codex/ or generated machine-local .codex wiring. Global Codex config/auth untouched."
            )
            return
        if args.command == "uninstall":
            from .install import uninstall

            uninstall(args.project)
            print("BCG project wiring removed. Memories retained; restart Codex.")
            return
        config = Config.load(args.config)
        if args.command == "hook":
            if os.environ.get("BCG_NATIVE_HOOKS_DISABLED") == "1":
                print("{}")
                return
            from .hooks import handle

            raw = sys.stdin.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError("hook input exceeds capture safety limit")
            print(json.dumps(handle(json.loads(raw), config), ensure_ascii=False))
        elif args.command == "mcp":
            from .mcp import serve

            serve(args.config)
        elif args.command == "run":
            from .gateway import run

            arguments = args.codex_args
            if arguments[:1] == ["--"]:
                arguments = arguments[1:]
            raise SystemExit(run(config, arguments, record_io=args.record_model_io))
        elif args.command in {"disable", "enable"}:
            raw = json.loads(args.config.read_text())
            raw["enabled"] = args.command == "enable"
            private_write(args.config, raw)
            print(
                f"BCG {args.command}d. An already-running Graph request may finish; restart Codex for config wiring changes."
            )
        else:
            from .store import Store

            store = Store(config.state_dir)
            if args.command == "retry":
                print(
                    f"Queued {store.retry()} failed events for a new graph attempt (may incur provider cost)."
                )
            else:
                print(json.dumps(store.status(), indent=2))
    except Exception as exc:
        if args.command == "hook":
            # Always advisory. No permission decisions, input rewriting or
            # synthetic Stop continuation; malformed input must not stop Codex.
            print(
                json.dumps(
                    {
                        "systemMessage": f"BCG hook unavailable ({type(exc).__name__}); Codex continues normally."
                    }
                )
            )
            return
        print(
            f"bcg codex: {exc if isinstance(exc, ValueError) else type(exc).__name__}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
