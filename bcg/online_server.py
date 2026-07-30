#!/usr/bin/env python3
"""
bcg/online_server.py
====================
Single dependency-free HTTP server for BOTH belief-context-graph
construction backends. Pick one with the first positional argument, same as
``bcg/run.py``:

  python bcg/online_server.py light      --config bcg/model_config.json --port 8848
  python bcg/online_server.py api_based  --config bcg/model_config.json --port 8848

The HTTP request-handling code below (routing, (de)serialisation, endpoints)
is identical for both backends — only the config wiring / SessionManager
construction differs, so that part is factored into one shared handler and
each backend gets its own small ``build_manager`` + argument parser.

The server supports two ways to feed data:

  1. True online mode: POST one completed turn at a time to /turn.
  2. run.py-compatible mode: POST a whole trajectory / messages object /
     multi-session QA payload to /input; it is normalised with the same
     loader used by run.py, then ingested turn-by-turn through the online
     engine.

Endpoints
---------
  GET  /health
        -> {"status": "ok", "active": [...problem_ids...], "all": [...]}

  POST /turn
        body: one turn dict, e.g.
              {"problem_id": "p1", "role": "user", "content": "..."}
              {"problem_id": "p1", "role": "assistant", "content": "...",
               "is_trajectory_end": true}
        -> the current belief-graph snapshot for that problem_id.

  POST /turns
        body: either a JSON array of online turn dicts, or NDJSON.
        -> {"pushed": n, "finalized": [...], "latest": {problem_id: snapshot}}

  POST /input
        body: any JSON shape accepted by run.py/loaders.py:
              {"trajectory": [...]}, {"messages": [...]}, a bare message list,
              or multi-session QA data with sessions/session_ids/dates.
        query params:
              ?item=<id-or-index>       process one item only
              ?keep_order=1             do not chronologically sort sessions
              ?finalize=0               ingest but do not auto-finalize
        -> {"items": n, "finalized": [...], "latest": {...}}

  POST /finalize
        body: {"problem_id": "p1"}
        -> the FINAL belief-graph snapshot.

  GET  /graph?problem_id=p1
        -> the latest snapshot for that trajectory (404 if unknown).

Concurrency
-----------
Each problem_id is backed by its own StreamingTrajectorySession, which guards
its own mutable state (belief graph, builder, output files) with its own
lock, so different problem_ids run fully concurrently on
ThreadingHTTPServer's per-request threads while turns for the SAME
problem_id stay strictly in arrival order.

Run
---
  python bcg/online_server.py light --config bcg/model_config.json \\
      --model-key gpt-5.5 --host 127.0.0.1 --port 8848 --output-dir outputs_stream

  python bcg/online_server.py api_based --config bcg/model_config.json \\
      --model-key gpt-5.5 --host 127.0.0.1 --port 8848 --output-dir outputs_stream

  curl -s -X POST localhost:8848/turn -H 'content-type: application/json' \\
       -d '{"problem_id":"p1","role":"user","content":"hello"}'
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# Allow running as `python bcg/online_server.py ...` from the project root
# (the parent directory of this `bcg` package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bcg.cli_help import RichArgumentParser  # noqa: E402
from bcg.construct.dispatch import (  # noqa: E402
    DEFAULT_BACKEND,
    split_backend_args,
)


# ---------------------------------------------------------------------------
# Shared HTTP plumbing (backend-agnostic: works against any SessionManager
# exposing push / push_many / push_input / finalize / get_graph /
# active_problem_ids / all_problem_ids).
# ---------------------------------------------------------------------------

def _parse_turns_body(raw: bytes) -> List[Dict[str, Any]]:
    """Accept a JSON array of dicts OR NDJSON (one dict per line)."""
    text = raw.decode("utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("body must be a JSON array of turn objects")
        return [t for t in data if isinstance(t, dict)]
    turns: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            turns.append(obj)
    return turns


def _parse_json_body(raw: bytes) -> Any:
    text = raw.decode("utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def _query_bool(qs: Dict[str, List[str]], name: str, default: bool) -> bool:
    vals = qs.get(name)
    if not vals:
        return default
    s = str(vals[-1]).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _query_str(qs: Dict[str, List[str]], name: str) -> Optional[str]:
    vals = qs.get(name)
    if not vals:
        return None
    val = vals[-1]
    return val if val != "" else None


def make_handler(manager, trajectory_closed_error: type, *, quiet: bool = False):
    """Build a request-handler class bound to one manager.

    ``trajectory_closed_error`` is the backend-specific ``TrajectoryClosedError``
    class, passed in so this module never needs to import both backends just
    to catch it.
    """

    class Handler(BaseHTTPRequestHandler):
        server_version = "ConstructBeliefsOnline/3.2"

        # -- helpers --------------------------------------------------------
        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # The caller may time out while a long graph update is still
                # completing. The update has already been committed, so a
                # disconnected response socket is not a server-side 500.
                return

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length > 0 else b""

        def log_message(self, fmt, *args):  # quieter access log to stderr
            if not quiet:
                sys.stderr.write("  [http] %s - %s\n" % (self.address_string(), fmt % args))

        # -- GET ------------------------------------------------------------
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                self._send(200, {
                    "status": "ok",
                    "active": manager.active_problem_ids(),
                    "all": manager.all_problem_ids(),
                })
                return
            if url.path == "/graph":
                pid = (parse_qs(url.query).get("problem_id") or [None])[0]
                if not pid:
                    self._send(400, {"error": "missing ?problem_id="})
                    return
                graph = manager.get_graph(pid)
                if graph is None:
                    self._send(404, {"error": f"no trajectory for problem_id {pid!r}"})
                else:
                    self._send(200, graph)
                return
            self._send(404, {"error": f"unknown path {url.path!r}"})

        # -- POST -----------------------------------------------------------
        def do_POST(self):
            url = urlparse(self.path)
            qs = parse_qs(url.query)
            try:
                raw = self._read_body()

                if url.path == "/turn":
                    turn = _parse_json_body(raw)
                    if not isinstance(turn, dict):
                        raise ValueError("body must be a single JSON object (one turn)")
                    graph = manager.push(turn)
                    self._send(200, graph)
                    return

                if url.path == "/turns":
                    turns = _parse_turns_body(raw)
                    result = manager.push_many(turns)
                    self._send(200, result)
                    return

                if url.path in {"/input", "/run"}:
                    data = _parse_json_body(raw)
                    keep_order = _query_bool(qs, "keep_order", False)
                    finalize = _query_bool(qs, "finalize", True)
                    item_selector = _query_str(qs, "item")
                    result = manager.push_input(
                        data,
                        keep_order=keep_order,
                        item_selector=item_selector,
                        finalize=finalize,
                    )
                    self._send(200, result)
                    return

                if url.path == "/finalize":
                    body = _parse_json_body(raw)
                    pid = body.get("problem_id") if isinstance(body, dict) else None
                    if not pid:
                        raise ValueError("body must be {\"problem_id\": \"...\"}")
                    graph = manager.finalize(pid)
                    self._send(200, graph)
                    return

                if url.path == "/release":
                    body = _parse_json_body(raw)
                    pid = body.get("problem_id") if isinstance(body, dict) else None
                    if not pid:
                        raise ValueError("body must be {\"problem_id\": \"...\"}")
                    self._send(200, manager.release(pid))
                    return

                self._send(404, {"error": f"unknown path {url.path!r}"})

            except json.JSONDecodeError as e:
                self._send(400, {"error": f"invalid JSON: {e}"})
            except trajectory_closed_error as e:
                self._send(409, {"error": str(e)})
            except KeyError as e:
                self._send(404, {"error": str(e)})
            except ValueError as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 - report, don't crash the server
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def serve(manager, trajectory_closed_error: type, host: str, port: int, *, quiet: bool = False):
    """Create (but do not block) a ThreadingHTTPServer. Caller runs serve_forever."""
    httpd = ThreadingHTTPServer(
        (host, port), make_handler(manager, trajectory_closed_error, quiet=quiet)
    )
    return httpd


def _serve_forever(manager, trajectory_closed_error: type, args) -> None:
    httpd = serve(manager, trajectory_closed_error, args.host, args.port, quiet=args.quiet)
    print(
        f"[online-server] listening on http://{args.host}:{args.port}  "
        f"(POST /turn, /turns, /input, /finalize, /release ; GET /graph, /health)"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[online-server] shutting down")
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# light backend
# ---------------------------------------------------------------------------

def _run_light(argv: list[str]) -> None:
    from bcg.construct.light.online import SessionManager, TrajectoryClosedError

    p = argparse.ArgumentParser(
        prog="bcg/online_server.py light",
        description="construct_beliefs v3 streaming HTTP server (light backend).",
    )
    p.add_argument("--host", default="0.0.0.0", help="interface to bind to (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8848)
    p.add_argument("--config", "-c", default="bcg/model_config.json")
    p.add_argument("--output-dir", "-o", default="outputs_stream")
    p.add_argument("--model-key", default="gpt-5.5")
    p.add_argument("--embedding-key", default="embedding")
    p.add_argument("--quiet", "-q", default=False, action="store_true")
    args = p.parse_args(argv)

    manager = SessionManager(
        config_path=args.config,
        model_key=args.model_key,
        embedding_key=args.embedding_key,
        output_root=Path(args.output_dir),
    )
    _serve_forever(manager, TrajectoryClosedError, args)


# ---------------------------------------------------------------------------
# api_based backend
# ---------------------------------------------------------------------------

def _run_api_based(argv: list[str]) -> None:
    from bcg.construct.api_based.online import (
        SessionManager,
        TrajectoryClosedError,
        resolve_dated_output_root,
    )
    from bcg.construct.api_based.stream import StreamOptions

    p = argparse.ArgumentParser(
        prog="bcg/online_server.py api_based",
        description="construct_beliefs v3 streaming HTTP server (api_based backend).",
    )
    p.add_argument("--host", default="0.0.0.0", help="interface to bind to (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8848)
    p.add_argument("--config", "-c", default="bcg/model_config.json")
    p.add_argument(
        "--output-dir", "-o", default="outputs_stream",
        help="output root. Basenames like outputs_2026_7_6 auto-roll to "
             "today's outputs_Y_M_D for new sessions; templates such as "
             "outputs_{Y}_{m}_{d} or outputs_{date} are also supported. "
             "Plain outputs_stream stays fixed.",
    )
    p.add_argument("--model-key", default="gpt-5.5")
    p.add_argument("--embedding-key", default="embedding")

    p.add_argument("--evidence-mode", choices=["sentence", "excerpt"], default="sentence")

    p.add_argument("--incremental-merge", dest="incremental_merge",
                   default=True, action="store_true",
                   help="Per-turn embedding merge. Default: ON.")
    p.add_argument("--no-incremental-merge", dest="incremental_merge",
                   action="store_false", help="Disable the per-turn incremental merge.")
    p.add_argument("--incremental-merge-threshold", type=float, default=0.86,
                   help="Cosine threshold for the per-turn incremental merge. Default: 0.86.")
    p.add_argument("--verify-merge", dest="verify_merge",
                   default=True, action="store_true",
                   help="LLM-verify and rewrite per-turn embedding merge groups. "
                        "Default: ON, matching run.py.")
    p.add_argument("--no-verify-merge", dest="verify_merge",
                   action="store_false",
                   help="Disable LLM verification for the per-turn incremental merge.")
    p.add_argument("--context-chars", type=int, default=100000)
    p.add_argument("--min-content-len", type=int, default=0)

    p.add_argument("--quiet", "-q", default=False, action="store_true")
    args = p.parse_args(argv)

    options = StreamOptions(
        evidence_mode=args.evidence_mode,
        incremental_merge=args.incremental_merge,
        incremental_merge_threshold=args.incremental_merge_threshold,
        verify_merge=args.verify_merge,
        context_chars=args.context_chars,
        min_content_len=args.min_content_len,
    )
    manager = SessionManager(
        config_path=args.config,
        model_key=args.model_key,
        embedding_key=args.embedding_key,
        output_root=Path(args.output_dir),
        options=options,
    )
    print(f"[online-server] output-tpl = {Path(args.output_dir)}")
    print(f"[online-server] output-dir = {resolve_dated_output_root(args.output_dir)}")
    _serve_forever(manager, TrajectoryClosedError, args)


_BACKENDS = {"light": _run_light, "api_based": _run_api_based}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        parser = RichArgumentParser(
            prog="bcg construct server",
            description="construct_beliefs v3 streaming belief-graph HTTP server.",
            epilog="Run 'bcg construct server <backend> --help' for a backend's "
                   "full option list. If omitted, the backend defaults to "
                   f"{DEFAULT_BACKEND!r} for compatibility.",
        )
        parser.add_argument(
            "backend",
            choices=list(_BACKENDS),
            nargs="?",
            default=DEFAULT_BACKEND,
            help=f"Which construct backend to use (default: {DEFAULT_BACKEND}).",
        )
        parser.print_help()
        raise SystemExit(0)

    try:
        backend, rest = split_backend_args(argv, backends=_BACKENDS)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

    _BACKENDS[backend](rest)


if __name__ == "__main__":
    main()
