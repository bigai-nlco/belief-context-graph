#!/usr/bin/env python3
"""
scripts/online_server.py
========================
Dependency-free HTTP server for the same streaming pipeline used by ``run.py``.

The server supports two ways to feed data:

  1. True online mode: POST one completed turn at a time to /turn.
  2. run.py-compatible mode: POST a whole trajectory / messages object /
     multi-session QA payload to /input; it is normalised with the same loader
     used by run.py, then ingested turn-by-turn through the online engine.

Endpoints
---------
  GET  /health
        -> {"status": "ok", "active": [...problem_ids...], "all": [...]}

  POST /turn
        body: one turn dict, e.g.
              {"problem_id": "p1", "role": "user", "content": "..."}
              {"problem_id": "p1", "role": "assistant", "content": "...",
               "is_trajectory_end": true}
        Optional fields such as date/session_date and has_answer are passed
        through exactly like the local run.py pipeline.
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

All artifacts are written to <resolved-output-dir>/<problem_id>/. If
<output-dir> is a daily root such as outputs_2026_7_6 or a template such as
outputs_{Y}_{m}_{d}, new sessions resolve it at creation time so a long-running
server rolls over after midnight without restart:
trajectory_stream.jsonl, trajectory.json, belief_graph.jsonl,
belief_graph_latest.json, result.json, final_graph.json, events.jsonl,
token_usage.*, and logs/.

Concurrency
-----------
Each problem_id is backed by its own StreamingTrajectorySession, which now
guards its own mutable state (belief graph, builder, output files) with its
own lock, and binds its own token-usage tracker / audit-log paths through
belief_graph.llm's context-local helpers instead of process-global state.
Because of that, this server no longer needs a single global lock: turns for
the SAME problem_id are still processed strictly in arrival order (whichever
request gets to that session first), while turns for DIFFERENT problem_ids
run fully concurrently on ThreadingHTTPServer's per-request threads. /turns
and /input additionally fan out their distinct problem_ids/items across a
small thread pool so one batch request doesn't itself serialize unrelated
trajectories.

Run
---
  python scripts/online_server.py --config model_config.json \
      --model-key gpt-5.5 --host 127.0.0.1 --port 8848 --output-dir outputs_stream

  curl -s -X POST localhost:8848/turn -H 'content-type: application/json' \
       -d '{"problem_id":"p1","role":"user","content":"hello"}'

  curl -s -X POST localhost:8848/input -H 'content-type: application/json' \
       --data-binary @data.json
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bcg.cli_help import RichArgumentParser
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bcg.construct.online import (  # noqa: E402
    SessionManager,
    TrajectoryClosedError,
    resolve_dated_output_root,
)
from bcg.construct.stream import StreamOptions  # noqa: E402


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


def make_handler(manager: SessionManager, *, quiet: bool = False):
    """Build a request-handler class bound to one manager.

    No serialization lock here: concurrency safety now lives inside
    SessionManager / StreamingTrajectorySession (see construct/online.py),
    so different problem_ids can be handled by different threads with no
    coordination required at this layer.
    """

    class Handler(BaseHTTPRequestHandler):
        server_version = "ConstructBeliefsOnline/3.2"

        # -- helpers --------------------------------------------------------
        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
                    # Different problem_ids in this batch run concurrently;
                    # each problem_id's own turns stay in order. See
                    # SessionManager.push_many.
                    result = manager.push_many(turns)
                    self._send(200, result)
                    return

                if url.path in {"/input", "/run"}:
                    data = _parse_json_body(raw)
                    keep_order = _query_bool(qs, "keep_order", False)
                    finalize = _query_bool(qs, "finalize", True)
                    item_selector = _query_str(qs, "item")
                    # Different items (problem_ids) in this payload run
                    # concurrently; see SessionManager.push_input.
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

                self._send(404, {"error": f"unknown path {url.path!r}"})

            except json.JSONDecodeError as e:
                self._send(400, {"error": f"invalid JSON: {e}"})
            except TrajectoryClosedError as e:
                self._send(409, {"error": str(e)})
            except KeyError as e:
                self._send(404, {"error": str(e)})
            except ValueError as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 - report, don't crash the server
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def serve(manager: SessionManager, host: str, port: int, *, quiet: bool = False):
    """Create (but do not block) a ThreadingHTTPServer. Caller runs serve_forever."""
    httpd = ThreadingHTTPServer((host, port), make_handler(manager, quiet=quiet))
    return httpd


def build_manager(args) -> SessionManager:
    options = StreamOptions(
        evidence_mode=args.evidence_mode,
        use_clustering=args.use_clustering,
        cluster_threshold=args.cluster_threshold,
        cluster_min_sentences=args.cluster_min_sentences,
        cluster_buffer=args.cluster_buffer,
        merge_strategy=args.merge_strategy,
        merge_threshold=args.merge_threshold,
        incremental_merge=args.incremental_merge,
        incremental_merge_threshold=args.incremental_merge_threshold,
        verify_merge=args.verify_merge,
        factor_similarity_threshold=args.factor_similarity_threshold,
        factor_input_confidence_threshold=args.factor_input_confidence_threshold,
        context_chars=args.context_chars,
        min_content_len=args.min_content_len,
    )
    return SessionManager(
        config_path=args.config,
        model_key=args.model_key,
        embedding_key=args.embedding_key,
        output_root=Path(args.output_dir),
        options=options,
    )


def main(argv: list[str] | None = None) -> None:
    p = RichArgumentParser(
        prog="bcg construct server",
        description="construct_beliefs v3 streaming HTTP server",
    )
    p.add_argument("--host", default="0.0.0.0", help="interface to bind to (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8848)
    p.add_argument("--config", "-c", default="bcg/model_config.json")
    p.add_argument(
        "--output-dir",
        "-o",
        default="outputs_stream",
        help="output root. Basenames like outputs_2026_7_6 auto-roll to "
             "today's outputs_Y_M_D for new sessions; templates such as "
             "outputs_{Y}_{m}_{d} or outputs_{date} are also supported. "
             "Plain outputs_stream stays fixed.",
    )
    p.add_argument("--model-key", default="gpt-5.5")
    p.add_argument("--embedding-key", default="embedding")

    p.add_argument("--evidence-mode", choices=["sentence", "excerpt"], default="sentence")
    p.add_argument("--use-clustering", default=False, action="store_true")
    p.add_argument("--cluster-threshold", type=float, default=0.6)
    p.add_argument("--cluster-min-sentences", type=int, default=4)
    p.add_argument("--cluster-buffer", type=int, default=0)

    # Defaults intentionally match the current run.py in this project.
    p.add_argument("--merge-strategy", choices=["embedding", "llm", "off"], default="off")
    p.add_argument("--merge-threshold", type=float, default=0.86)
    p.add_argument(
        "--incremental-merge",
        dest="incremental_merge",
        default=True,
        action="store_true",
        help="Per-turn embedding merge. Default: ON.",
    )
    p.add_argument(
        "--no-incremental-merge",
        dest="incremental_merge",
        action="store_false",
        help="Disable the per-turn incremental merge.",
    )
    p.add_argument(
        "--incremental-merge-threshold",
        type=float,
        default=0.86,
        help="Cosine threshold for the per-turn incremental merge. Default: 0.86.",
    )
    p.add_argument(
        "--verify-merge",
        dest="verify_merge",
        default=True,
        action="store_true",
        help="LLM-verify and rewrite per-turn embedding merge groups. Default: ON, matching run.py.",
    )
    p.add_argument(
        "--no-verify-merge",
        dest="verify_merge",
        action="store_false",
        help="Disable LLM verification for the per-turn incremental merge.",
    )
    p.add_argument(
        "--factor-similarity-threshold",
        type=float,
        default=0.80,
        help="Cosine threshold for reusing an existing same-type factor by "
             "activation_condition[note] embedding similarity.",
    )
    p.add_argument(
        "--factor-input-confidence-threshold",
        type=float,
        default=0.5,
        help="Only activate a depends_on/contradicts factor when its input node "
             "confidence is greater than this value. Default: 0.5.",
    )
    p.add_argument("--context-chars", type=int, default=100000)
    p.add_argument("--min-content-len", type=int, default=0)

    p.add_argument("--quiet", "-q", default=False, action="store_true")
    args = p.parse_args(argv)

    manager = build_manager(args)
    httpd = serve(manager, args.host, args.port, quiet=args.quiet)
    print(
        f"[online-server] listening on http://{args.host}:{args.port}  "
        f"(POST /turn, /turns, /input, /finalize ; GET /graph, /health)"
    )
    print(f"[online-server] output-tpl = {Path(args.output_dir)}")
    print(f"[online-server] output-dir = {resolve_dated_output_root(args.output_dir)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[online-server] shutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
