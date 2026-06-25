#!/usr/bin/env python3
"""
scripts/online_server.py
========================
A tiny, dependency-free HTTP server (Python stdlib only) that exposes the v3
streaming interface over JSON. You POST one turn at a time; it pushes the turn
into a SessionManager and returns the current belief graph.

Endpoints
---------
  GET  /health
        -> {"status": "ok", "active": [...problem_ids...], "all": [...]}

  POST /turn
        body: one turn dict (see STREAMING.md §2), e.g.
              {"problem_id": "p1", "role": "user", "content": "..."}
              {"problem_id": "p1", "role": "assistant", "content": "...",
               "is_trajectory_end": true}
        -> the current belief-graph snapshot for that problem_id.
           When the turn carries is_trajectory_end=true the trajectory is
           finalized and the returned snapshot is the COMPLETE graph.

  POST /turns
        body: either a JSON array of turn dicts, or NDJSON (one dict per line).
        -> {"pushed": n, "finalized": [...], "latest": {problem_id: snapshot}}

  POST /finalize
        body: {"problem_id": "p1"}
        -> the FINAL belief-graph snapshot.

  GET  /graph?problem_id=p1
        -> the latest snapshot for that trajectory (404 if unknown).

All artifacts are still written to <output-dir>/<problem_id>/ exactly as the
file/CLI driver writes them (trajectory_stream.jsonl, trajectory.json,
belief_graph.jsonl, belief_graph_latest.json, result.json, ...).

Concurrency
-----------
The engine keeps process-global / shared state (token ledger, log paths) that
each session swaps in and out around every call; that swap is NOT thread-safe.
This server therefore serializes ALL engine interactions behind a single lock,
so it is safe to run with ThreadingHTTPServer (health checks etc. never block
on it), but only one push/finalize executes at a time. For a sequential
data-generation stream this is exactly the desired behaviour. If you need true
parallelism, run several processes, each owning a disjoint set of problem_ids.

Run
---
  python scripts/online_server.py --config model_config.json \
      --model-key gpt-5.5 --host 127.0.0.1 --port 8848 --output-dir outputs_stream

  # then, from anywhere:
  curl -s -X POST localhost:8848/turn -H 'content-type: application/json' \
       -d '{"problem_id":"p1","role":"user","content":"hello"}'
  curl -s -X POST localhost:8848/turn -H 'content-type: application/json' \
       -d '{"problem_id":"p1","role":"assistant","content":"hi","is_trajectory_end":true}'
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belief_graph.online import SessionManager, TrajectoryClosedError   # noqa: E402
from belief_graph.stream import StreamOptions                           # noqa: E402


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


def make_handler(manager: SessionManager, lock: threading.Lock, *, quiet: bool = False):
    """Build a request-handler class bound to one manager + serialization lock."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "ConstructBeliefsOnline/3.0"

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
                self._send(200, {"status": "ok",
                                 "active": manager.active_problem_ids(),
                                 "all": manager.all_problem_ids()})
                return
            if url.path == "/graph":
                pid = (parse_qs(url.query).get("problem_id") or [None])[0]
                if not pid:
                    self._send(400, {"error": "missing ?problem_id="})
                    return
                with lock:
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
            try:
                raw = self._read_body()
                if url.path == "/turn":
                    turn = json.loads(raw.decode("utf-8")) if raw else {}
                    if not isinstance(turn, dict):
                        raise ValueError("body must be a single JSON object (one turn)")
                    with lock:
                        graph = manager.push(turn)
                    self._send(200, graph)
                    return

                if url.path == "/turns":
                    turns = _parse_turns_body(raw)
                    finalized: List[str] = []
                    latest: Dict[str, Any] = {}
                    with lock:
                        for t in turns:
                            snap = manager.push(t)
                            pid = snap.get("problem_id")
                            latest[pid] = snap
                            if snap.get("finalized"):
                                finalized.append(pid)
                    self._send(200, {"pushed": len(turns),
                                     "finalized": finalized, "latest": latest})
                    return

                if url.path == "/finalize":
                    body = json.loads(raw.decode("utf-8")) if raw else {}
                    pid = body.get("problem_id") if isinstance(body, dict) else None
                    if not pid:
                        raise ValueError("body must be {\"problem_id\": \"...\"}")
                    with lock:
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
    lock = threading.Lock()
    httpd = ThreadingHTTPServer((host, port), make_handler(manager, lock, quiet=quiet))
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
        context_chars=args.context_chars,
    )
    return SessionManager(
        config_path=args.config, model_key=args.model_key,
        embedding_key=args.embedding_key, output_root=Path(args.output_dir),
        options=options,
    )


def main():
    p = argparse.ArgumentParser(description="construct_beliefs v3 streaming HTTP server")
    p.add_argument("--host", default="0.0.0.0", help="interface to bind to (default: localhost only: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8848)
    p.add_argument("--config", "-c", default="model_config.json")
    p.add_argument("--output-dir", "-o", default="outputs_stream")
    p.add_argument("--model-key", default="gpt-5.5")
    p.add_argument("--embedding-key", default="embedding")
    p.add_argument("--evidence-mode", choices=["sentence", "excerpt"], default="sentence")
    p.add_argument("--use-clustering", default=False, action="store_true")
    p.add_argument("--cluster-threshold", type=float, default=0.6)
    p.add_argument("--cluster-min-sentences", type=int, default=4)
    p.add_argument("--cluster-buffer", type=int, default=0)
    p.add_argument("--merge-strategy", choices=["embedding", "llm", "off"], default="embedding")
    p.add_argument("--merge-threshold", type=float, default=0.86)
    p.add_argument("--incremental-merge", dest="incremental_merge",
                   default=True, action="store_true",
                   help="Per-turn embedding-only merge (no LLM verification). Default: ON.")
    p.add_argument("--no-incremental-merge", dest="incremental_merge", action="store_false",
                   help="Disable the per-turn incremental merge.")
    p.add_argument("--incremental-merge-threshold", type=float, default=0.86,
                   help="Cosine threshold for the per-turn incremental merge. Default 0.86.")
    p.add_argument("--context-chars", type=int, default=100000)
    p.add_argument("--quiet", "-q", default=False, action="store_true")
    args = p.parse_args()

    manager = build_manager(args)
    httpd = serve(manager, args.host, args.port, quiet=args.quiet)
    print(f"[online-server] listening on http://{args.host}:{args.port}  "
          f"(POST /turn, /turns, /finalize ; GET /graph, /health)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[online-server] shutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
