#!/usr/bin/env python3
"""Backend for belief_graph_stream_viewer.html.

Does two jobs:
  1. Serves the static viewer + stream output files (with no-store on the
     jsonl/json so the browser can live-tail growing files).
  2. Exposes a tiny JSON API so the viewer's "Run new case" button can launch
     a construct run (configured via --start-sh), watch for the new output
     <id>/ directory it produces, and stream it live.

This is a deprecated read-only devtool (see dashboard/README.md). It reads
the belief-graph stream from the repository's default construct output
directory (--stream-dir, default: ../outputs when present, else the
bundled demo_output directory) and maps the URL prefix /outputs_stream/
onto it.

Run it from the dashboard/bcg_viewer directory:

    python3 serve_viewer.py --start-sh /path/to/bcg/scripts/start_construct.sh

Then open  http://127.0.0.1:8123/belief_graph_stream_viewer.html

Options:
    --host / --port                 bind address (default 0.0.0.0:8123)
    --root DIR                      dir to serve (default: this file's dir)
    --stream-dir DIR                stream output root; outputs_2026_7_6
                                    auto-rolls daily, or use templates like
                                    outputs_{Y}_{m}_{d}
                                    (default: the repository's outputs
                                    directory if present, else the bundled demo_output directory)
    --start-sh PATH                 script the Run button launches
    --default-max-problems N        default problems per click (default 1)
    --login-shell                   run start.sh via `bash -lc` (loads ~/.bashrc,
                                    so conda auto-activation etc. can kick in)

API:
    GET  /api/ping                  -> {ok, run_enabled, start_sh}
    GET  /api/manifest              -> live manifest (same shape as manifest.json)
    GET  /api/result/<stream:id>    -> download that sample's result.json
    POST /api/run                   body {max_problems?, extra_args?,
                                          belief_graph_mode?} -> {run_id}
    GET  /api/run/<id>              -> {status, returncode, elapsed, new_samples, log_tail}
    GET  /api/sample/<rollout:id>   -> synthetic no-graph sample from trajectories.jsonl
    POST /api/run/<id>/stop         -> {ok}
"""

import argparse
import contextlib
import glob
import importlib.util
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.parse
import uuid
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
_CURRENT_STREAM_DIR = os.path.join(_REPO_ROOT, "outputs")
_BUNDLED_DEMO_DIR = os.path.join(HERE, "demo_output")
# Prefer the repository's construct output when present; otherwise run standalone with the bundled demo.
DEFAULT_STREAM_DIR = next(
    (
        p
        for p in (_CURRENT_STREAM_DIR,)
        if os.path.isdir(p)
    ),
    _BUNDLED_DEMO_DIR,
)
STREAM_URL_PREFIX = "/outputs_stream"
_DATED_STREAM_RE = re.compile(
    r"^(?P<prefix>outputs_)(?:(?P<year>\d{4})_)?(?P<month>\d{1,2})_(?P<day>\d{1,2})$"
)


def _resolve_dated_dir(path, now=None):
    """Resolve daily stream dirs while leaving plain dirs such as outputs_stream unchanged."""
    local_now = now or datetime.now().astimezone()
    values = {
        "Y": f"{local_now.year:04d}",
        "y": f"{local_now.year % 100:02d}",
        "m": str(local_now.month),
        "mm": f"{local_now.month:02d}",
        "month": str(local_now.month),
        "d": str(local_now.day),
        "dd": f"{local_now.day:02d}",
        "day": str(local_now.day),
        "date": f"{local_now.year:04d}-{local_now.month:02d}-{local_now.day:02d}",
    }
    raw = os.fspath(path)
    if any(("{" + key + "}") in raw for key in values):
        try:
            return os.path.abspath(raw.format(**values))
        except (KeyError, ValueError):
            return os.path.abspath(raw)

    root = os.path.abspath(raw)
    name = os.path.basename(root)
    m = _DATED_STREAM_RE.match(name)
    if not m:
        return root
    month = (
        f"{local_now.month:02d}" if len(m.group("month")) == 2 else str(local_now.month)
    )
    day = f"{local_now.day:02d}" if len(m.group("day")) == 2 else str(local_now.day)
    year = f"{local_now.year:04d}_" if m.group("year") else ""
    return os.path.join(
        os.path.dirname(root), f"{m.group('prefix')}{year}{month}_{day}"
    )


def _current_stream_dir():
    stream_dir = _resolve_dated_dir(CFG.get("stream_dir_template") or CFG["stream_dir"])
    CFG["stream_dir"] = stream_dir
    return stream_dir


def _lan_ip():
    """Best-effort primary LAN IPv4 (no traffic actually sent)."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()


# ---- config filled in by main() ------------------------------------------
CFG = {
    "root": HERE,
    "stream_dir": DEFAULT_STREAM_DIR,
    "stream_dir_template": DEFAULT_STREAM_DIR,
    "start_sh": "",
    "default_max_problems": 1,
    "default_save_alias": "demo_test",
    "login_shell": False,
    "data_file": "",
    "rollout_output_dir": "",
}
_TIMINGS_CACHE = {}  # graph_problem_id -> result dict
RUNS_DIR = os.path.join(HERE, ".viewer_runs")
ROLLOUT_SAMPLE_PREFIX = "rollout:"
BELIEF_GRAPH_MODES = {"none", "augment", "only"}


# ---- import the manifest builder from the sibling script ------------------
def _load_manifest_builder():
    path = os.path.join(HERE, "build_stream_manifest.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("build_stream_manifest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MANIFEST = _load_manifest_builder()

# ---- run registry ---------------------------------------------------------
RUNS = {}  # run_id -> dict
RUNS_LOCK = threading.Lock()


def _list_sample_dirs():
    stream_dir = _current_stream_dir()
    try:
        names = os.listdir(stream_dir)
    except FileNotFoundError:
        return set()
    out = set()
    for n in names:
        if n.startswith("."):
            continue
        if os.path.isdir(os.path.join(stream_dir, n)):
            out.add(n)
    return out


def _new_samples(before):
    stream_dir = _current_stream_dir()
    now = _list_sample_dirs()
    fresh = []
    for name in now - before:
        d = os.path.join(stream_dir, name)
        if os.path.exists(os.path.join(d, "trajectory_stream.jsonl")):
            try:
                mtime = os.path.getmtime(d)
            except OSError:
                mtime = 0
            fresh.append((mtime, name))
    fresh.sort(reverse=True)  # newest first
    return [name for _, name in fresh]


def _entry_completed_at(entry):
    ts = entry.get("completed_at")
    if isinstance(ts, (int, float)):
        return float(ts)
    iso = entry.get("completed_iso")
    if isinstance(iso, str) and iso:
        try:
            return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _iso_from_ts(ts):
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), UTC).isoformat()
    return datetime.now(UTC).isoformat()


def _rollout_sample_id(path, line_no):
    rel = os.path.relpath(path, CFG["rollout_output_dir"])
    return f"{ROLLOUT_SAMPLE_PREFIX}{urllib.parse.quote(rel, safe='')}:{line_no}"


def _decode_rollout_sample_id(sample_id):
    if not sample_id.startswith(ROLLOUT_SAMPLE_PREFIX):
        return None
    rest = sample_id[len(ROLLOUT_SAMPLE_PREFIX) :]
    try:
        rel_enc, line_s = rest.rsplit(":", 1)
        line_no = int(line_s)
    except (ValueError, TypeError) as exc:
        raise ValueError("bad rollout sample id") from exc
    rel = urllib.parse.unquote(rel_enc)
    root = os.path.abspath(CFG["rollout_output_dir"])
    path = os.path.abspath(os.path.join(root, rel))
    if path != root and not path.startswith(root + os.sep):
        raise ValueError("rollout sample path escapes output root")
    return path, line_no


def _rollout_trajectory_files(save_alias=None):
    root = CFG["rollout_output_dir"]
    pat = os.path.join(root, "**", "averitec", "trajectories*.jsonl")
    files = []
    for f in glob.glob(pat, recursive=True):
        if save_alias:
            # output dirs are <model>_<thinking>_<save_alias>; allow an exact
            # directory name too for manually arranged outputs.
            out_dir = os.path.basename(os.path.dirname(os.path.dirname(f)))
            if out_dir != save_alias and not out_dir.endswith("_" + save_alias):
                continue
        try:
            mtime = os.path.getmtime(f)
        except OSError:
            mtime = 0
        files.append((mtime, f))
    files.sort(reverse=True)
    return [f for _, f in files]


def _read_rollout_entry(sample_id):
    decoded = _decode_rollout_sample_id(sample_id)
    if not decoded:
        return None
    path, line_no = decoded
    if line_no < 0:
        raise ValueError("bad rollout sample line")
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i != line_no:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json in rollout sample: {exc}") from exc
    raise FileNotFoundError("rollout sample line not found")


def _timing_steps_from_entry(entry):
    sample = entry.get("sample") or {}
    steps = []
    for m in sample.get("model_io") or []:
        t = m.get("timings") or {}
        steps.append(
            {"llm": t.get("llm"), "tool": t.get("tool"), "graph": t.get("graph")}
        )
    return steps


def _rollout_entry_info(path, line_no, entry):
    sample = entry.get("sample") or {}
    traj = (
        sample.get("trajectory") if isinstance(sample.get("trajectory"), list) else []
    )
    claim = (
        entry.get("question")
        or entry.get("problem_id")
        or sample.get("trajectory_id")
        or ""
    )
    task_id = entry.get("task_id") or ""
    label = (
        f"#{task_id} · {claim}"
        if task_id and claim
        else (claim or task_id or "rollout sample")
    )
    return {
        "id": _rollout_sample_id(path, line_no),
        "claim": claim,
        "claim_id": str(task_id) if task_id is not None else "",
        "task_id": str(task_id) if task_id is not None else "",
        "label": label[:140],
        "n_turns": len(traj),
        "n_nodes": 0,
        "source": "rollout",
        "graph_problem_id": sample.get("graph_problem_id") or "",
        "completed_at": _entry_completed_at(entry),
    }


def _new_rollout_samples(rec):
    """Discover completed rollout trajectory rows for belief-graph-mode=none.

    The graph server does not write trajectory_stream.jsonl/belief_graph.jsonl in
    this mode, but the rollout process still appends completed tasks to
    output/<model>_<alias>/averitec/trajectories.jsonl.
    """
    started_at = float(rec.get("started_at") or 0)
    save_alias = rec.get("save_alias") or None
    found = []
    for path in _rollout_trajectory_files(save_alias):
        try:
            with open(path, encoding="utf-8") as fh:
                for line_no, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    completed_at = _entry_completed_at(entry)
                    if completed_at is not None and completed_at < started_at - 5:
                        continue
                    # If a legacy row has no timestamp, only accept it from a file
                    # touched after this backend run started.
                    if completed_at is None:
                        try:
                            if os.path.getmtime(path) < started_at - 5:
                                continue
                        except OSError:
                            continue
                    info = _rollout_entry_info(path, line_no, entry)
                    found.append(info)
        except OSError:
            continue
    found.sort(key=lambda x: x.get("completed_at") or 0, reverse=True)
    return found


def _rollout_payload(sample_id):
    entry = _read_rollout_entry(sample_id)
    sample = entry.get("sample") or {}
    trajectory = (
        sample.get("trajectory") if isinstance(sample.get("trajectory"), list) else []
    )
    completed_at = _entry_completed_at(entry)
    completed_iso = entry.get("completed_iso") or _iso_from_ts(completed_at)

    lines = []
    for i, msg in enumerate(trajectory):
        turn = (
            dict(msg)
            if isinstance(msg, dict)
            else {"role": "unknown", "content": str(msg)}
        )
        turn.setdefault("role", "unknown")
        turn.setdefault("content", "")
        lines.append(
            {
                "recv_index": i,
                "recv_ts": completed_iso,
                "ingested": True,
                "is_message_end": True,
                "is_trajectory_end": False,
                "turn": turn,
            }
        )

    final_turn = max(0, len(lines) - 1)
    graph_problem_id = (
        sample.get("graph_problem_id") or entry.get("problem_id") or sample_id
    )
    graph = {
        "schema_version": 1,
        "problem_id": graph_problem_id,
        "stream_turn_index": final_turn,
        "n_turns_ingested": len(lines),
        "stage": "final",
        "finalized": True,
        "generated_at": completed_iso,
        "nodes": [],
        "relations": [],
        "evidence": [],
        "beliefs": [],
        "decisions": [],
    }
    return {"trajLines": lines, "graphLines": [graph], "eventLines": []}


def _find_timings(graph_problem_id):
    """Join a graph_problem_id to rollout agent-side per-step timings."""
    if graph_problem_id.startswith(ROLLOUT_SAMPLE_PREFIX):
        try:
            entry = _read_rollout_entry(graph_problem_id)
        except Exception:
            entry = None
        if entry:
            path, _line_no = _decode_rollout_sample_id(graph_problem_id)
            return {
                "found": True,
                "graph_problem_id": (entry.get("sample") or {}).get("graph_problem_id")
                or graph_problem_id,
                "alias": os.path.basename(os.path.dirname(os.path.dirname(path))),
                "elapsed_seconds": entry.get("elapsed_seconds"),
                "steps": _timing_steps_from_entry(entry),
            }
    if graph_problem_id in _TIMINGS_CACHE:
        return _TIMINGS_CACHE[graph_problem_id]
    result = {"found": False, "graph_problem_id": graph_problem_id, "steps": []}
    pat = os.path.join(CFG["rollout_output_dir"], "*", "averitec", "trajectories.jsonl")
    files = sorted(glob.glob(pat), key=lambda p: os.path.getmtime(p), reverse=True)
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if (
                        graph_problem_id not in line
                    ):  # cheap prefilter before json.loads
                        continue
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    s = o.get("sample") or {}
                    if s.get("graph_problem_id") != graph_problem_id:
                        continue
                    steps = _timing_steps_from_entry(o)
                    result = {
                        "found": True,
                        "graph_problem_id": graph_problem_id,
                        "alias": os.path.basename(os.path.dirname(os.path.dirname(f))),
                        "elapsed_seconds": o.get("elapsed_seconds"),
                        "steps": steps,
                    }
                    _TIMINGS_CACHE[graph_problem_id] = result
                    return result
        except OSError:
            continue
    # cache negatives only briefly? keep simple: don't cache misses (run may still finish)
    return result


def _log_tail(path, max_bytes=6000, max_lines=45):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except FileNotFoundError:
        return ""
    text = data.decode("utf-8", "replace")
    if len(data) >= max_bytes:
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _start_run(
    max_problems,
    extra_args,
    task_ids=None,
    save_alias=None,
    belief_graph_mode="augment",
):
    os.makedirs(RUNS_DIR, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    log_path = os.path.join(RUNS_DIR, run_id + ".log")

    inner = ["bash", CFG["start_sh"]]
    if task_ids:
        # --task-ids filters to exactly these claims (runner ignores --max-problems then)
        inner += ["--task-ids"] + [str(t) for t in task_ids]
    else:
        inner += ["--max-problems", str(int(max_problems))]
    if save_alias:
        # appended last so it overrides start.sh's hardcoded --save-alias
        inner += ["--save-alias", str(save_alias)]
    if extra_args:
        inner += list(extra_args)
    if belief_graph_mode:
        # appended last so it overrides start.sh's hardcoded --belief-graph-mode
        inner += ["--belief-graph-mode", str(belief_graph_mode)]
    if CFG["login_shell"]:
        # re-quote into a single -lc string so ~/.bashrc (conda etc.) is sourced
        import shlex

        cmd = ["bash", "-lc", " ".join(shlex.quote(a) for a in inner)]
    else:
        cmd = inner

    before = _list_sample_dirs()
    started_at = time.time()
    # The handle stays open while the child runs and _run_status closes it.
    logf = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
    logf.write(f"$ {' '.join(cmd)}\n(cwd={os.path.dirname(CFG['start_sh'])})\n\n")
    logf.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(CFG["start_sh"]) or None,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group so we can stop the tree
        env=os.environ.copy(),
    )
    with RUNS_LOCK:
        RUNS[run_id] = {
            "run_id": run_id,
            "proc": proc,
            "logf": logf,
            "log_path": log_path,
            "before": before,
            "started_at": started_at,
            "cmd": cmd,
            "max_problems": int(max_problems),
            "task_ids": list(task_ids or []),
            "save_alias": save_alias or "",
            "belief_graph_mode": belief_graph_mode,
        }
    return run_id


def _run_status(run_id):
    with RUNS_LOCK:
        rec = RUNS.get(run_id)
    if not rec:
        return None
    proc = rec["proc"]
    rc = proc.poll()
    if rc is None:
        stop_requested_at = rec.get("stop_requested_at")
        if stop_requested_at:
            status = "stopping"
            # If the process group ignores SIGTERM, escalate after a short grace
            # period. This happens on some shell/script trees.
            if time.time() - stop_requested_at > 5 and not rec.get("kill_sent"):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    with contextlib.suppress(Exception):
                        proc.kill()
                rec["kill_sent"] = True
        else:
            status = "running"
    elif rc == 0:
        status = "stopped" if rec.get("stop_requested_at") else "finished"
    else:
        status = "stopped" if rec.get("stop_requested_at") else "failed"
    if rc is not None and not rec.get("logf_closed"):
        with contextlib.suppress(Exception):
            rec["logf"].close()
        rec["logf_closed"] = True
    rollout_samples = (
        _new_rollout_samples(rec) if rec.get("belief_graph_mode") == "none" else []
    )
    stream_samples = (
        _new_samples(rec["before"]) if rec.get("belief_graph_mode") != "none" else []
    )
    sample_details = rollout_samples or [
        {"id": sid, "label": sid, "source": "stream"} for sid in stream_samples
    ]
    return {
        "run_id": run_id,
        "status": status,
        "returncode": rc,
        "elapsed": round(time.time() - rec["started_at"], 1),
        "new_samples": [s["id"] for s in sample_details],
        "new_sample_details": sample_details,
        "log_tail": _log_tail(rec["log_path"]),
        "max_problems": rec["max_problems"],
        "belief_graph_mode": rec.get("belief_graph_mode") or "augment",
    }


def _stop_run(run_id):
    with RUNS_LOCK:
        rec = RUNS.get(run_id)
    if not rec:
        return False
    proc = rec["proc"]
    rec["stop_requested_at"] = time.time()
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            with contextlib.suppress(Exception):
                proc.terminate()
    return True


# ---- HTTP handler ---------------------------------------------------------
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=CFG["root"], **kw)

    def log_message(self, fmt, *args):
        pass  # quiet

    # ---- helpers ----
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _download_file(self, path, filename=None):
        if not os.path.isfile(path):
            return self._json({"error": "file not found: " + path}, 404)
        filename = filename or os.path.basename(path)
        quoted = urllib.parse.quote(filename)
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.send_header(
            "Content-Disposition",
            f"attachment; filename=\"{filename}\"; filename*=UTF-8''{quoted}",
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _sample_result_path(self, sid):
        if sid.startswith(ROLLOUT_SAMPLE_PREFIX):
            return None
        parts = [seg for seg in sid.split("/") if seg not in ("", ".", "..")]
        if len(parts) != 1 or parts[0] != sid:
            return None
        return os.path.join(_current_stream_dir(), sid, "result.json")

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    # no-store for live-tailed data files
    def end_headers(self):
        p = self.path.split("?", 1)[0]
        if p.endswith(".jsonl") or p.endswith(".json") or p.endswith(".html"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # map /outputs_stream/... onto the current stream dir (which lives outside --root)
    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean == STREAM_URL_PREFIX or clean.startswith(STREAM_URL_PREFIX + "/"):
            rel = urllib.parse.unquote(clean[len(STREAM_URL_PREFIX) :])
            parts = [seg for seg in rel.split("/") if seg not in ("", ".", "..")]
            return os.path.join(_current_stream_dir(), *parts)
        return super().translate_path(path)

    # ---- routing ----
    def do_GET(self):
        p = self.path.split("?", 1)[0]
        # friendly alias: /demo (and /) serve the viewer page. Keep it slash-less
        # so the page's relative fetches (api/*, outputs_stream/) resolve to root.
        if p in ("/", "/demo", "/demo.html"):
            self.path = "/belief_graph_stream_viewer.html"
            return super().do_GET()
        if p == "/demo/":
            self.send_response(301)
            self.send_header("Location", "/demo")
            self.end_headers()
            return
        if p == "/api/ping":
            return self._json(
                {
                    "ok": True,
                    "run_enabled": os.path.exists(CFG["start_sh"]),
                    "start_sh": CFG["start_sh"],
                    "default_max_problems": CFG["default_max_problems"],
                    "default_save_alias": CFG["default_save_alias"],
                    "data_file": os.path.basename(CFG["data_file"]),
                    "has_problems": os.path.exists(CFG["data_file"]),
                    "stream_dir": _current_stream_dir(),
                }
            )
        if p.startswith("/api/result/"):
            sid = urllib.parse.unquote(p[len("/api/result/") :])
            result_path = self._sample_result_path(sid)
            if not result_path:
                return self._json({"error": "unsupported sample id"}, 404)
            return self._download_file(result_path, f"{sid}_result.json")
        if p == "/api/manifest":
            if not _MANIFEST:
                return self._json({"error": "build_stream_manifest.py not found"}, 500)
            try:
                stream_dir = _current_stream_dir()
                if not os.path.isdir(stream_dir):
                    return self._json(
                        {
                            "generated_at": datetime.now(UTC).isoformat(),
                            "count": 0,
                            "samples": [],
                        }
                    )
                m = _MANIFEST.build(stream_dir)
                return self._json(m)
            except Exception as e:
                return self._json({"error": str(e)}, 500)
        if p == "/api/problems":
            try:
                with open(CFG["data_file"], encoding="utf-8") as f:
                    rows = json.load(f)
            except Exception as e:
                return self._json({"error": str(e), "problems": []})
            probs = []
            if isinstance(rows, list):
                for i, row in enumerate(rows):
                    if not isinstance(row, dict) or not row.get("claim"):
                        continue
                    cid = str(row.get("claim_id") or row.get("id") or i)
                    probs.append(
                        {
                            "claim_id": cid,
                            "claim": row.get("claim", ""),
                            "label": row.get("label", ""),
                        }
                    )
            return self._json(
                {"data_file": os.path.basename(CFG["data_file"]), "problems": probs}
            )
        if p.startswith("/api/timings/"):
            sid = urllib.parse.unquote(p[len("/api/timings/") :])
            # 450_0_abcd -> 450:0:abcd, but do not rewrite synthetic rollout ids.
            gid = (
                sid if sid.startswith(ROLLOUT_SAMPLE_PREFIX) else sid.replace("_", ":")
            )
            return self._json(_find_timings(gid))
        if p.startswith("/api/sample/"):
            sid = urllib.parse.unquote(p[len("/api/sample/") :])
            if not sid.startswith(ROLLOUT_SAMPLE_PREFIX):
                return self._json({"error": "unsupported sample id"}, 404)
            try:
                return self._json(_rollout_payload(sid))
            except Exception as e:
                return self._json({"error": str(e)}, 404)
        if p.startswith("/api/run/"):
            run_id = p[len("/api/run/") :]
            st = _run_status(run_id)
            return self._json(
                st if st else {"error": "unknown run"}, 200 if st else 404
            )
        return super().do_GET()

    def do_POST(self):
        p = self.path.split("?", 1)[0]
        if p == "/api/run":
            if not os.path.exists(CFG["start_sh"]):
                return self._json(
                    {"error": "start.sh not found: " + CFG["start_sh"]}, 400
                )
            body = self._read_body()
            mp = body.get("max_problems") or CFG["default_max_problems"]
            extra = body.get("extra_args") or []
            if isinstance(extra, str):
                extra = extra.split()
            task_ids = body.get("task_ids") or (
                [str(body["claim_id"])] if body.get("claim_id") else []
            )
            if isinstance(task_ids, str):
                task_ids = [t for t in task_ids.replace(",", " ").split() if t]
            save_alias = (
                (body.get("save_alias") or "").strip()
                or CFG["default_save_alias"]
                or None
            )
            belief_graph_mode = str(body.get("belief_graph_mode") or "augment").strip()
            if belief_graph_mode not in BELIEF_GRAPH_MODES:
                return self._json(
                    {"error": "bad belief_graph_mode: " + belief_graph_mode}, 400
                )
            try:
                run_id = _start_run(mp, extra, task_ids, save_alias, belief_graph_mode)
            except Exception as e:
                return self._json({"error": str(e)}, 500)
            return self._json(
                {
                    "run_id": run_id,
                    "status": "running",
                    "task_ids": task_ids,
                    "save_alias": save_alias or "",
                    "belief_graph_mode": belief_graph_mode,
                }
            )
        if p.startswith("/api/run/") and p.endswith("/stop"):
            run_id = p[len("/api/run/") : -len("/stop")]
            ok = _stop_run(run_id)
            return self._json({"ok": ok}, 200 if ok else 404)
        self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--root", default=HERE)
    ap.add_argument(
        "--stream-dir",
        default=None,
        help="stream output root. Basenames like outputs_2026_7_6 auto-roll to "
        "today's outputs_Y_M_D; templates such as outputs_{Y}_{m}_{d} "
        "or outputs_{date} are also supported. Plain outputs_stream stays fixed.",
    )
    ap.add_argument("--start-sh", default=CFG["start_sh"])
    ap.add_argument(
        "--data-file",
        default=None,
        help="AVeriTeC data json used to list selectable cases "
        "(default: <repo>/datasets/sub_AVeriTeC/data/dev_subset10.json)",
    )
    ap.add_argument("--default-max-problems", type=int, default=1)
    ap.add_argument(
        "--default-save-alias",
        default="demo_test",
        help="save-alias prefilled in the UI / used when none is sent",
    )
    ap.add_argument(
        "--rollout-output",
        default=None,
        help="rollout output root with <alias>/averitec/trajectories.jsonl "
        "(agent-side per-step timings; default: <repo>/output)",
    )
    ap.add_argument("--login-shell", action="store_true")
    args = ap.parse_args()

    CFG["root"] = os.path.abspath(args.root)
    CFG["stream_dir_template"] = (
        os.path.abspath(args.stream_dir) if args.stream_dir else DEFAULT_STREAM_DIR
    )
    CFG["stream_dir"] = _current_stream_dir()
    CFG["start_sh"] = os.path.abspath(args.start_sh)
    repo = os.path.dirname(os.path.dirname(CFG["start_sh"]))
    CFG["data_file"] = (
        os.path.abspath(args.data_file)
        if args.data_file
        else os.path.join(repo, "datasets", "sub_AVeriTeC", "data", "dev_subset10.json")
    )
    CFG["rollout_output_dir"] = (
        os.path.abspath(args.rollout_output)
        if args.rollout_output
        else os.path.join(repo, "output")
    )
    CFG["default_max_problems"] = args.default_max_problems
    CFG["default_save_alias"] = args.default_save_alias
    CFG["login_shell"] = args.login_shell

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    page = "demo"  # /demo alias -> belief_graph_stream_viewer.html
    print(f"[serve_viewer] root       = {CFG['root']}")
    print(f"[serve_viewer] stream-tpl = {CFG['stream_dir_template']}")
    print(f"[serve_viewer] stream-dir = {_current_stream_dir()}")
    print(
        f"[serve_viewer] start.sh   = {CFG['start_sh']} "
        f"({'ok' if os.path.exists(CFG['start_sh']) else 'MISSING'})"
    )
    print(
        f"[serve_viewer] data-file  = {CFG['data_file']} "
        f"({'ok' if os.path.exists(CFG['data_file']) else 'MISSING'})"
    )
    print(f"[serve_viewer] bind       = {args.host}:{args.port}")
    if args.host in ("0.0.0.0", "::"):
        lan = _lan_ip()
        print(f"[serve_viewer] local      http://127.0.0.1:{args.port}/{page}")
        if lan:
            print(f"[serve_viewer] LAN        http://{lan}:{args.port}/{page}")
    else:
        print(f"[serve_viewer] open       http://{args.host}:{args.port}/{page}")
    print(f"[serve_viewer] Ctrl-C to stop  ·  {datetime.now(UTC).isoformat()}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve_viewer] shutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
